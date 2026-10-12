"""Fallback tests using a fake transport (no real API calls)."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main, router

OK = {"id": "x", "object": "chat.completion",
      "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}


def setup(monkeypatch, handler, keys=("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY")):
    for p in ["GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "CLOUDFLARE_API_TOKEN",
              "SAMBANOVA_API_KEY", "NVIDIA_API_KEY", "ZAI_API_KEY", "MISTRAL_API_KEY",
              "COHERE_API_KEY", "HF_TOKEN", "PROVIDER_ORDER"]:
        monkeypatch.delenv(p, raising=False)
    for k in keys:
        monkeypatch.setenv(k, "test-key")
    router.STATE.clear()
    main._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TestClient(main.app)


def host(req):
    return req.url.host


def test_first_provider_used(monkeypatch):
    c = setup(monkeypatch, lambda r: httpx.Response(200, json=OK))
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "gemini"


def test_daily_limit_switches_and_cools_down(monkeypatch):
    calls = []

    def handler(req):
        calls.append(host(req))
        if "googleapis" in host(req):
            return httpx.Response(429, json={"error": {"message": "Quota exceeded: requests per day"}})
        return httpx.Response(200, json=OK)

    c = setup(monkeypatch, handler)
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.headers["x-gateway-provider"] == "groq"
    st = {p["provider"]: p for p in c.get("/status").json()["providers"]}
    assert st["gemini"]["available"] is False
    assert st["gemini"]["last_error"] == "daily limit reached"
    assert st["gemini"]["cooldown_seconds_left"] > 3000
    # second request skips gemini entirely
    calls.clear()
    c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert calls == ["api.groq.com"]


def test_minute_limit_short_cooldown(monkeypatch):
    def handler(req):
        if "googleapis" in host(req):
            return httpx.Response(429, headers={"retry-after": "20"}, json={"error": "slow down"})
        return httpx.Response(200, json=OK)

    c = setup(monkeypatch, handler)
    c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    gem = c.get("/status").json()["providers"][0]
    assert 0 < gem["cooldown_seconds_left"] <= 60


def test_all_fail_returns_502(monkeypatch):
    c = setup(monkeypatch, lambda r: httpx.Response(500, text="boom"))
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert len(r.json()["error"]["attempts"]) == 3
    # now all on cooldown -> 503
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503


def test_no_keys_returns_503(monkeypatch):
    c = setup(monkeypatch, lambda r: httpx.Response(200, json=OK), keys=())
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503


def test_pin_provider_and_model(monkeypatch):
    seen = {}

    def handler(req):
        seen["host"] = host(req)
        seen["model"] = json.loads(req.content)["model"]
        return httpx.Response(200, json=OK)

    c = setup(monkeypatch, handler)
    r = c.post("/v1/chat/completions",
               json={"model": "groq:llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "hi"}]})
    assert r.headers["x-gateway-provider"] == "groq"
    assert seen["model"] == "llama-3.3-70b-versatile"


def test_streaming_fallback(monkeypatch):
    sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    def handler(req):
        if "googleapis" in host(req):
            return httpx.Response(402, json={"error": "no credits"})
        async def gen():
            yield sse  # real streaming body, like a live provider
        return httpx.Response(200, content=gen(), headers={"content-type": "text/event-stream"})

    c = setup(monkeypatch, handler)
    r = c.post("/v1/chat/completions",
               json={"stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.headers["x-gateway-provider"] == "groq"
    assert b"[DONE]" in r.content


def test_gateway_key(monkeypatch):
    c = setup(monkeypatch, lambda r: httpx.Response(200, json=OK))
    monkeypatch.setattr(main, "GATEWAY_API_KEY", "secret")
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert c.post("/v1/chat/completions", json=body).status_code == 401
    ok = c.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_provider_order_env(monkeypatch):
    c = setup(monkeypatch, lambda r: httpx.Response(200, json=OK))
    monkeypatch.setenv("PROVIDER_ORDER", "openrouter,groq")
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.headers["x-gateway-provider"] == "openrouter"
