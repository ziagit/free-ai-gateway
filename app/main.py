"""Free AI Gateway - one OpenAI-compatible endpoint, many free providers.

POST /v1/chat/completions  -> tries providers in order, switches on limits/errors
GET  /v1/models            -> 'auto' plus one entry per enabled provider
GET  /status               -> live provider status / cooldowns
"""
from __future__ import annotations

import json
import os
import secrets
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

load_dotenv()

from . import router  # noqa: E402  (needs env loaded first)
from .providers import ordered_providers  # noqa: E402

GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")


@asynccontextmanager
async def lifespan(_app):
    yield
    if _client is not None:
        await _client.aclose()


app = FastAPI(title="Free AI Gateway", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=router.TIMEOUT)
    return _client


def check_auth(request: Request) -> None:
    if not GATEWAY_API_KEY:
        return
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    if not secrets.compare_digest(token, GATEWAY_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing gateway API key")


@app.get("/")
async def root():
    return {"name": "Free AI Gateway", "endpoint": "/v1/chat/completions", "status": "/status"}


@app.get("/status")
async def status(request: Request):
    check_auth(request)
    return {"providers": router.status_report()}


@app.get("/v1/models")
async def models(request: Request):
    check_auth(request)
    data = [{"id": "auto", "object": "model", "owned_by": "gateway"}]
    for p in ordered_providers():
        if p.enabled:
            data.append({"id": p.name, "object": "model", "owned_by": p.name, "default_model": p.model})
    return {"object": "list", "data": data}


@app.get("/v1/providers/{name}/models")
async def provider_models(name: str, request: Request):
    """List the models a provider offers for your key (to pick <NAME>_MODEL values)."""
    check_auth(request)
    p = next((x for x in ordered_providers() if x.name == name.lower()), None)
    if p is None or not p.enabled:
        raise HTTPException(404, f"Provider '{name}' not found or has no API key set")
    url = p.url.rsplit("/chat/completions", 1)[0] + "/models"
    r = await client().get(url, headers={"Authorization": f"Bearer {p.api_key}", **p.extra_headers})
    try:
        body = r.json()
    except ValueError:
        body = {"raw": r.text[:2000]}
    if r.status_code != 200:
        return JSONResponse(status_code=r.status_code, content=body)
    items = body.get("data", body.get("models", [])) if isinstance(body, dict) else body
    ids = [m.get("id") or m.get("name") for m in items if isinstance(m, dict)]
    return {"provider": p.name, "current_model": p.model, "models": sorted(i for i in ids if i)}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    check_auth(request)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(400, "Body must be JSON")
    if not body.get("messages"):
        raise HTTPException(400, "'messages' is required")

    preferred, model_override = router.parse_model(body.get("model"))
    provs = router.candidates(preferred)
    if not provs:
        return JSONResponse(status_code=503, content={"error": {
            "message": "No provider available: all are on cooldown or have no API key set. See /status.",
            "providers": router.status_report()}})

    stream = bool(body.get("stream"))
    attempts = []

    for p in provs:
        override = model_override if p.name == preferred else None
        payload, headers = router.build_request(p, body, override)
        try:
            req = client().build_request("POST", p.url, json=payload, headers=headers)
            resp = await client().send(req, stream=stream)
        except httpx.HTTPError as e:
            reason = router.mark_fail(p, None, None, str(e))
            attempts.append({"provider": p.name, "error": reason})
            continue

        if resp.status_code != 200:
            text = (await resp.aread()).decode("utf-8", "replace")
            await resp.aclose()
            reason = router.mark_fail(p, resp.status_code, resp.headers, text)
            attempts.append({"provider": p.name, "status": resp.status_code,
                             "error": reason, "detail": text[:300]})
            continue

        router.mark_ok(p)
        out_headers = {"X-Gateway-Provider": p.name, "X-Gateway-Model": payload["model"]}
        if attempts:
            out_headers["X-Gateway-Fallbacks"] = ",".join(a["provider"] for a in attempts)

        if stream:
            async def relay(r=resp):
                try:
                    async for chunk in r.aiter_raw():
                        yield chunk
                finally:
                    await r.aclose()
            return StreamingResponse(relay(), media_type="text/event-stream", headers=out_headers)

        data = resp.json()
        data["provider"] = p.name
        if attempts:
            data["gateway_fallbacks"] = attempts  # providers that failed before this one
        return JSONResponse(content=data, headers=out_headers)

    return JSONResponse(status_code=502, content={"error": {
        "message": "All providers failed for this request.", "attempts": attempts}})
