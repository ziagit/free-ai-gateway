"""Fallback router: try providers in order, skip ones that are cooling down,
and put a provider on cooldown when it hits a rate/daily limit or fails."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from .providers import Provider, ordered_providers

TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
MINUTE_COOLDOWN = int(os.getenv("MINUTE_COOLDOWN", "60"))      # per-minute 429
DAILY_COOLDOWN = int(os.getenv("DAILY_COOLDOWN", "3600"))      # daily quota / no credits
ERROR_COOLDOWN = int(os.getenv("ERROR_COOLDOWN", "30"))        # 5xx / network errors
AUTH_COOLDOWN = int(os.getenv("AUTH_COOLDOWN", "21600"))       # bad key (6h)

DAILY_HINTS = ("per day", "daily", "per_day", "perday", "quota", "exceeded your current",
               "credits", "insufficient", "limit reached for today", "rpd", "tpd")


@dataclass
class State:
    cooldown_until: float = 0.0
    reason: str = ""
    successes: int = 0
    failures: int = 0
    last_used: float = 0.0


STATE: dict[str, State] = {}


def state(name: str) -> State:
    return STATE.setdefault(name, State())


def classify(status: int | None, headers: httpx.Headers | None, body: str) -> tuple[int, str]:
    """Return (cooldown_seconds, reason) for a failed attempt."""
    text = body.lower()[:2000]
    if status is None:
        return ERROR_COOLDOWN, "network error / timeout"
    if status == 429:
        retry_after = None
        if headers is not None:
            try:
                retry_after = int(float(headers.get("retry-after", "")))
            except ValueError:
                retry_after = None
        if any(h in text for h in DAILY_HINTS) or (retry_after and retry_after > 300):
            return max(retry_after or 0, DAILY_COOLDOWN), "daily limit reached"
        return max(retry_after or 0, MINUTE_COOLDOWN), "rate limited (per-minute)"
    if status == 402:
        return DAILY_COOLDOWN, "out of free credits"
    if status in (401, 403):
        return AUTH_COOLDOWN, f"auth error {status} (check API key)"
    if status >= 500:
        return ERROR_COOLDOWN, f"server error {status}"
    # 400/404/422 etc.: usually model name or unsupported param on this provider
    return 0, f"client error {status}"


def parse_model(requested: str | None) -> tuple[str | None, str | None]:
    """'auto' -> (None, None); 'groq' -> ('groq', None); 'groq:llama-3.3' -> ('groq','llama-3.3')."""
    if not requested or requested == "auto":
        return None, None
    names = {p.name for p in ordered_providers()}
    if ":" in requested:
        prov, model = requested.split(":", 1)
        if prov.lower() in names:
            return prov.lower(), model
    if requested.lower() in names:
        return requested.lower(), None
    return None, None


def candidates(preferred: str | None) -> list[Provider]:
    now = time.time()
    provs = [p for p in ordered_providers() if p.enabled]
    if preferred:
        provs.sort(key=lambda p: 0 if p.name == preferred else 1)
    ready = [p for p in provs if state(p.name).cooldown_until <= now]
    return ready


def build_request(p: Provider, body: dict, model_override: str | None) -> tuple[dict, dict]:
    payload = dict(body)
    payload["model"] = model_override or p.model
    headers = {
        "Authorization": f"Bearer {p.api_key}",
        "Content-Type": "application/json",
        **p.extra_headers,
    }
    return payload, headers


def mark_fail(p: Provider, status, headers, text) -> str:
    cooldown, reason = classify(status, headers, text)
    s = state(p.name)
    s.failures += 1
    s.reason = reason
    if cooldown:
        s.cooldown_until = time.time() + cooldown
    return reason


def mark_ok(p: Provider) -> None:
    s = state(p.name)
    s.successes += 1
    s.last_used = time.time()
    s.reason = ""


def status_report() -> list[dict]:
    now = time.time()
    out = []
    for p in ordered_providers():
        s = state(p.name)
        out.append({
            "provider": p.name,
            "enabled": p.enabled,
            "model": p.model,
            "available": p.enabled and s.cooldown_until <= now,
            "cooldown_seconds_left": max(0, int(s.cooldown_until - now)),
            "last_error": s.reason or None,
            "successes": s.successes,
            "failures": s.failures,
        })
    return out
