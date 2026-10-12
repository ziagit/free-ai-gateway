"""Provider registry.

Every provider here speaks the OpenAI chat-completions format, so the gateway
can forward the same request body to any of them. Order = priority.

Each provider is enabled only if its API key env var is set. Override the
model per provider with <NAME>_MODEL (e.g. GROQ_MODEL=llama-3.3-70b-versatile)
and the order with PROVIDER_ORDER=groq,gemini,...
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Provider:
    name: str
    base_url: str
    key_env: str
    default_model: str
    extra_headers: dict = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.key_env) or None

    @property
    def model(self) -> str:
        return os.getenv(f"{self.name.upper()}_MODEL", self.default_model)

    @property
    def url(self) -> str:
        base = self.base_url
        if "{CLOUDFLARE_ACCOUNT_ID}" in base:
            base = base.replace("{CLOUDFLARE_ACCOUNT_ID}", os.getenv("CLOUDFLARE_ACCOUNT_ID", ""))
        return base.rstrip("/") + "/chat/completions"

    @property
    def enabled(self) -> bool:
        if not self.api_key:
            return False
        if "{CLOUDFLARE_ACCOUNT_ID}" in self.base_url and not os.getenv("CLOUDFLARE_ACCOUNT_ID"):
            return False
        return True


# Default models were current in late 2026; change them with <NAME>_MODEL env vars
# if a provider renames or retires one.
ALL_PROVIDERS: list[Provider] = [
    Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
             "GEMINI_API_KEY", "gemini-2.5-flash"),
    Provider("groq", "https://api.groq.com/openai/v1",
             "GROQ_API_KEY", "openai/gpt-oss-120b"),
    Provider("openrouter", "https://openrouter.ai/api/v1",
             "OPENROUTER_API_KEY", "openai/gpt-oss-20b:free",
             {"HTTP-Referer": "https://github.com/free-ai-gateway", "X-Title": "free-ai-gateway"}),
    Provider("cloudflare", "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
             "CLOUDFLARE_API_TOKEN", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
    Provider("sambanova", "https://api.sambanova.ai/v1",
             "SAMBANOVA_API_KEY", "Meta-Llama-3.3-70B-Instruct"),
    Provider("nvidia", "https://integrate.api.nvidia.com/v1",
             "NVIDIA_API_KEY", "meta/llama-3.3-70b-instruct"),
    Provider("zai", "https://api.z.ai/api/paas/v4",
             "ZAI_API_KEY", "glm-4.5-flash"),
    Provider("mistral", "https://api.mistral.ai/v1",
             "MISTRAL_API_KEY", "mistral-small-latest"),
    Provider("cohere", "https://api.cohere.ai/compatibility/v1",
             "COHERE_API_KEY", "command-a-03-2025"),
    Provider("huggingface", "https://router.huggingface.co/v1",
             "HF_TOKEN", "openai/gpt-oss-120b"),
]


def ordered_providers() -> list[Provider]:
    """Providers in priority order (PROVIDER_ORDER env overrides)."""
    order = os.getenv("PROVIDER_ORDER")
    if not order:
        return list(ALL_PROVIDERS)
    by_name = {p.name: p for p in ALL_PROVIDERS}
    names = [n.strip().lower() for n in order.split(",") if n.strip()]
    chosen = [by_name[n] for n in names if n in by_name]
    # anything not listed goes to the end, in default order
    chosen += [p for p in ALL_PROVIDERS if p.name not in names]
    return chosen
