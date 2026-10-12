# Free AI Gateway

One API endpoint that uses **10 free AI providers**. When one hits its daily or per-minute limit, the gateway automatically switches to the next one.

It speaks the **OpenAI format**, so any OpenAI SDK, LangChain, n8n, etc. works by changing the base URL.

```
Your app ──► /v1/chat/completions ──► Gemini ──(limit hit)──► Groq ──► OpenRouter ──► Cloudflare
                                      ──► SambaNova ──► NVIDIA ──► Z.ai ──► Mistral ──► Cohere ──► Hugging Face
```

## How switching works

| Provider response | What the gateway does |
|---|---|
| `429` daily quota ("per day", "quota", "credits"…) | Skips that provider for **1 hour**, then tries it again (so it picks it up after the daily reset) |
| `429` per-minute limit | Skips it for **60 s** (or the provider's `Retry-After`) |
| `402` out of credits | Skips it for 1 hour |
| `401/403` bad key | Skips it for 6 hours |
| `5xx` / timeout | Skips it for 30 s |
| `400/404` (model/param issue) | Tries the next one, no cooldown |

Providers with no API key are skipped. You don't need all 10 — even 2–3 keys work.

## Quick start (local)

```bash
git clone https://github.com/ziagit/free-ai-gateway.git
cd free-ai-gateway
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then paste your keys into .env
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/status to see which providers are active.

## Call it

**curl**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello!"}]}'
```

**Python (OpenAI SDK)**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_GATEWAY_API_KEY")
r = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

**JavaScript**
```js
const r = await fetch("http://localhost:8000/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json", Authorization: "Bearer YOUR_GATEWAY_API_KEY" },
  body: JSON.stringify({ model: "auto", messages: [{ role: "user", content: "Hello!" }] }),
});
console.log((await r.json()).choices[0].message.content);
```

Streaming (`"stream": true`) works too.

The response header `X-Gateway-Provider` (and the `provider` field in the JSON) tells you which provider answered.

## Choosing the model

| `model` value | Meaning |
|---|---|
| `auto` (or empty) | Normal fallback order, each provider's default model |
| `groq` | Try Groq first, then fall back to the others |
| `groq:llama-3.3-70b-versatile` | Try Groq with that exact model first, then fall back |

Change default models with env vars like `GEMINI_MODEL=...`, and the order with `PROVIDER_ORDER=groq,gemini,...`.

## Endpoints

| Method | Path | |
|---|---|---|
| POST | `/v1/chat/completions` | Chat (OpenAI format) |
| GET | `/v1/models` | `auto` + enabled providers |
| GET | `/status` | Live status, cooldowns, success/failure counts |

## Providers and keys

| # | Provider | Env var | Free tier | Get key |
|---|---|---|---|---|
| 1 | Google Gemini | `GEMINI_API_KEY` | Daily quota (Flash models) | aistudio.google.com/apikey |
| 2 | Groq | `GROQ_API_KEY` | ~1,000 req / 200K tokens per day per model | console.groq.com/keys |
| 3 | OpenRouter | `OPENROUTER_API_KEY` | 50 req/day on `:free` models | openrouter.ai/keys |
| 4 | Cloudflare Workers AI | `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN` | 10,000 Neurons/day | dash.cloudflare.com |
| 5 | SambaNova | `SAMBANOVA_API_KEY` | 20 req/day per model | cloud.sambanova.ai |
| 6 | NVIDIA NIM | `NVIDIA_API_KEY` | ~40 req/min on free endpoints | build.nvidia.com |
| 7 | Z.ai (GLM) | `ZAI_API_KEY` | GLM-Flash models free | z.ai |
| 8 | Mistral | `MISTRAL_API_KEY` | ~$10/month credit | console.mistral.ai |
| 9 | Cohere | `COHERE_API_KEY` | 1,000 calls/month | dashboard.cohere.com |
| 10 | Hugging Face | `HF_TOKEN` | $0.10/month credit | huggingface.co/settings/tokens |

Free-tier limits change often — check each provider's console. If a default model is retired, set `<NAME>_MODEL`.

## Deploy for free

**Render** (easiest): New → Blueprint → pick this repo. `render.yaml` is included; paste your keys when asked. Free instances sleep after inactivity, so the first request can take ~30 s.

**Docker** (any VPS):
```bash
docker build -t free-ai-gateway .
docker run -d -p 8000:8000 --env-file .env free-ai-gateway
```

Always set `GATEWAY_API_KEY` when deployed publicly, otherwise anyone can use up your quotas.

## Notes

- Cooldown state is in memory; a restart resets it (harmless — a limited provider just gets one retry).
- Run with **one worker** so all requests share the same cooldown state.
- Gemini's free tier may use your prompts to improve Google's products — don't send private data through it.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
