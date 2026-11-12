# meatproxy

A small, unbloated LLM router in Python. An IDE points its OpenAI client at
meatproxy; meatproxy uses a **small local model on your NPU** to triage each
prompt and routes it to the right backend — a local NPU model, an on-prem
model, or the cloud (GitHub Copilot in *Auto* mode).

No agent loop, no RAG, no vector store, no web UI, no plugin system. One
process, one YAML file, one job.

## Quick start

The minimal setup is Copilot-only: it works on any machine with a GitHub
Copilot subscription and no local model.

1. **Install** (Python 3.11+):

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e .
   ```

2. **Sign in to GitHub Copilot** (device-code flow: prints a URL + code
   to enter in the browser):

   ```bash
   .venv/bin/meatproxy copilot login
   ```

   The Copilot CLI runtime is downloaded automatically on first use.

3. **Write a config** — `meatproxy.yaml` in the working directory, or
   pass `--config` / `MEATPROXY_CONFIG`:

   ```yaml
   api:
     host: 127.0.0.1
     port: 8700

   models:
     copilot:
       tier: heavy
       kind: copilot

   routing:
     chains:
       heavy: [copilot]

   conversations:
     db_path: meatproxy.db
   ```

   With no triage model, `model: "auto"` goes straight to the heavy
   chain. To add local models (NPU/GPU) or triage, copy one of the
   examples in `config/` instead — they use Docker-style `/data/...`
   paths, so adjust `conversations.db_path` and `copilot.base_directory`
   when running outside Docker (omitting `base_directory` uses
   `~/.copilot`, where `copilot login` stores credentials by default).

4. **Run** it:

   ```bash
   MEATPROXY_API_KEY=... .venv/bin/meatproxy
   ```

5. **Point your IDE at it** — the OpenAI-compatible API at
   `http://127.0.0.1:8700/v1`, the API key as bearer token, model
   `auto` (or `copilot`). Sanity check:

   ```bash
   curl -s http://127.0.0.1:8700/health
   ```

## How it works

```
IDE ──OpenAI API (SSE)──▶ meatproxy (FastAPI, Docker, SQLite)
                             │  auth: Bearer $MEATPROXY_API_KEY
                             │  triage: small NPU model (Lemonade OpenAI API),
                             │          2s budget, cached by prompt hash
                             │          on failure → routes to on_failure model
                             │  routing: model="auto" → triage → tier
                             │           explicit model → direct
                             │  failover: escalates UP light → mid → heavy
                             ▼
        ┌─────────────────────┬────────────────────────────┬──────────────────
        │ light/mid: generic  │ heavy: Copilot adapter     │ triage: NPU model
        │ OpenAI adapters     │ (github-copilot-sdk, one   │ (same OpenAI
        │ (Lemonade NPU,      │  CLI runtime process,      │  adapter)
        │  vLLM/GPU)          │  model="auto")
        └─────────────────────┴────────────────────────────┴──────────────────
        tools pass through verbatim on OpenAI backends; Copilot path is
        text-only (it is itself an agent, so `tools` are dropped there)
```

- **`model: "auto"`** (the default) runs triage on the NPU and picks a tier:
  `light`, `mid` or `heavy`.
- **Any explicit model name** bypasses triage and hits that model directly
  (with its chain as fallback).
- If a model fails **before its first token**, the next model in the chain
  takes over. Once tokens are flowing, the stream is committed.
- **Conversations**: send an `X-Conversation-Id` header and only the newest
  user message; meatproxy stores history in SQLite and rebuilds context for
  stateless backends. Copilot conversations map to persistent Copilot
  sessions (state in the mounted `base_directory`), pooled with an idle
  timeout and resumed across evictions.

## Configuration (YAML)

See `config/example.npu-only.yaml`, `config/example.npu-gpu.yaml` and
`config/example.no-npu.yaml`. Secrets are referenced by env-var name, never
stored in the file:

```yaml
[models]
npu-small = { tier = "light", endpoint = "http://npu-host:11434/v1", model = "fastflow-small" }
onprem-vllm = { tier = "mid", endpoint = "http://gpu-box:8000/v1", model = "qwen2.5-coder-32b",
                api_key_env = "ONPREM_API_KEY" }        # optional, per model
copilot = { tier = "heavy", kind = "copilot" }

[routing]
  [routing.triage]
  model = "npu-small"     # the model that classifies
  timeout_s = 2           # budget; on failure → on_failure
  on_failure = "copilot"
  cache_ttl_s = 300

  [routing.chains]        # fallback order per triage tier (escalates up)
  light = ["npu-small", "onprem-vllm", "copilot"]
  mid = ["onprem-vllm", "copilot"]
  heavy = ["copilot", "onprem-vllm"]

[conversations]
db_path = "/data/meatproxy.db"
max_context_messages = 40
copilot_idle_timeout_s = 900

[copilot]
base_directory = "/data/copilot"   # where Copilot session state + login live
auto_tier = ""                     # optional: efficiency | balance | intelligence | fast
timeout_s = 60
```

The config file is **hot-reloaded** when its mtime changes; an invalid new
file is rejected (logged once per file version) and the old config keeps
serving.

## API

| Endpoint | Description |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-compatible chat; JSON by default, `stream: true` → SSE |
| `GET /v1/models` | `auto` + all configured models |
| `GET /v1/conversations` | list stored conversations |
| `DELETE /v1/conversations/{id}` | delete a conversation |
| `GET /health` | backend reachability |
| `GET /metrics` | Prometheus metrics |

TLS to on-prem backends uses the default (certifi) CA store. For gateways
behind an internal CA, set `ca_bundle: /path/to/ca-bundle.crt` on the model
(the whole endpoint uses it); `ssl_insecure: true` disables verification
entirely (loudly discouraged).

All `/v1/*` endpoints require `Authorization: Bearer $MEATPROXY_API_KEY`.

```bash
curl -N http://localhost:8700/v1/chat/completions \
  -H "Authorization: Bearer $MEATPROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Conversation-Id: my-ide-session" \
  -d '{"model": "auto", "stream": true, "messages": [{"role": "user", "content": "hi"}]}'
```

## Running with Docker

```bash
docker run -d --name meatproxy \
  -p 8700:8700 \
  -e MEATPROXY_API_KEY=... \
  -v $(pwd)/meatproxy.yaml:/meatproxy.yaml:ro \
  -v meatproxy-data:/data \
  ghcr.io/your-org/meatproxy
```

`/data` holds the SQLite DB and the Copilot session state.

### Copilot auth

meatproxy signs in to GitHub Copilot through the
[GitHub Copilot SDK](https://github.com/github/copilot-sdk) (Python).
Sign in once; a running proxy picks the credentials up from the copilot
`base_directory` with no restart or config change:

```bash
meatproxy copilot login        # device-code flow (prints a URL + code)
meatproxy copilot status       # show the sign-in state
meatproxy copilot logout       # sign out
```

`login` defaults to the device-code flow (works from any terminal,
including containers); `--web` uses the browser variant, and
`--token <token>` / `--token-stdin` install a GitHub token instead of
interactive sign-in. In a container there is no loopback browser or
keychain, so use `--yes` (accepts plaintext token storage, no
interactive prompts):

```bash
docker run -it --rm <image> copilot login --yes
```

The token alternative for the server remains: set an env var with a
GitHub token and point `[copilot] token_env` at it, e.g.
`token_env = "COPILOT_GITHUB_TOKEN"`.

The Copilot CLI runtime is downloaded at image build time
(`python -m copilot download-runtime`), so the image is self-contained.

## CLI

```
meatproxy [serve]             run the proxy (default)
meatproxy copilot login       sign in to GitHub Copilot (--web, --token, --token-stdin)
meatproxy copilot status      show the Copilot sign-in state
meatproxy copilot logout      sign out
meatproxy config check        validate the config file
meatproxy version             print the version
```

Flags override the `MEATPROXY_*` environment variables. With no
subcommand the proxy runs, so `docker run <image>` and
`python -m meatproxy` keep working unchanged.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

Run locally as in the [Quick start](#quick-start).

## Metrics

`meatproxy_requests_total{model,tier,source}`,
`meatproxy_failovers_total{from_model,to_model}`,
`meatproxy_triage_latency_seconds`,
`meatproxy_triage_decisions_total{tier,cached}`,
`meatproxy_conversations_active`.
