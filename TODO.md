# TODO

## finish_reason normalization (truncation signal)

Backends in the field (Lemonade/llama.cpp, on-prem vLLM gateway) return
`finish_reason: "stop"` even when the completion budget was exhausted —
the spec says `"length"`. An IDE checking only `finish_reason` misses
truncation.

**Proposed fix (router-side):** in non-stream responses (where `usage` is
always available), rewrite `finish_reason` `stop` → `length` when
`usage.completion_tokens >=` the requested budget
(`max_tokens` / `max_completion_tokens`). Safe to stack with upstream
compliance (only rewrites `stop`).

Open questions:
- streaming: usage isn't guaranteed in stream chunks (OpenAI requires
  `stream_options: {"include_usage": true}`; some servers send it anyway).
  Decide: also send `stream_options` and normalize the final chunk, or
  leave streaming to IDE-side `usage` heuristics.
- reasoning models: budget exhausted by reasoning tokens with empty
  content should probably also surface as `length`
  (`reasoning_tokens == completion_tokens == budget`).

**Fallback (IDE-side, works today):** treat as truncated when
`usage.completion_tokens >= budget`, or content empty with reasoning
tokens present — regardless of `finish_reason`.

## Triage tuning

- Current local setup: Qwen3 0.6B on CPU (llama.cpp :8080, 4 threads, 4k
  ctx) with `timeout_s: 5` — the original 2s budget was too tight for
  4000-char prefills on CPU.
- The classifier prompt now asks for a one-line `tier | brief decision
  summary` (2026-11-12, for debug logging). Re-verify the "do NOT shorten"
  finding below with the new format on the 7B/NPU models.
- Externalize the default triage system prompt: ship it as a packaged text
  file, add an optional `routing.triage.prompt_file` (resolved relative to
  the YAML config), read once at config load. Keep inline `prompt`; reject
  both set at once; fail clearly on a missing/empty file. Editing the file
  requires a restart (no file watching).
- Profile triage latency on NPU under concurrent load (single call ~1s,
  queued calls up to ~9s observed). `triage.timeout_s` is 10–20 in test
  configs; the original plan assumed 2s.
- Consider `Qwen2.5-Coder-1.5B-Instruct-NPU` as the triage model (faster,
  likely sufficient for a 3-way split) vs the 7B.
- Classifier prompt: do NOT shorten it — with a stripped-down prompt the
  7B ignored the task (answered the example instead of the tier word);
  with the full prompt it was 6/6 accurate.

## Copilot

- Copilot sign-in works end-to-end (`meatproxy copilot login`,
  device-code flow; credentials land in the copilot home the server
  uses). Live test of the Copilot adapter through a running proxy is
  done (2026-11-12): non-streaming, streaming and multi-turn
  conversation requests all route to Copilot correctly; found and
  fixed a stream stall where end-of-turn events never woke a consumer
  blocked on the event queue.
- `copilot login --token`: verify which token kinds the runtime accepts
  (classic PATs unverified).
- Verify session resume across evictions/restarts on the real runtime.
- Consider: forward IDE conversation state into Copilot sessions on cold
  resume for *text-only* turns (still relies on Copilot's own persisted
  session state in `base_directory`). Tool-bearing turns already replay
  the full transcript into a fresh session via the terminal-tool bridge
  (`copilot_tools`).

## Misc

- NPU context limit (32768) is too small for agentic IDE traffic: opencode's
  system prompt + tools + history reaches ~34K tokens, so every agentic
  request fails over from npu-7b to on-prem (works, ~100ms overhead). The
  NPU still serves triage (short prompt) and small non-agentic requests.
  Consider a `context_window` model config option so the router can skip
  an over-context model proactively (estimate tokens from chars+tools)
  instead of paying the failed attempt each time.
- WSL2 note: NPU models are only servable in mirror networking mode
  (NAT mode → `model_not_supported: Requires Windows`).
- `model` field in upstream responses can be an opaque hash (Lemonade
  ryzenai models) — already handled by model-name rewriting, no action.
- Docker image: consider distroless/non-root polish + healthcheck
  (`CMD curl -f http://localhost:8700/health` or wget equivalent).
- Config: `api.port` exists; document `MEATPROXY_PORT` override precedence.
