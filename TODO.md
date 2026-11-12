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

## NPU triage tuning

- Profile triage latency on NPU under concurrent load (single call ~1s,
  queued calls up to ~9s observed). `triage.timeout_s` is 10–20 in test
  configs; the original plan assumed 2s.
- Consider `Qwen2.5-Coder-1.5B-Instruct-NPU` as the triage model (faster,
  likely sufficient for a 3-way split) vs the 7B.
- Classifier prompt: do NOT shorten it — with a stripped-down prompt the
  7B ignored the task (answered the example instead of the tier word);
  with the full prompt it was 6/6 accurate.

## Copilot

- Live test of the Copilot adapter still pending (needs a GitHub
  Copilot subscription + `copilot login` or token; runtime is already
  baked into the Docker image).
- Verify session resume across evictions/restarts on the real runtime.
- Consider: forward IDE conversation state into Copilot sessions on cold
  resume (currently relies on Copilot's own persisted session state in
  `base_directory`).

## Misc

- WSL2 note: NPU models are only servable in mirror networking mode
  (NAT mode → `model_not_supported: Requires Windows`).
- `model` field in upstream responses can be an opaque hash (Lemonade
  ryzenai models) — already handled by model-name rewriting, no action.
- Docker image: consider distroless/non-root polish + healthcheck
  (`CMD curl -f http://localhost:8700/health` or wget equivalent).
- Config: `api.port` exists; document `MEATPROXY_PORT` override precedence.
