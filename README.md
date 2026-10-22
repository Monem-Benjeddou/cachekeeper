# cachekeeper

Shows exactly why your agent's prompt cache keeps missing on llama.cpp, Ollama and LM Studio.

Local engines reuse the KV cache only for the exact matching start of the prompt. One timestamp,
nonce or reordered tool list near the top of the prompt, and every turn pays the full prefill again.
On a laptop that is the difference between a quarter of a second and 25 seconds, every turn:

![Prefill time per request: 25.13 s cold, 0.06 s warm, about 25 s on every turn once a timestamp is in the system prompt](docs/images/prefill.svg)

Measured with `cachekeeper probe` on an M1 Pro, qwen3:4b, Ollama 0.30.10 and a ~6.3k-token system
prompt. Run it on your own machine to get your numbers.

## What it does

cachekeeper is a local proxy you put between your agent and the engine. In this first version it
only watches: it forwards your request bytes unchanged and tells you, per request, whether the cache
was reused, where the prompt stopped matching, and why.

![How cachekeeper sees a request: agent, proxy, engine; divergence detection, backend stats, status line, SQLite](docs/images/how-it-works.svg)

1. It turns each request into one string: system messages, then tools, then history.
2. It finds the closest of the last 8 prompts for the same model (longest common prefix), so two
   agents sharing one proxy don't confuse each other.
3. It records where the two prompts diverge, and the line before and after.
4. It names the cause: ISO datetime, Unix epoch, UUID, hex nonce, reordered tools, dropped leading
   history, summarized history, a block re-sent after the history, whitespace drift, or a different
   system prompt (see [docs/causes.md](docs/causes.md)).
5. It reads the engine's own answer from the response: llama.cpp counts reused tokens exactly; Ollama
   and LM Studio only allow an estimate from timing, and the output says so.
6. It prints one status line per request, and stores a summary in SQLite off the request path.

## A real session

Six turns of an agent whose system prompt starts with `Current time: …`, through `cachekeeper serve`:

![Terminal: each turn misses at sys[0]:31 on the Current time line; the report attributes 128 of 155 seconds to iso_datetime](docs/images/terminal.svg)

The report ranks causes by wasted time and gives the fix at the source. Fixing the client beats
rewriting its prompts forever.

## Run it

```
pip install cachekeeper
cachekeeper probe --upstream http://localhost:11434 --model qwen3:4b   # the chart above, on your machine
cachekeeper serve --upstream http://localhost:11434                    # point your agent at http://127.0.0.1:4141/v1
cachekeeper report --since 24h
cachekeeper diff <request-id>
```

`serve` proxies `POST /v1/chat/completions`, streaming or not. `GET /metrics` serves Prometheus
counters; `GET /health` shows the detected backend and its version.

## What "hit" means

**A text match is not a token match**: the engine's chat template and tokenizer decide what it
really reuses. The backend's own fields are the truth, and every hit ratio says where it came from:

| Engine | Source | Label |
|---|---|---|
| llama.cpp | `timings.cache_n` and `timings.prompt_n` | measured |
| Ollama native API | `prompt_eval_cached_count` when present, else `prompt_eval_duration` against the slowest rate seen (0.30.x omits the count) | measured / inferred |
| Ollama `/v1`, LM Studio | time to first token on streaming requests; non-streaming shows `n/a` | inferred |

## Guarantees

- Request bytes are forwarded unchanged; this version ships no rewrite rules.
- Upstream errors reach the client with the same status and body; 502 or 504 when the engine is
  unreachable or times out.
- Responses stream chunk by chunk; a client disconnect closes the upstream request and frees the slot.
- A failing storage write never fails a request.
- `Authorization` passes through untouched, headers are never logged, no telemetry, binds
  `127.0.0.1` by default.

## Data and secrets

Prompts contain code and secrets. By default only hashes, offsets, the two differing lines and stats
are stored, in `~/.local/share/cachekeeper/cachekeeper.db`. `--keep-bodies` also stores full request
bodies for `diff`. Rows older than `retention_days` (default 7) are deleted. Treat the file like
the prompts themselves.

## Configuration

Flags beat environment variables, which beat `~/.config/cachekeeper/config.toml`:

```toml
upstream = "http://localhost:8080"
backend = "llamacpp"        # auto | ollama | llamacpp | lmstudio
port = 4141
retention_days = 7
upstream_timeout_seconds = 600
```

Every key also works as an environment variable with the `CACHEKEEPER_` prefix; empty values count
as unset. An invalid configuration fails at startup with every error listed.

## How it is built

![Architecture: core with prompt, classify, stats, report, rules and ports, surrounded by inbound, backend and storage adapters and the container](docs/images/architecture.svg)

The detection logic in `core/` is pure Python with no I/O, so the whole suite runs without a model
in about two seconds. Each engine is one adapter file with recorded responses as test fixtures.
The proxy adds under 5 ms at the 95th percentile for a 30k-token request; a test fails the build
above that.

## Not this

Response caching, KV cache to disk, routing, load balancing, model management, cloud providers.

## Contributing

`make setup`, then `make check` (ruff, mypy strict, tests with a 95% coverage gate on `core/`).
`make perf` runs the overhead budget; `make live` runs against a real Ollama and llama.cpp.
