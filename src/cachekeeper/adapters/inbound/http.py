import contextlib
import hashlib
import json
import logging
import sys
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from cachekeeper import __version__
from cachekeeper.container import Services
from cachekeeper.core.causes import SKIPPED_LARGE, SYSTEM_REPLACED, UNPARSED
from cachekeeper.core.classify import Diagnosis, confirm, diagnose
from cachekeeper.core.errors import UpstreamError
from cachekeeper.core.prompt import PromptHistory, canonical_prompt
from cachekeeper.core.report import RequestRecord, status_line
from cachekeeper.core.rules import apply_rules
from cachekeeper.core.stats import UNAVAILABLE, CacheStats, wasted_ms

LOG = logging.getLogger("cachekeeper")
MAX_INSPECT_BYTES = 2 * 1024 * 1024
HIT_RATIO = 0.5
BAD_GATEWAY = 502
GATEWAY_TIMEOUT = 504
CLIENT_ERROR = 400
HOP_BY_HOP = frozenset(
    {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade"}
)
COUNTERS = (
    "requests",
    "cache_hits",
    "cache_misses",
    "prefill_seconds",
    "wasted_seconds",
    SKIPPED_LARGE,
    UNPARSED,
)


@dataclass(frozen=True)
class Exchange:
    request_id: str
    ts: float
    started: float
    body: bytes
    parsed: Mapping[str, Any] | None
    diagnosis: Diagnosis | None
    cause: str
    client_hint: str
    rules: tuple[str, ...] = ()


class Metrics:
    def __init__(self) -> None:
        self.counts: defaultdict[str, float] = defaultdict(float)
        self.rules: defaultdict[str, int] = defaultdict(int)

    def observe(self, record: RequestRecord) -> None:
        self.counts["requests"] += 1
        ratio = record.stats.hit_ratio
        if ratio is not None:
            self.counts["cache_hits" if ratio >= HIT_RATIO else "cache_misses"] += 1
        self.counts["prefill_seconds"] += (record.stats.prefill_ms or 0.0) / 1000
        self.counts["wasted_seconds"] += record.wasted_ms / 1000
        if record.cause in (SKIPPED_LARGE, UNPARSED):
            self.counts[record.cause] += 1
        for rule in record.rules:
            self.rules[rule] += 1

    def render(self) -> str:
        lines = []
        for name in COUNTERS:
            lines += [
                f"# TYPE cachekeeper_{name}_total counter",
                f"cachekeeper_{name}_total {self.counts[name]}",
            ]
        lines.append("# TYPE cachekeeper_rule_applied_total counter")
        lines += [f'cachekeeper_rule_applied_total{{rule="{r}"}} {n}' for r, n in sorted(self.rules.items())]
        return "\n".join(lines) + "\n"


class RelayResponse(StreamingResponse):
    def __init__(self, relay: AsyncGenerator[bytes, None], status: int, headers: Mapping[str, str]) -> None:
        super().__init__(relay, status, headers)
        self._relay = relay

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Starlette never closes the iterator on disconnect; closing it here frees the backend slot.
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._relay.aclose()


class Proxy:
    def __init__(self, services: Services) -> None:
        self._s = services
        self._history = PromptHistory()
        self.metrics = Metrics()

    async def chat(self, request: Request) -> Response:
        exchange = self._inspect(await request.body(), request.headers.get("user-agent", ""))
        rewritten, exchange = self._rewrite(exchange)
        url = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        headers = {k: v for k, v in request.headers.items() if k not in HOP_BY_HOP}
        try:
            upstream = await self._send(url, rewritten or exchange.body, headers)
            if rewritten is not None and upstream.status_code >= CLIENT_ERROR:
                await upstream.aclose()
                LOG.warning("rewrite_reverted", extra={"request_id": exchange.request_id})
                exchange = replace(exchange, rules=())
                upstream = await self._send(url, exchange.body, headers)
        except UpstreamError as error:
            self._finish(exchange, UNAVAILABLE, None)
            return JSONResponse({"error": {"message": error.message, "type": "upstream_error"}}, error.status)
        response_headers = {
            k: v for k, v in upstream.headers.items() if k not in HOP_BY_HOP - {"content-length"}
        }
        return RelayResponse(self._relay(exchange, upstream), upstream.status_code, response_headers)

    def _inspect(self, body: bytes, user_agent: str) -> Exchange:
        exchange = Exchange(
            request_id=self._s.new_id(), ts=self._s.clock.now(), started=self._s.clock.monotonic(),
            body=body, parsed=None, diagnosis=None, cause=UNPARSED, client_hint=user_agent.split("/", 1)[0],
        )  # fmt: skip
        if len(body) > MAX_INSPECT_BYTES:
            return replace(exchange, cause=SKIPPED_LARGE)
        try:
            parsed = json.loads(body)
        except (ValueError, RecursionError):
            return exchange
        prompt = canonical_prompt(parsed) if isinstance(parsed, dict) else None
        if prompt is None:
            return exchange
        diagnosis = diagnose(prompt, self._history)
        self._history.remember(exchange.request_id, prompt)
        return replace(exchange, parsed=parsed, diagnosis=diagnosis, cause=diagnosis.cause)

    def _rewrite(self, exchange: Exchange) -> tuple[bytes | None, Exchange]:
        if not self._s.rules or exchange.parsed is None:
            return None, exchange
        outcome = apply_rules(exchange.parsed, self._history, self._s.rules)
        if outcome.failed:
            LOG.error("rule_failed", extra={"request_id": exchange.request_id, "rule": outcome.failed})
        if not outcome.changes:
            return None, exchange
        applied = tuple(change.rule for change in outcome.changes)
        return json.dumps(outcome.body).encode(), replace(exchange, rules=applied)

    async def _send(self, url: str, body: bytes, headers: dict[str, str]) -> httpx.Response:
        request = self._s.client.build_request("POST", url, content=body, headers=headers)
        # httpx adds its own defaults (gzip, user-agent); the upstream must see only the client's headers.
        for name in set(self._s.client.headers) - set(headers):
            del request.headers[name]
        try:
            return await self._s.client.send(request, stream=True)
        except httpx.TimeoutException as error:
            raise UpstreamError(GATEWAY_TIMEOUT, f"upstream timed out: {error!r}") from error
        except httpx.HTTPError as error:
            raise UpstreamError(BAD_GATEWAY, f"upstream unreachable: {error!r}") from error

    async def _relay(self, exchange: Exchange, upstream: httpx.Response) -> AsyncGenerator[bytes, None]:
        model = exchange.diagnosis.prompt.model if exchange.diagnosis else ""
        tap = self._s.backend.tap(model)
        ttft_ms: float | None = None
        complete = False
        try:
            async for chunk in upstream.aiter_raw():
                if ttft_ms is None:
                    ttft_ms = (self._s.clock.monotonic() - exchange.started) * 1000
                tap.feed(chunk)
                yield chunk
            complete = True
        except httpx.HTTPError as error:
            LOG.warning(
                "upstream_stream_failed", extra={"request_id": exchange.request_id, "error": repr(error)}
            )
        finally:
            streaming = bool(exchange.parsed and exchange.parsed.get("stream") is True)
            usable = complete and upstream.status_code < CLIENT_ERROR and exchange.diagnosis is not None
            chars = len(exchange.diagnosis.prompt.text) if exchange.diagnosis else 0
            stats = tap.finish(ttft_ms if streaming else None, chars) if usable else UNAVAILABLE
            self._finish(exchange, stats, ttft_ms)
            await upstream.aclose()

    def _finish(self, exchange: Exchange, stats: CacheStats, ttft_ms: float | None) -> None:
        record = build_record(exchange, stats, ttft_ms, self._s)
        self._s.storage.save(record, exchange.body if self._s.settings.keep_bodies else None)
        self.metrics.observe(record)
        sys.stdout.write(status_line(record) + "\n")
        sys.stdout.flush()


def build_record(
    exchange: Exchange, stats: CacheStats, ttft_ms: float | None, services: Services
) -> RequestRecord:
    diagnosis = exchange.diagnosis
    cause = confirm(exchange.cause, stats)
    match = diagnosis.match if diagnosis else None
    divergence = diagnosis.divergence if diagnosis else None
    text = diagnosis.prompt.text if diagnosis else ""
    segments = diagnosis.prompt.segments if diagnosis else ()
    system = next((s.text for s in segments if s.label.startswith("sys")), "")
    previous = len(match.prompt.text) if match and cause != SYSTEM_REPLACED.name else 0
    return RequestRecord(
        request_id=exchange.request_id, ts=exchange.ts, model=diagnosis.prompt.model if diagnosis else "",
        backend=services.backend.name, cause=cause, stats=stats, client_hint=exchange.client_hint,
        prompt_chars=len(text), prompt_hash=_digest(text), system_hash=_digest(system),
        match_id=match.request_id if match else None,
        divergence_offset=divergence.offset if divergence else None,
        location=divergence.location if divergence else "",
        before_line=divergence.before_line if divergence else "",
        after_line=divergence.after_line if divergence else "",
        ttft_ms=ttft_ms, total_ms=(services.clock.monotonic() - exchange.started) * 1000,
        wasted_ms=wasted_ms(stats, previous, len(text)), rules=exchange.rules,
    )  # fmt: skip


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16] if text else ""


def create_app(services: Services) -> Starlette:
    proxy = Proxy(services)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        services.storage.start()
        yield
        await services.storage.stop()
        await services.client.aclose()

    async def metrics(_request: Request) -> Response:
        return PlainTextResponse(proxy.metrics.render(), media_type="text/plain; version=0.0.4")

    async def health(_request: Request) -> Response:
        backend = {"backend": services.backend.name, "backend_version": services.backend_version}
        return JSONResponse({"status": "ok", "version": __version__, **backend})

    routes = [
        Route("/v1/chat/completions", proxy.chat, methods=["POST"]),
        Route("/metrics", metrics),
        Route("/health", health),
    ]
    return Starlette(routes=routes, lifespan=lifespan)
