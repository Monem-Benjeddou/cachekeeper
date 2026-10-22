import argparse
import asyncio
import difflib
import json
import logging
import re
import sys
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone

import httpx
from pydantic import ValidationError

from cachekeeper import __version__
from cachekeeper.config import Settings
from cachekeeper.core.errors import UnsupportedBackendError, UpstreamError
from cachekeeper.core.ports import Backend, Clock, StatsTap
from cachekeeper.core.report import build_report, render_report
from cachekeeper.core.stats import CacheStats

LOG = logging.getLogger("cachekeeper")
DURATION = re.compile(r"(\d+)([smhd])")
UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
PROBE_LINES = 244
PROBE_WORDS = (
    "parser",
    "router",
    "cache",
    "storage",
    "session",
    "config",
    "client",
    "report",
    "worker",
    "index",
)
CONFIG_ERROR = 2


def duration_seconds(text: str) -> float:
    match = DURATION.fullmatch(text)
    if match is None:
        raise argparse.ArgumentTypeError(f"expected a duration like 30m, 24h or 7d, got {text!r}")
    return int(match.group(1)) * UNIT_SECONDS[match.group(2)]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="cachekeeper", description="Why your local prompt cache keeps missing."
    )
    root.add_argument("--version", action="version", version=f"cachekeeper {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the shadow proxy")
    serve.add_argument("--upstream")
    serve.add_argument("--backend", choices=["auto", "ollama", "llamacpp", "lmstudio"])
    serve.add_argument("--rule", dest="rules", action="append")
    serve.add_argument("--keep-bodies", action="store_true", default=None)
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    report = commands.add_parser("report", help="summarize stored requests")
    report.add_argument("--since", type=duration_seconds, default=duration_seconds("24h"))
    report.add_argument("--model")
    diff = commands.add_parser("diff", help="show where one request diverged from its match")
    diff.add_argument("request_id")
    probe = commands.add_parser("probe", help="measure cold, warm and timestamp prefill on this machine")
    probe.add_argument("--upstream")
    probe.add_argument("--backend", choices=["auto", "ollama", "llamacpp", "lmstudio"])
    probe.add_argument("--model", required=True)
    return root


def load_settings(args: argparse.Namespace) -> Settings | None:
    keys = ("upstream", "backend", "rules", "keep_bodies", "host", "port")
    overrides = {key: getattr(args, key) for key in keys if getattr(args, key, None) is not None}
    try:
        return Settings(**overrides)
    except ValidationError as error:
        for detail in error.errors():
            location = ".".join(str(part) for part in detail["loc"])
            sys.stderr.write(f"config error: {location}: {detail['msg']}\n")
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings = load_settings(args)
    if settings is None:
        return CONFIG_ERROR
    commands = {"serve": serve, "report": report, "diff": diff, "probe": probe}
    return commands[args.command](settings, args)


def serve(settings: Settings, _args: argparse.Namespace) -> int:
    import uvicorn  # noqa: PLC0415 - only serve needs the server

    from cachekeeper import container  # noqa: PLC0415
    from cachekeeper.adapters.inbound.http import create_app  # noqa: PLC0415
    from cachekeeper.adapters.storage.sqlite import SqliteStorage  # noqa: PLC0415

    container.configure_logging()
    try:
        backend, version = asyncio.run(_detect(settings))
    except UnsupportedBackendError as error:
        LOG.error("startup_failed", extra={"error": str(error)})  # noqa: TRY400 - the message is the whole story
        return 1
    clock = container.SystemClock()
    services = container.Services(
        settings=settings,
        backend=backend,
        backend_version=version,
        storage=SqliteStorage(settings.db_path, clock, settings.retention_days),
        clock=clock,
        client=container.upstream_client(settings),
        new_id=lambda: container.ulid(clock),
        rules=container.enabled_rules(settings.rules),
    )
    LOG.info(
        "startup",
        extra={"version": __version__, "backend": backend.name, "backend_version": version or "unknown",
               "listen": f"http://{settings.host}:{settings.port}", "upstream": settings.upstream},
    )  # fmt: skip
    uvicorn.run(
        create_app(services), host=settings.host, port=settings.port, access_log=False, log_config=None
    )
    return 0


async def _detect(settings: Settings) -> tuple[Backend, str | None]:
    from cachekeeper.container import detect_backend, upstream_client  # noqa: PLC0415

    async with upstream_client(settings) as client:
        return await detect_backend(client, settings.backend)


def report(settings: Settings, args: argparse.Namespace) -> int:
    from cachekeeper.adapters.storage.sqlite import SqliteStorage  # noqa: PLC0415
    from cachekeeper.container import SystemClock  # noqa: PLC0415

    clock = SystemClock()
    storage = SqliteStorage(settings.db_path, clock, settings.retention_days)
    records = storage.records(clock.now() - args.since, args.model)
    sys.stdout.write(render_report(build_report(records)) + "\n")
    return 0


def diff(settings: Settings, args: argparse.Namespace) -> int:
    from cachekeeper.adapters.storage.sqlite import SqliteStorage  # noqa: PLC0415
    from cachekeeper.container import SystemClock  # noqa: PLC0415

    storage = SqliteStorage(settings.db_path, SystemClock(), settings.retention_days)
    record = storage.record(args.request_id)
    if record is None:
        sys.stderr.write(f"unknown request id {args.request_id}\n")
        return 1
    lines = [
        f"request   {record.request_id}  model {record.model}  backend {record.backend}",
        f"matched   {record.match_id or 'none'}",
        f"cause     {record.cause}" + (f" at {record.location} (chars)" if record.location else ""),
    ]
    if record.location:
        lines += [f"- {record.before_line}", f"+ {record.after_line}"]
    before, after = storage.body(record.match_id or ""), storage.body(record.request_id)
    if before is not None and after is not None:
        lines += difflib.unified_diff(
            _pretty(before), _pretty(after), record.match_id or "", record.request_id
        )
    sys.stdout.write("\n".join(line.rstrip("\n") for line in lines) + "\n")
    return 0


def _pretty(body: bytes) -> list[str]:
    try:
        return json.dumps(json.loads(body), indent=2, ensure_ascii=False).splitlines()
    except ValueError:
        return body.decode(errors="replace").splitlines()


def probe(settings: Settings, args: argparse.Namespace) -> int:
    from cachekeeper.container import SystemClock  # noqa: PLC0415

    try:
        header, rows = asyncio.run(run_probe(settings, args.model, SystemClock()))
    except (UpstreamError, UnsupportedBackendError) as error:
        sys.stderr.write(f"probe failed: {error}\n")
        return 1
    sys.stdout.write(header + "\n\n" + render_probe(rows) + "\n")
    return 0


def probe_system_prompt(run_id: str) -> str:
    words = PROBE_WORDS
    lines = [f"Probe run {run_id}. You are a coding agent. Follow every rule below."]
    lines += [
        f"{i:04d}. Keep the {words[i % 10]} module small, test the {words[i * 7 % 10]} path before merging, "
        f"and document how the {words[i * 3 % 10]} handles errors."
        for i in range(PROBE_LINES)
    ]
    return "\n".join(lines)


async def run_probe(
    settings: Settings, model: str, clock: Clock
) -> tuple[str, list[tuple[str, float, CacheStats]]]:
    from cachekeeper.container import detect_backend, upstream_client  # noqa: PLC0415

    stable = probe_system_prompt(uuid.uuid4().hex[:8])
    async with upstream_client(settings) as client:
        backend, version = await detect_backend(client, settings.backend)
        rows = []
        for label, timestamped in (
            ("Stable prompt, cold", False),
            ("Stable prompt, warm (same prefix)", False),
            ("`Current time: …` in system prompt, turn 1", True),
            ("Same, turn 2 (timestamp changed)", True),
        ):
            now = datetime.fromtimestamp(clock.now(), tz=timezone.utc).isoformat(timespec="milliseconds")
            system = f"Current time: {now}\n{stable}" if timestamped else stable
            messages = [{"role": "system", "content": system}, {"role": "user", "content": "hi"}]
            request = client.build_request(
                "POST", backend.probe_path, json=backend.probe_body(model, messages)
            )
            tap = backend.tap(model)
            wall_ms, ttft_ms = await _probe_once(client, request, tap, clock)
            rows.append((label, wall_ms, tap.finish(ttft_ms, len(system))))
    header = f"{backend.name} {version or 'unknown'} | {model} | system prompt {len(stable)} chars"
    return header, rows


async def _probe_once(
    client: httpx.AsyncClient, request: httpx.Request, tap: StatsTap, clock: Clock
) -> tuple[float, float | None]:
    started, ttft_ms = clock.monotonic(), None
    try:
        response = await client.send(request, stream=True)
        try:
            if response.status_code >= httpx.codes.BAD_REQUEST:
                raise UpstreamError(response.status_code, (await response.aread()).decode(errors="replace"))
            async for chunk in response.aiter_raw():
                ttft_ms = ttft_ms or (clock.monotonic() - started) * 1000
                tap.feed(chunk)
        finally:
            await response.aclose()
    except httpx.HTTPError as error:
        raise UpstreamError(httpx.codes.BAD_GATEWAY, f"upstream failed: {error!r}") from error
    return (clock.monotonic() - started) * 1000, ttft_ms


def render_probe(rows: list[tuple[str, float, CacheStats]]) -> str:
    width = max(len(label) for label, _, _ in rows)
    lines = [
        f"| {'Request':<{width}} | Prefill  | Wall     | Cache hit          |",
        f"|{'-' * (width + 2)}|----------|----------|--------------------|",
    ]
    for label, wall_ms, stats in rows:
        prefill = "n/a" if stats.prefill_ms is None else f"{stats.prefill_ms / 1000:.2f} s"
        ratio = stats.hit_ratio
        hit = "n/a" if ratio is None else f"{ratio:.0%} {stats.source}"
        lines.append(f"| {label:<{width}} | {prefill:<8} | {wall_ms / 1000:>6.2f} s | {hit:<18} |")
    return "\n".join(lines)
