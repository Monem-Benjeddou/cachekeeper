from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .causes import FIXES, NEW_PROMPT, NOT_MISSES, PREFIX_MATCH
from .prompt import common_prefix_length
from .stats import UNAVAILABLE, CacheStats

TOP_CAUSES = 3
_EXCERPT_CHARS = 60
_EXCERPT_LEAD = 40


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    ts: float
    model: str
    backend: str
    cause: str
    stats: CacheStats = UNAVAILABLE
    client_hint: str = ""
    prompt_chars: int = 0
    prompt_hash: str = ""
    system_hash: str = ""
    match_id: str | None = None
    divergence_offset: int | None = None
    location: str = ""
    before_line: str = ""
    after_line: str = ""
    ttft_ms: float | None = None
    total_ms: float | None = None
    wasted_ms: float = 0.0
    rules: tuple[str, ...] = ()


@dataclass(frozen=True)
class CauseSummary:
    cause: str
    count: int
    wasted_ms: float
    example: RequestRecord
    fix: str


@dataclass(frozen=True)
class Traffic:
    requests: int
    with_stats: int
    hit_rate: float | None


@dataclass(frozen=True)
class Report:
    requests: int
    main: Traffic
    side: Traffic
    prefill_ms: float
    wasted_ms: float
    top_causes: tuple[CauseSummary, ...]


def status_line(record: RequestRecord) -> str:
    """One line per request: hit ratio, where the prompt diverged, prefill time, rules applied."""
    ratio = record.stats.hit_ratio
    hit = "hit n/a" if ratio is None else f"hit {ratio:.0%}"
    if record.stats.source == "inferred":
        hit += " (inferred)"
    prefill = record.stats.prefill_ms
    return " | ".join(
        (
            f"{record.request_id} {hit}",
            _divergence_text(record),
            "prefill n/a" if prefill is None else f"prefill {prefill / 1000:.1f}s",
            f"rule: {', '.join(record.rules) or 'none'}",
        )
    )


def split_main(records: Sequence[RequestRecord]) -> tuple[list[RequestRecord], list[RequestRecord]]:
    """Split into main turns and side requests.

    A model's main conversation is its most frequent system prompt; requests with any other system
    prompt (summaries, commit messages, titles) are side requests.
    """
    main_system: dict[str, str] = {}
    for (model, system_hash), _ in Counter((r.model, r.system_hash) for r in records).most_common():
        main_system.setdefault(model, system_hash)
    main = [r for r in records if main_system[r.model] == r.system_hash]
    return main, [r for r in records if main_system[r.model] != r.system_hash]


def _traffic(records: Sequence[RequestRecord]) -> Traffic:
    measured = [r.stats for r in records if r.stats.hit_ratio is not None]
    total = sum(s.reused + s.processed for s in measured)
    hit_rate = sum(s.reused for s in measured) / total if total else None
    return Traffic(len(records), len(measured), hit_rate)


def build_report(records: Sequence[RequestRecord]) -> Report:
    """Aggregate records into main and side traffic and the top causes ranked by estimated waste."""
    main, side = split_main(records)
    misses = [r for r in records if r.cause not in NOT_MISSES]
    counts = Counter(r.cause for r in misses)
    wasted: defaultdict[str, float] = defaultdict(float)
    for record in misses:
        wasted[record.cause] += record.wasted_ms
    ranked = sorted(counts, key=lambda cause: (-wasted[cause], -counts[cause], cause))[:TOP_CAUSES]
    return Report(
        requests=len(records),
        main=_traffic(main),
        side=_traffic(side),
        prefill_ms=sum(r.stats.prefill_ms or 0.0 for r in records),
        wasted_ms=sum(r.wasted_ms for r in records),
        top_causes=tuple(
            CauseSummary(
                cause,
                counts[cause],
                wasted[cause],
                next(r for r in misses if r.cause == cause),
                FIXES.get(cause, ""),
            )
            for cause in ranked
        ),
    )


def render_report(report: Report) -> str:
    """Plain-text report; offsets are in characters of the canonical prompt."""
    lines = [
        f"requests        {report.requests}",
        f"main turns      {_traffic_text(report.main)}",
        f"side requests   {_traffic_text(report.side)}",
        f"prefill         {report.prefill_ms / 1000:.1f} s",
        f"wasted (est.)   {report.wasted_ms / 1000:.1f} s",
        "",
        "top causes" if report.top_causes else "no cache misses with a known cause",
    ]
    for rank, summary in enumerate(report.top_causes, 1):
        example = summary.example
        lines += [
            f"{rank}. {summary.cause}: {summary.count} requests, {summary.wasted_ms / 1000:.1f} s wasted",
            f"   e.g. {example.request_id} at {example.location or 'n/a'} (chars)",
            f"   - {_excerpt(example.before_line, example.after_line)}",
            f"   + {_excerpt(example.after_line, example.before_line)}",
        ]
        if summary.fix:
            lines.append(f"   fix: {summary.fix}")
    return "\n".join(lines)


def _traffic_text(traffic: Traffic) -> str:
    hit = "n/a" if traffic.hit_rate is None else f"{traffic.hit_rate:.0%}"
    return f"{traffic.requests}, hit rate {hit} ({traffic.with_stats} with backend stats)"


def _divergence_text(record: RequestRecord) -> str:
    if record.location:
        excerpt = _excerpt(record.after_line, record.before_line)
        return f'miss at {record.location} "{excerpt}" ({record.cause})'
    return {PREFIX_MATCH: "prefix match", NEW_PROMPT: "new prompt"}.get(record.cause, record.cause)


def _excerpt(line: str, other: str) -> str:
    if len(line) <= _EXCERPT_CHARS:
        return line.strip()
    start = max(0, min(common_prefix_length(line, other) - _EXCERPT_LEAD, len(line) - _EXCERPT_CHARS))
    text = line[start : start + _EXCERPT_CHARS]
    return ("…" if start else "") + text + ("…" if start + _EXCERPT_CHARS < len(line) else "")
