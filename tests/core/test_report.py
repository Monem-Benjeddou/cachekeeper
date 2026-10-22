from cachekeeper.core import causes
from cachekeeper.core.causes import NEW_PROMPT, PREFIX_MATCH, UNPARSED
from cachekeeper.core.report import (
    RequestRecord,
    Traffic,
    build_report,
    render_report,
    split_main,
    status_line,
)
from cachekeeper.core.stats import CacheStats, measured_stats

ISO_DATETIME, TOOL_ORDER = causes.ISO_DATETIME.name, causes.TOOL_ORDER.name


def record(cause: str, **fields: object) -> RequestRecord:
    return RequestRecord(request_id="01ID", ts=0.0, model="m", backend="llamacpp", cause=cause, **fields)  # type: ignore[arg-type]


def test_status_line_miss() -> None:
    line = status_line(
        record(
            ISO_DATETIME,
            stats=measured_stats(48, 1152, 24700.0),
            location="sys[0]:312",
            after_line="Current time: 2026-10-21T14:03",
        )
    )
    assert line == (
        '01ID hit 4% | miss at sys[0]:312 "Current time: 2026-10-21T14:03" (iso_datetime)'
        " | prefill 24.7s | rule: none"
    )


def test_status_line_unavailable_and_inferred() -> None:
    assert status_line(record(UNPARSED)) == "01ID hit n/a | unparsed | prefill n/a | rule: none"
    inferred = record(PREFIX_MATCH, stats=CacheStats(9, 1, 300.0, "inferred"))
    assert status_line(inferred) == "01ID hit 90% (inferred) | prefix match | prefill 0.3s | rule: none"
    assert "new prompt" in status_line(record(NEW_PROMPT))


def test_status_line_excerpt_is_centred_on_the_divergence() -> None:
    before, after = "a" * 100 + "OLD" + "b" * 100, "a" * 100 + "NEW" + "b" * 100
    line = status_line(record(ISO_DATETIME, location="sys[0]:100", before_line=before, after_line=after))
    assert '"…' + "a" * 40 + "NEW" + "b" * 17 + '…"' in line
    tail_record = record(
        ISO_DATETIME, location="sys[0]:1", after_line="x" * 100 + "END", before_line="x" * 102
    )
    tail = status_line(tail_record)
    assert '"…' + "x" * 57 + 'END"' in tail
    head = status_line(record(ISO_DATETIME, location="sys[0]:1", after_line="y" * 100, before_line="z"))
    assert '"' + "y" * 60 + '…"' in head


def test_build_report_ranks_causes_by_wasted_time() -> None:
    records = [
        record(NEW_PROMPT, stats=measured_stats(0, 100, 1000.0)),
        record(TOOL_ORDER, stats=measured_stats(0, 100, 1000.0), wasted_ms=900.0),
        record(TOOL_ORDER, stats=measured_stats(0, 100, 1000.0), wasted_ms=900.0),
        record(ISO_DATETIME, stats=measured_stats(0, 100, 3000.0), wasted_ms=3000.0),
        record(PREFIX_MATCH, stats=measured_stats(100, 0, 10.0)),
        record(UNPARSED),
    ]
    report = build_report(records)
    assert report.requests == 6
    assert report.main == Traffic(6, 5, 100 / 500)
    assert report.side == Traffic(0, 0, None)
    assert report.prefill_ms == 6010.0
    assert report.wasted_ms == 4800.0
    assert [(s.cause, s.count) for s in report.top_causes] == [(ISO_DATETIME, 1), (TOOL_ORDER, 2)]
    assert "deterministic order" in report.top_causes[1].fix


def test_build_report_empty() -> None:
    report = build_report([])
    assert report.main.hit_rate is None
    assert report.top_causes == ()


def test_render_report_without_causes() -> None:
    assert "no cache misses with a known cause" in render_report(build_report([]))
    assert "main turns      0, hit rate n/a (0 with backend stats)" in render_report(build_report([]))


def test_render_report_skips_missing_fix() -> None:
    text = render_report(build_report([record("custom", stats=measured_stats(0, 1, 1.0))]))
    assert "1. custom: 1 requests" in text
    assert "fix:" not in text


def test_split_main_uses_each_models_most_frequent_system_prompt() -> None:
    main_turns = [
        record(PREFIX_MATCH, system_hash="agent", stats=measured_stats(0, 100, 1.0)) for _ in range(3)
    ]
    side = record(PREFIX_MATCH, system_hash="summarizer", stats=measured_stats(90, 10, 1.0))
    other_model = RequestRecord("02", 0.0, "other", "llamacpp", PREFIX_MATCH, system_hash="summarizer")
    main, rest = split_main([*main_turns, side, other_model])
    assert main == [*main_turns, other_model]
    assert rest == [side]
    report = build_report([*main_turns, side])
    assert (report.main.hit_rate, report.side.hit_rate) == (0.0, 0.9)
    assert "side requests   1, hit rate 90% (1 with backend stats)" in render_report(report)
