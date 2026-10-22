import pytest

from cachekeeper.core.stats import UNAVAILABLE, CacheStats, infer_stats, measured_stats, wasted_ms


def test_measured_stats_ratio() -> None:
    stats = measured_stats(96, 4, 12.0)
    assert stats.source == "measured"
    assert stats.hit_ratio == pytest.approx(0.96)
    assert measured_stats(0, 0, None).hit_ratio is None
    assert UNAVAILABLE.hit_ratio is None


def test_infer_stats_from_cold_rate() -> None:
    assert infer_stats(1000, 100.0, 1.0) == CacheStats(900, 100, 100.0, "inferred")
    assert infer_stats(1000, 5000.0, 1.0).reused == 0
    assert infer_stats(1000, 100.0, 0.0).processed == 1000


def test_wasted_ms_full_miss_on_a_known_prompt() -> None:
    assert wasted_ms(measured_stats(0, 1000, 2000.0), 1000, 1000) == pytest.approx(2000.0)


def test_wasted_ms_partial() -> None:
    stats = measured_stats(500, 500, 1000.0)
    assert wasted_ms(stats, previous_chars=900, prompt_chars=1000) == pytest.approx(800.0)


def test_wasted_ms_zero_cases() -> None:
    assert wasted_ms(UNAVAILABLE, 100, 100) == 0.0
    assert wasted_ms(measured_stats(100, 0, 5.0), 100, 100) == 0.0
    assert wasted_ms(measured_stats(0, 100, 5.0), 0, 100) == 0.0
    assert wasted_ms(measured_stats(0, 100, None), 100, 100) == 0.0
    assert wasted_ms(measured_stats(0, 100, 5.0), 100, 0) == 0.0
