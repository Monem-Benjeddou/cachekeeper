from dataclasses import dataclass
from typing import Literal

StatsSource = Literal["measured", "inferred", "unavailable"]


@dataclass(frozen=True)
class CacheStats:
    reused: int = 0
    processed: int = 0
    prefill_ms: float | None = None
    source: StatsSource = "unavailable"

    @property
    def hit_ratio(self) -> float | None:
        total = self.reused + self.processed
        return None if self.source == "unavailable" or total == 0 else self.reused / total


UNAVAILABLE = CacheStats()


def measured_stats(reused: int, processed: int, prefill_ms: float | None) -> CacheStats:
    """Stats read from backend fields that count reused and processed tokens directly."""
    return CacheStats(reused, processed, prefill_ms, "measured")


def infer_stats(units: int, prefill_ms: float, cold_ms_per_unit: float) -> CacheStats:
    """Estimate reuse from prefill time, assuming each uncached unit costs `cold_ms_per_unit`.

    A unit is whatever `units` counts: tokens for Ollama native, characters for TTFT inference.
    """
    processed = units if cold_ms_per_unit <= 0 else min(units, round(prefill_ms / cold_ms_per_unit))
    return CacheStats(units - processed, processed, prefill_ms, "inferred")


def wasted_ms(stats: CacheStats, previous_chars: int, prompt_chars: int) -> float:
    """Estimate prefill time spent re-processing text the previous prompt already had.

    With a stable prompt only the part beyond the previous prompt would need processing; any
    uncached share above that, as reported by the backend, counts as wasted.
    """
    ratio = stats.hit_ratio
    if ratio is None or stats.prefill_ms is None or prompt_chars == 0 or ratio >= 1:
        return 0.0
    ideal_uncached = max(0, prompt_chars - previous_chars) / prompt_chars
    actual_uncached = 1 - ratio
    return stats.prefill_ms * max(0.0, actual_uncached - ideal_uncached) / actual_uncached
