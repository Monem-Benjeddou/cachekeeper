import json
from collections.abc import Callable
from typing import Any

from cachekeeper.core.stats import UNAVAILABLE, CacheStats, infer_stats

Reader = Callable[[dict[str, Any]], CacheStats | None]
Fallback = Callable[[float | None, int], CacheStats]


# The slowest prefill rate seen per model is taken as the cost of an uncached unit.
class ColdRates:
    def __init__(self) -> None:
        self._ms_per_unit: dict[str, float] = {}

    def infer(self, model: str, units: int, prefill_ms: float) -> CacheStats:
        if units <= 0:
            return UNAVAILABLE
        rate = max(self._ms_per_unit.get(model, 0.0), prefill_ms / units)
        self._ms_per_unit[model] = rate
        return infer_stats(units, prefill_ms, rate)


class FieldTap:
    def __init__(self, read: Reader, fallback: Fallback) -> None:
        self._read = read
        self._fallback = fallback
        self._chunks: list[bytes] = []

    def feed(self, chunk: bytes) -> None:
        self._chunks.append(chunk)

    def finish(self, ttft_ms: float | None, prompt_chars: int) -> CacheStats:
        payloads = parse_payloads(b"".join(self._chunks))
        if payloads is None:
            return UNAVAILABLE
        for payload in reversed(payloads):
            stats = self._read(payload)
            if stats is not None:
                return stats
        return self._fallback(ttft_ms, prompt_chars)


def parse_payloads(body: bytes) -> list[dict[str, Any]] | None:
    try:
        whole = json.loads(body)
    except ValueError:
        pass
    else:
        return [whole] if isinstance(whole, dict) else None
    payloads = []
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith(b"data:"):
            line = line[5:].strip()
        elif line.startswith((b"event:", b"id:", b"retry:", b":")):
            continue
        if not line or line == b"[DONE]":
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            return None
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def unavailable(_ttft_ms: float | None, _prompt_chars: int) -> CacheStats:
    return UNAVAILABLE


def ttft_inference(rates: ColdRates, model: str) -> Fallback:
    def infer(ttft_ms: float | None, prompt_chars: int) -> CacheStats:
        return UNAVAILABLE if ttft_ms is None else rates.infer(model, prompt_chars, ttft_ms)

    return infer


def never(_payload: dict[str, Any]) -> CacheStats | None:
    return None


def openai_probe_body(model: str, messages: list[dict[str, str]]) -> dict[str, object]:
    return {"model": model, "messages": messages, "stream": True, "max_tokens": 1}
