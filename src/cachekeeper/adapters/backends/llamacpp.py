from typing import Any

from cachekeeper.core.stats import CacheStats, measured_stats

from .taps import FieldTap, openai_probe_body, unavailable


def read_timings(payload: dict[str, Any]) -> CacheStats | None:
    timings = payload.get("timings")
    if not isinstance(timings, dict):
        return None
    reused, processed, prefill_ms = timings.get("cache_n"), timings.get("prompt_n"), timings.get("prompt_ms")
    if not isinstance(reused, int) or not isinstance(processed, int):
        return None
    return measured_stats(reused, processed, prefill_ms if isinstance(prefill_ms, (int, float)) else None)


class LlamaCppBackend:
    name = "llamacpp"
    version_path = "/props"
    probe_path = "/v1/chat/completions"

    def version(self, payload: object) -> str | None:
        build = payload.get("build_info") if isinstance(payload, dict) else None
        return build if isinstance(build, str) else None

    def tap(self, model: str) -> FieldTap:  # noqa: ARG002 - timings are per request, not per model
        return FieldTap(read_timings, unavailable)

    def probe_body(self, model: str, messages: list[dict[str, str]]) -> dict[str, object]:
        return openai_probe_body(model, messages)
