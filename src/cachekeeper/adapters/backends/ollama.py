from typing import Any

from cachekeeper.core.stats import CacheStats, measured_stats

from .taps import ColdRates, FieldTap, ttft_inference

NS_PER_MS = 1_000_000


class OllamaBackend:
    name = "ollama"
    version_path = "/api/version"
    probe_path = "/api/chat"

    def __init__(self) -> None:
        self._token_rates = ColdRates()
        self._char_rates = ColdRates()

    def version(self, payload: object) -> str | None:
        version = payload.get("version") if isinstance(payload, dict) else None
        return version if isinstance(version, str) else None

    def tap(self, model: str) -> FieldTap:
        return FieldTap(
            lambda payload: self.read_native(model, payload), ttft_inference(self._char_rates, model)
        )

    def read_native(self, model: str, payload: dict[str, Any]) -> CacheStats | None:
        count, duration_ns = payload.get("prompt_eval_count"), payload.get("prompt_eval_duration")
        if not isinstance(count, int) or not isinstance(duration_ns, int):
            return None
        prefill_ms = duration_ns / NS_PER_MS
        cached = payload.get("prompt_eval_cached_count")
        if isinstance(cached, int):
            return measured_stats(cached, count - cached, prefill_ms)
        return self._token_rates.infer(model, count, prefill_ms)

    def probe_body(self, model: str, messages: list[dict[str, str]]) -> dict[str, object]:
        return {"model": model, "messages": messages, "stream": True, "options": {"num_predict": 1}}
