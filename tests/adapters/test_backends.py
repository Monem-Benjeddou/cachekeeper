from pathlib import Path

import pytest

from cachekeeper.adapters.backends.llamacpp import LlamaCppBackend
from cachekeeper.adapters.backends.lmstudio import LmStudioBackend
from cachekeeper.adapters.backends.ollama import OllamaBackend
from cachekeeper.adapters.backends.taps import FieldTap, parse_payloads
from cachekeeper.core.stats import UNAVAILABLE, CacheStats
from tests.conftest import FIXTURES

UPSTREAM = FIXTURES / "upstream"


def run(tap: FieldTap, path: Path, ttft_ms: float | None = None, chars: int = 1000) -> CacheStats:
    data = path.read_bytes()
    for start in range(0, len(data), 7):
        tap.feed(data[start : start + 7])
    return tap.finish(ttft_ms, chars)


@pytest.mark.parametrize("name", ["chat.json", "chat.sse"])
def test_llamacpp_reads_timings_in_both_modes(name: str) -> None:
    stats = run(LlamaCppBackend().tap("m"), UPSTREAM / "llamacpp" / name)
    assert stats.source == "measured"
    assert stats.reused + stats.processed > 0
    assert stats.hit_ratio == stats.reused / (stats.reused + stats.processed)


def test_llamacpp_stream_ratio_from_capture() -> None:
    stats = run(LlamaCppBackend().tap("m"), UPSTREAM / "llamacpp" / "chat.sse")
    assert (stats.reused, stats.processed, stats.prefill_ms) == (271, 1, 23.194)


def test_llamacpp_without_timings_is_unavailable() -> None:
    assert run(LlamaCppBackend().tap("m"), UPSTREAM / "llamacpp" / "error.json") == UNAVAILABLE
    tap = LlamaCppBackend().tap("m")
    tap.feed(b'{"timings": {"cache_n": "x", "prompt_n": 1}}')
    assert tap.finish(None, 10) == UNAVAILABLE


def test_ollama_native_without_cached_count_is_inferred_from_duration() -> None:
    backend = OllamaBackend()
    cold = backend.read_native("m", {"prompt_eval_count": 1538, "prompt_eval_duration": 4_324_445_000})
    warm = backend.read_native("m", {"prompt_eval_count": 1538, "prompt_eval_duration": 30_747_000})
    assert cold is not None
    assert warm is not None
    assert (cold.source, cold.reused) == ("inferred", 0)
    assert warm.source == "inferred"
    assert warm.hit_ratio is not None
    assert warm.hit_ratio > 0.99


def test_ollama_native_uses_cached_count_when_present() -> None:
    stats = run(OllamaBackend().tap("m"), UPSTREAM / "ollama" / "native_cached.json")
    assert (stats.source, stats.reused, stats.processed) == ("measured", 1500, 38)


def test_ollama_native_stream_capture() -> None:
    stats = run(OllamaBackend().tap("m"), UPSTREAM / "ollama" / "native.ndjson")
    assert stats.source == "inferred"
    assert stats.prefill_ms is not None


@pytest.mark.parametrize(
    ("backend", "path"),
    [(OllamaBackend(), "ollama/chat.sse"), (LmStudioBackend(), "lmstudio/chat.sse")],
)
def test_openai_shape_is_ttft_inference(backend: OllamaBackend | LmStudioBackend, path: str) -> None:
    cold = run(backend.tap("m"), UPSTREAM / path, ttft_ms=2000.0)
    warm = run(backend.tap("m"), UPSTREAM / path, ttft_ms=100.0)
    assert (cold.source, cold.reused) == ("inferred", 0)
    assert warm.source == "inferred"
    assert warm.hit_ratio == pytest.approx(0.95)


def test_non_streaming_openai_shape_is_unavailable() -> None:
    assert run(OllamaBackend().tap("m"), UPSTREAM / "ollama" / "chat.json") == UNAVAILABLE
    assert (
        run(LmStudioBackend().tap("m"), UPSTREAM / "ollama" / "chat.sse", ttft_ms=1.0, chars=0) == UNAVAILABLE
    )


def test_malformed_sse_is_unavailable() -> None:
    assert run(OllamaBackend().tap("m"), UPSTREAM / "ollama" / "malformed.sse", ttft_ms=5.0) == UNAVAILABLE


def test_parse_payloads_skips_sse_metadata() -> None:
    body = b': ping\nevent: message\nid: 1\ndata: {"a": 1}\n\ndata: [1]\n\ndata: [DONE]\n\n'
    assert parse_payloads(body) == [{"a": 1}]
    assert parse_payloads(b"[1, 2]") is None


def test_versions_and_probe_bodies() -> None:
    assert OllamaBackend().version({"version": "0.30.10"}) == "0.30.10"
    assert OllamaBackend().version([]) is None
    assert LlamaCppBackend().version({"build_info": "b11669"}) == "b11669"
    assert LlamaCppBackend().version(None) is None
    assert LmStudioBackend().version({"data": []}) is None
    messages = [{"role": "user", "content": "x"}]
    assert OllamaBackend().probe_body("m", messages)["options"] == {"num_predict": 1}
    for backend in (LlamaCppBackend(), LmStudioBackend()):
        assert backend.probe_body("m", messages)["max_tokens"] == 1
