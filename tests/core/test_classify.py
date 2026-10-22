from pathlib import Path

import pytest

from cachekeeper.core import causes
from cachekeeper.core import classify as c
from cachekeeper.core.prompt import Prompt, PromptHistory, canonical_prompt
from cachekeeper.core.stats import CacheStats, measured_stats
from tests.conftest import PAIRS, load_json


def load_prompt(path: Path) -> Prompt:
    result = canonical_prompt(load_json(path))
    assert result is not None
    return result


@pytest.mark.parametrize("pair", sorted(p.name for p in PAIRS.iterdir()))
def test_diagnose_fixture_pairs(pair: str) -> None:
    expected = load_json(PAIRS / pair / "expected.json")
    history = PromptHistory()
    history.remember("before", load_prompt(PAIRS / pair / "before.json"))
    diagnosis = c.diagnose(load_prompt(PAIRS / pair / "after.json"), history)
    assert diagnosis.cause == expected["cause"]
    assert diagnosis.divergence is not None
    assert diagnosis.divergence.location == expected["location"]
    for key in ("before_line", "after_line"):
        if key in expected:
            assert getattr(diagnosis.divergence, key) == expected[key]


def test_diagnose_new_and_prefix() -> None:
    history = PromptHistory()
    first = load_prompt(PAIRS / "iso_datetime" / "before.json")
    assert c.diagnose(first, history).cause == causes.NEW_PROMPT
    history.remember("1", first)
    assert c.diagnose(first, history).cause == causes.PREFIX_MATCH


def test_classify_falls_back_to_content_changed() -> None:
    before = canonical_prompt({"model": "m", "messages": [{"role": "user", "content": "alpha beta"}]})
    after = canonical_prompt({"model": "m", "messages": [{"role": "user", "content": "alpha gamma"}]})
    assert before is not None
    assert after is not None
    history = PromptHistory()
    history.remember("1", before)
    assert c.diagnose(after, history).cause == causes.CONTENT_CHANGED.name


def test_classify_role_change_without_system_is_content_changed() -> None:
    before = canonical_prompt({"model": "m", "messages": [{"role": "system", "content": "s"}]})
    after = canonical_prompt({"model": "m", "messages": [{"role": "user", "content": "s"}]})
    assert before is not None
    assert after is not None
    history = PromptHistory()
    history.remember("1", before)
    assert c.diagnose(after, history).cause == causes.CONTENT_CHANGED.name


def test_classify_none_divergence_is_prefix_match() -> None:
    prompt = load_prompt(PAIRS / "uuid" / "before.json")
    assert c.classify(None, prompt, prompt) == causes.PREFIX_MATCH


@pytest.mark.parametrize(
    ("cause", "stats", "expected"),
    [
        (causes.PREFIX_MATCH, measured_stats(0, 500, 10.0), causes.REPROCESSED.name),
        (causes.PREFIX_MATCH, measured_stats(400, 100, 10.0), causes.PREFIX_MATCH),
        (causes.PREFIX_MATCH, CacheStats(0, 500, 10.0, "inferred"), causes.PREFIX_MATCH),
        (causes.ISO_DATETIME.name, measured_stats(0, 500, 10.0), causes.ISO_DATETIME.name),
    ],
)
def test_confirm(cause: str, stats: CacheStats, expected: str) -> None:
    assert c.confirm(cause, stats) == expected
