from cachekeeper.core.causes import CAUSES, CONTENT_CHANGED, FIXES, REPROCESSED
from tests.conftest import PAIRS, load_json


def test_every_fixture_cause_is_in_the_catalogue_with_a_fix() -> None:
    names = {load_json(p / "expected.json")["cause"] for p in PAIRS.iterdir()} | {REPROCESSED.name}
    assert names <= set(FIXES)
    assert all(FIXES[name] for name in names)


def test_catalogue_names_are_unique_and_fallback_is_last() -> None:
    names = [cause.name for cause in CAUSES]
    assert len(names) == len(set(names))
    assert CAUSES[-1] is CONTENT_CHANGED
