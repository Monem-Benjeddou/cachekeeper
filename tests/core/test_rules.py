from collections.abc import Mapping

from cachekeeper.core.prompt import PromptHistory
from cachekeeper.core.rules import RULES, Change, apply_rules


class AddKey:
    def __init__(self, name: str, key: str) -> None:
        self.name = name
        self.key = key

    def apply(self, body: Mapping[str, object], history: PromptHistory) -> Change | None:
        if self.key in body:
            return None
        return Change(self.name, {**body, self.key: True}, "", self.key)


class Broken:
    name = "broken"

    def apply(self, body: Mapping[str, object], history: PromptHistory) -> Change | None:
        raise ValueError(body)


def test_registry_ships_no_rules() -> None:
    assert RULES == ()


def test_apply_rules_without_rules_returns_same_body() -> None:
    body = {"model": "m"}
    outcome = apply_rules(body, PromptHistory(), RULES)
    assert outcome.body is body
    assert outcome.changes == ()


def test_apply_rules_chains_in_order() -> None:
    outcome = apply_rules({"a": True}, PromptHistory(), [AddKey("one", "a"), AddKey("two", "b")])
    assert outcome.body == {"a": True, "b": True}
    assert [change.rule for change in outcome.changes] == ["two"]


def test_apply_rules_keeps_original_when_a_rule_raises() -> None:
    body = {"model": "m"}
    outcome = apply_rules(body, PromptHistory(), [AddKey("one", "x"), Broken()])
    assert outcome.body is body
    assert outcome.changes == ()
    assert outcome.failed == "broken"
