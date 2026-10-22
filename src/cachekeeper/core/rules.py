from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .prompt import PromptHistory


@dataclass(frozen=True)
class Change:
    rule: str
    body: Mapping[str, object]
    before_excerpt: str
    after_excerpt: str


class Rule(Protocol):
    name: str

    def apply(self, body: Mapping[str, object], history: PromptHistory) -> Change | None: ...


@dataclass(frozen=True)
class RuleOutcome:
    body: Mapping[str, object]
    changes: tuple[Change, ...] = ()
    failed: str | None = None


RULES: tuple[Rule, ...] = ()


def apply_rules(body: Mapping[str, object], history: PromptHistory, rules: Sequence[Rule]) -> RuleOutcome:
    """Run rules in order, each on the previous one's output.

    If a rule raises, every change is dropped and the original body is kept; `failed` names the rule.
    """
    current = body
    changes: list[Change] = []
    for rule in rules:
        try:
            change = rule.apply(current, history)
        except Exception:  # noqa: BLE001 - a rule bug must never break a request
            return RuleOutcome(body, failed=rule.name)
        if change is not None:
            changes.append(change)
            current = change.body
    return RuleOutcome(current, tuple(changes))
