from dataclasses import dataclass

from .causes import CAUSES, NEW_PROMPT, PREFIX_MATCH, REPROCESSED
from .prompt import Divergence, Match, Prompt, PromptHistory, find_divergence
from .stats import CacheStats


@dataclass(frozen=True)
class Diagnosis:
    prompt: Prompt
    match: Match | None
    divergence: Divergence | None
    cause: str


def diagnose(prompt: Prompt, history: PromptHistory) -> Diagnosis:
    """Find the closest recent prompt for the same model, where the two diverge, and why."""
    match = history.best_match(prompt)
    if match is None:
        return Diagnosis(prompt, None, None, NEW_PROMPT)
    divergence = find_divergence(prompt, match.prompt, match.shared)
    return Diagnosis(prompt, match, divergence, classify(divergence, prompt, match.prompt))


def classify(divergence: Divergence | None, current: Prompt, previous: Prompt) -> str:
    """Name the first cause in CAUSES whose detector matches the divergence."""
    if divergence is None:
        return PREFIX_MATCH
    return next(cause.name for cause in CAUSES if cause.detect(divergence, current, previous))


def confirm(cause: str, stats: CacheStats) -> str:
    """Turn an unchanged prefix into REPROCESSED when the backend measured no reuse at all."""
    if cause == PREFIX_MATCH and stats.source == "measured" and stats.reused == 0 and stats.processed:
        return REPROCESSED.name
    return cause
