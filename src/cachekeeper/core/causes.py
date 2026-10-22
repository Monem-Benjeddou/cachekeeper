import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .prompt import Divergence, Prompt, Segment

Detector = Callable[[Divergence, Prompt, Prompt], bool]

NEW_PROMPT = "new_prompt"
PREFIX_MATCH = "prefix_match"
UNPARSED = "unparsed"
SKIPPED_LARGE = "skipped_large"
NOT_MISSES = frozenset({NEW_PROMPT, PREFIX_MATCH, UNPARSED, SKIPPED_LARGE})

_HEAD_CHARS = 64
_REPLACED_SHARE = 0.1


@dataclass(frozen=True)
class Cause:
    name: str
    fix: str
    detect: Detector


def _same_position(
    divergence: Divergence, current: Prompt, previous: Prompt
) -> tuple[Segment, Segment | None]:
    now = current.segments[divergence.segment_index]
    return now, next((s for s in previous.segments if s.label == now.label), None)


def _history(prompt: Prompt) -> list[Segment]:
    return [s for s in prompt.segments if s.label.startswith("msg")]


def _tools_reordered(divergence: Divergence, current: Prompt, previous: Prompt) -> bool:
    now, _ = _same_position(divergence, current, previous)
    reordered = current.tool_names != previous.tool_names
    return now.label == "tools" and reordered and sorted(current.tool_names) == sorted(previous.tool_names)


def _leading_history_dropped(divergence: Divergence, current: Prompt, previous: Prompt) -> bool:
    now = [(s.role, s.text) for s in _history(current)]
    before = [(s.role, s.text) for s in _history(previous)]
    in_history = current.segments[divergence.segment_index].label.startswith("msg")
    return in_history and bool(now) and bool(before) and now[0] != before[0] and now[0] in before[1:]


def _history_folded(divergence: Divergence, current: Prompt, previous: Prompt) -> bool:
    now, before = _same_position(divergence, current, previous)
    if before is None or not now.label.startswith("msg"):
        return False
    kept = {(s.role, s.text) for s in current.segments}
    following = previous.segments[previous.segments.index(before) + 1 :][:1]
    folded = [s for s in (before, *following) if s.label.startswith("msg") and (s.role, s.text) not in kept]
    return len(folded) == 1 + len(following) and len(_history(current)) < len(_history(previous))


def _block_moved_later(divergence: Divergence, current: Prompt, previous: Prompt) -> bool:
    now, before = _same_position(divergence, current, previous)
    if before is None or not now.label.startswith("msg"):
        return False
    head = before.text[:_HEAD_CHARS]
    later = current.segments[divergence.segment_index + 1 :]
    return any(s.role == before.role and s.text.startswith(head) for s in later)


def _whitespace_only(divergence: Divergence, current: Prompt, previous: Prompt) -> bool:
    now, before = _same_position(divergence, current, previous)
    same_text = before is not None and "".join(now.text.split()) == "".join(before.text.split())
    return same_text and before is not None and now.role == before.role


def _token_at_divergence(pattern: str) -> Detector:
    compiled = re.compile(pattern)

    def detect(divergence: Divergence, _current: Prompt, _previous: Prompt) -> bool:
        lines = (divergence.after_line, divergence.before_line)
        return any(
            m.start() <= divergence.column <= m.end() for line in lines for m in compiled.finditer(line)
        )

    return detect


def _system_prompt_swapped(divergence: Divergence, current: Prompt, previous: Prompt) -> bool:
    now, before = _same_position(divergence, current, previous)
    if before is None or not now.label.startswith("sys"):
        return False
    shared = divergence.offset - now.text_start
    early = shared < max(_HEAD_CHARS, min(len(now.text), len(before.text)) * _REPLACED_SHARE)
    return early and now.text[-_HEAD_CHARS:] != before.text[-_HEAD_CHARS:]


def _always(_divergence: Divergence, _current: Prompt, _previous: Prompt) -> bool:
    return True


TOOL_ORDER = Cause(
    "tool_order",
    "Emit tools in a deterministic order (qwen-code #8277)",
    _tools_reordered,
)
HISTORY_TRUNCATED = Cause(
    "history_truncated",
    "Truncate or summarize history once instead of dropping the oldest turn every request "
    "(smfclearinghouse prefix-caching guide)",
    _leading_history_dropped,
)
HISTORY_SUMMARIZED = Cause(
    "history_summarized",
    "Older turns were folded into a summary, which shifts everything after it; summarize rarely "
    "and in large steps (Aider: raise --max-chat-history-tokens)",
    _history_folded,
)
BLOCK_AFTER_HISTORY = Cause(
    "block_after_history",
    "New turns are inserted before a block the client re-sends after the history (Aider: the "
    "files in the chat), so the block is re-processed every turn; send it before the history",
    _block_moved_later,
)
WHITESPACE = Cause(
    "whitespace",
    "Serialize messages with stable whitespace and JSON formatting (smfclearinghouse prefix-caching guide)",
    _whitespace_only,
)
UUID = Cause(
    "uuid",
    "Move per-request IDs out of the prompt prefix",
    _token_at_divergence(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"),
)
ISO_DATETIME = Cause(
    "iso_datetime",
    "Use a date-only line or move the time to the end of the prompt (hermes-agent #8687, merged PR #27675)",
    _token_at_divergence(
        r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    ),
)
UNIX_EPOCH = Cause(
    "unix_epoch",
    "Move the timestamp to the end of the prompt or drop it (hermes-agent #8687)",
    _token_at_divergence(r"\b1\d{9}(?:\d{3})?\b"),
)
HEX_NONCE = Cause(
    "hex_nonce",
    "Per-request hash near the top; for Claude Code set CLAUDE_CODE_ATTRIBUTION_HEADER=0 "
    "(claude-code-router PR #1220)",
    _token_at_divergence(r"\b[0-9a-fA-F]{12,}\b"),
)
SYSTEM_REPLACED = Cause(
    "system_replaced",
    "A side request with another system prompt evicts the cache on single-slot backends; give it a "
    "separate slot or model (hermes-agent #8687; Aider: --weak-model)",
    _system_prompt_swapped,
)
CONTENT_CHANGED = Cause(
    "content_changed",
    "An earlier message was edited; keep sent history immutable (Aider rewrites two messages in place: "
    "the repo map, try --map-refresh manual; the history summary, raise --max-chat-history-tokens)",
    _always,
)
REPROCESSED = Cause(
    "reprocessed",
    "Model architecture forces re-processing (hybrid or sliding-window, "
    "https://github.com/ggml-org/llama.cpp/issues/24714), or the slot was evicted",
    _always,
)

# Checked in this order; the first match names the cause, so CONTENT_CHANGED must stay last.
CAUSES: tuple[Cause, ...] = (
    TOOL_ORDER,
    HISTORY_TRUNCATED,
    HISTORY_SUMMARIZED,
    BLOCK_AFTER_HISTORY,
    WHITESPACE,
    UUID,
    ISO_DATETIME,
    UNIX_EPOCH,
    HEX_NONCE,
    SYSTEM_REPLACED,
    CONTENT_CHANGED,
)

FIXES: Mapping[str, str] = MappingProxyType({cause.name: cause.fix for cause in (*CAUSES, REPROCESSED)})
