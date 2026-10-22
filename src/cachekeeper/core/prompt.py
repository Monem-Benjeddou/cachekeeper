import json
from bisect import bisect_right
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

SYSTEM_ROLES = frozenset({"system", "developer"})
HISTORY_SIZE = 8
WINDOW_CHARS = 120
_BLOCK_CHARS = 4096


@dataclass(frozen=True)
class Segment:
    label: str
    role: str
    start: int
    text_start: int
    text: str


@dataclass(frozen=True)
class Prompt:
    model: str
    text: str
    segments: tuple[Segment, ...]
    tool_names: tuple[str, ...]

    def segment_index(self, offset: int) -> int:
        return max(0, bisect_right([segment.start for segment in self.segments], offset) - 1)


@dataclass(frozen=True)
class Divergence:
    offset: int
    segment_index: int
    location: str
    before_line: str
    after_line: str
    column: int


@dataclass(frozen=True)
class Match:
    request_id: str
    prompt: Prompt
    shared: int


def canonical_prompt(body: Mapping[str, object]) -> Prompt | None:
    """Serialize a chat request as [system messages][tools][history], or None if it is not one.

    Each part is preceded by a `<role>` header line so a role change is a divergence too.
    """
    model, messages = body.get("model"), body.get("messages")
    if not isinstance(model, str) or not isinstance(messages, list):
        return None
    if not all(isinstance(message, dict) for message in messages):
        return None
    system = [(f"sys[{i}]", m) for i, m in enumerate(messages) if m.get("role") in SYSTEM_ROLES]
    history = [(f"msg[{i}]", m) for i, m in enumerate(messages) if m.get("role") not in SYSTEM_ROLES]
    parts = [(label, str(m.get("role")), _message_text(m)) for label, m in system]
    tools = body.get("tools")
    tool_list = tools if isinstance(tools, list) else []
    if tool_list:
        parts.append(("tools", "tools", _json(tool_list)))
    parts += [(label, str(m.get("role")), _message_text(m)) for label, m in history]
    return _assemble(model, parts, _tool_names(tool_list))


def common_prefix_length(a: str, b: str) -> int:
    """Length of the longest common prefix, in one linear pass compared block by block."""
    limit = min(len(a), len(b))
    start = 0
    while start < limit and a[start : start + _BLOCK_CHARS] == b[start : start + _BLOCK_CHARS]:
        start += _BLOCK_CHARS
    end = min(start + _BLOCK_CHARS, limit)
    while start < end and a[start] == b[start]:
        start += 1
    return min(start, limit)


def find_divergence(current: Prompt, previous: Prompt, shared: int) -> Divergence | None:
    """Locate where `current` stops matching `previous`; None when one is a prefix of the other.

    `shared` is their common prefix length. The lines are clipped to WINDOW_CHARS either side. A
    divergence right where a new part starts is reported at the end of the part before it.
    """
    if shared >= min(len(current.text), len(previous.text)):
        return None
    index = current.segment_index(shared)
    if shared == current.segments[index].start and index > 0:
        index -= 1
    segment = current.segments[index]
    start = max(current.text.rfind("\n", 0, shared) + 1, shared - WINDOW_CHARS)
    return Divergence(
        offset=shared,
        segment_index=index,
        location=f"{segment.label}:{max(0, shared - segment.text_start)}",
        before_line=_line(previous.text, start, shared),
        after_line=_line(current.text, start, shared),
        column=shared - start,
    )


class PromptHistory:
    def __init__(self, size: int = HISTORY_SIZE) -> None:
        self._size = size
        self._prompts: dict[str, deque[tuple[str, Prompt]]] = {}

    def best_match(self, prompt: Prompt) -> Match | None:
        best: Match | None = None
        for request_id, seen in reversed(self._prompts.get(prompt.model, ())):
            shared = common_prefix_length(prompt.text, seen.text)
            if best is None or shared > best.shared:
                best = Match(request_id, seen, shared)
        return best

    def remember(self, request_id: str, prompt: Prompt) -> None:
        self._prompts.setdefault(prompt.model, deque(maxlen=self._size)).append((request_id, prompt))


def _assemble(model: str, parts: list[tuple[str, str, str]], tool_names: tuple[str, ...]) -> Prompt:
    chunks: list[str] = []
    segments: list[Segment] = []
    offset = 0
    for label, role, text in parts:
        header = f"\n<{role}>\n"
        segments.append(Segment(label, role, offset, offset + len(header), text))
        chunks += (header, text)
        offset += len(header) + len(text)
    return Prompt(model, "".join(chunks), tuple(segments), tool_names)


def _message_text(message: Mapping[str, object]) -> str:
    text = _content_text(message.get("content"))
    tool_calls = message.get("tool_calls")
    return text + _json(tool_calls) if tool_calls else text


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_part_text(part) for part in content)
    return "" if content is None else _json(content)


def _part_text(part: object) -> str:
    text = part.get("text") if isinstance(part, dict) else None
    return text if isinstance(text, str) else _json(part)


def _tool_names(tools: list[object]) -> tuple[str, ...]:
    functions = [tool.get("function") for tool in tools if isinstance(tool, dict)]
    return tuple(str(function.get("name")) for function in functions if isinstance(function, dict))


def _line(text: str, start: int, offset: int) -> str:
    end = text.find("\n", offset)
    return text[start : min(len(text) if end == -1 else end, offset + WINDOW_CHARS)]


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
