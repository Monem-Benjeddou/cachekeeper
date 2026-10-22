from hypothesis import given
from hypothesis import strategies as st

from cachekeeper.core.prompt import (
    Prompt,
    PromptHistory,
    canonical_prompt,
    common_prefix_length,
    find_divergence,
)


def prompt(system: str, user: str = "hi", model: str = "m") -> Prompt:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    result = canonical_prompt(body)
    assert result is not None
    return result


def test_canonical_prompt_orders_system_tools_history() -> None:
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        ],
        "tools": [{"type": "function", "function": {"name": "f"}}, "junk"],
    }
    result = canonical_prompt(body)
    assert result is not None
    assert [s.label for s in result.segments] == ["sys[1]", "tools", "msg[0]", "msg[2]"]
    assert result.tool_names == ("f",)
    assert result.segments[3].text == '[{"id":"1"}]'
    assert result.text.startswith("\n<system>\ns\n<tools>\n")


def test_canonical_prompt_flattens_content_parts() -> None:
    parts = [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "x"}}]
    result = canonical_prompt({"model": "m", "messages": [{"role": "user", "content": parts}]})
    assert result is not None
    assert result.segments[0].text == 'a\n{"type":"image_url","image_url":{"url":"x"}}'


def test_canonical_prompt_serializes_other_content_as_json() -> None:
    result = canonical_prompt({"model": "m", "messages": [{"role": "user", "content": {"k": 1}}]})
    assert result is not None
    assert result.segments[0].text == '{"k":1}'


def test_canonical_prompt_rejects_non_chat_bodies() -> None:
    assert canonical_prompt({"model": "m"}) is None
    assert canonical_prompt({"messages": []}) is None
    assert canonical_prompt({"model": "m", "messages": ["x"]}) is None


def test_common_prefix_length_examples() -> None:
    assert common_prefix_length("", "abc") == 0
    assert common_prefix_length("abc", "abd") == 2
    assert common_prefix_length("abc", "abc") == 3
    long = "x" * 10_000
    assert common_prefix_length(long + "a", long + "b") == 10_000
    assert common_prefix_length(long, long + "tail") == 10_000


@given(st.text(), st.text(), st.text())
def test_common_prefix_length_matches_definition(shared: str, a: str, b: str) -> None:
    left, right = shared + a, shared + b
    length = common_prefix_length(left, right)
    assert left[:length] == right[:length]
    assert length == min(len(left), len(right)) or left[length] != right[length]


def test_find_divergence_none_when_prefix() -> None:
    short, longer = prompt("abc"), prompt("abc", user="hi there")
    shared = common_prefix_length(short.text, longer.text)
    assert find_divergence(longer, short, shared) is None


def test_find_divergence_clips_long_lines() -> None:
    before, after = prompt("a" * 500 + "X" + "b" * 500), prompt("a" * 500 + "Y" + "b" * 500)
    divergence = find_divergence(after, before, common_prefix_length(after.text, before.text))
    assert divergence is not None
    assert divergence.location == "sys[0]:500"
    assert divergence.after_line == "a" * 120 + "Y" + "b" * 119
    assert divergence.column == 120


def test_prompt_history_picks_longest_match_per_model() -> None:
    history = PromptHistory(size=2)
    history.remember("a", prompt("agent A system"))
    history.remember("b", prompt("agent B system"))
    history.remember("other", prompt("agent A system", model="other"))
    match = history.best_match(prompt("agent A system", user="next"))
    assert match is not None
    assert match.request_id == "a"


def test_prompt_history_is_bounded() -> None:
    history = PromptHistory(size=2)
    for request_id in ("1", "2", "3"):
        history.remember(request_id, prompt(f"sys {request_id}"))
    match = history.best_match(prompt("sys 1"))
    assert match is not None
    assert match.request_id != "1"
    assert history.best_match(prompt("x", model="unknown")) is None


def test_find_divergence_at_a_part_boundary_names_the_end_of_the_previous_part() -> None:
    body = {"model": "m", "messages": [{"role": "system", "content": "short"}], "tools": [{"type": "x"}]}
    longer = {"model": "m", "messages": [{"role": "system", "content": "short and more"}]}
    current, previous = canonical_prompt(body), canonical_prompt(longer)
    assert current is not None
    assert previous is not None
    divergence = find_divergence(current, previous, common_prefix_length(current.text, previous.text))
    assert divergence is not None
    assert divergence.location == "sys[0]:5"
