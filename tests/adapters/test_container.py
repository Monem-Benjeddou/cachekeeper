import json
import logging

import httpx
import pytest

from cachekeeper.container import CROCKFORD, JsonFormatter, detect_backend, enabled_rules, ulid
from cachekeeper.core.errors import UnsupportedBackendError
from tests.conftest import FakeUpstream, FrozenClock

pytestmark = pytest.mark.anyio


def test_ulid_is_sortable_crockford(clock: FrozenClock) -> None:
    first = ulid(clock)
    clock.advance(1)
    second = ulid(clock)
    assert len(first) == 26
    assert set(first) <= set(CROCKFORD)
    assert first[:10] < second[:10]


@pytest.mark.parametrize(
    ("backend", "version"), [("ollama", "0.30.10"), ("llamacpp", "b11669"), ("lmstudio", None)]
)
async def test_detect_backend_auto(backend: str, version: str | None) -> None:
    detected, detected_version = await detect_backend(FakeUpstream(backend).client(), "auto")
    assert (detected.name, detected_version) == (backend, version)


async def test_detect_backend_explicit_without_version_endpoint() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(404)), base_url="http://u"
    )
    detected, version = await detect_backend(client, "llamacpp")
    assert (detected.name, version) == ("llamacpp", None)
    with pytest.raises(UnsupportedBackendError):
        await detect_backend(client, "auto")


def test_enabled_rules_from_empty_registry() -> None:
    assert enabled_rules(()) == ()


def test_json_log_lines_have_fixed_keys() -> None:
    record = logging.makeLogRecord(
        {"msg": "rule_failed", "levelname": "ERROR", "request_id": "01X", "rule": "r"}
    )
    line = json.loads(JsonFormatter().format(record))
    assert list(line)[:6] == ["ts", "level", "event", "request_id", "model", "backend"]
    assert (line["event"], line["request_id"], line["model"], line["rule"]) == (
        "rule_failed",
        "01X",
        None,
        "r",
    )
