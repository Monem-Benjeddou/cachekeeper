import json
import sqlite3
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import httpx
import pytest
from starlette.responses import Response, StreamingResponse

from cachekeeper.adapters.inbound import cli
from cachekeeper.adapters.inbound.http import create_app
from cachekeeper.adapters.storage.sqlite import SqliteStorage
from cachekeeper.config import Settings
from cachekeeper.container import Services, upstream_client
from cachekeeper.core.prompt import PromptHistory
from cachekeeper.core.rules import Change
from tests.conftest import (
    PAIRS,
    UPSTREAM,
    AsgiClient,
    FakeUpstream,
    FrozenClock,
    MemoryStorage,
    asgi_post,
    make_services,
)

pytestmark = pytest.mark.anyio

CHAT = (PAIRS / "uuid" / "before.json").read_bytes()


class AddKey:
    name = "add-key"

    def apply(self, body: Mapping[str, object], history: PromptHistory) -> Change | None:
        return Change(self.name, {**body, "added": True}, "", "added")


class Broken:
    name = "broken-rule"

    def apply(self, body: Mapping[str, object], history: PromptHistory) -> Change | None:
        raise RuntimeError


async def post(services: Services, body: bytes) -> httpx.Response:
    transport = httpx.ASGITransport(create_app(services))
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        return await client.post("/v1/chat/completions", content=body)


async def test_non_json_is_forwarded_unchanged_and_recorded_unparsed(
    storage: MemoryStorage, clock: FrozenClock
) -> None:
    upstream = FakeUpstream("ollama")
    for body in (b"\xff\xfe not json", b'{"model": "m", "prompt": "completion style"}'):
        await post(make_services(upstream, storage, clock), body)
        assert upstream.received[-1][0] == body
    assert [record.cause for record, _ in storage.saved] == ["unparsed", "unparsed"]


@pytest.mark.parametrize(
    ("error", "status"),
    [(httpx.ConnectError("connection refused"), 502), (httpx.ReadTimeout("timed out"), 504)],
)
async def test_upstream_down_or_timeout(
    error: Exception, status: int, storage: MemoryStorage, clock: FrozenClock
) -> None:
    upstream = FakeUpstream("ollama")

    async def fail(_body: bytes) -> Response:
        raise error

    upstream.respond = fail
    response = await post(make_services(upstream, storage, clock), CHAT)
    assert response.status_code == status
    assert str(error) in response.json()["error"]["message"]
    assert storage.saved[0][0].stats.source == "unavailable"


def test_upstream_client_never_waits_past_the_configured_timeout() -> None:
    client = upstream_client(Settings(upstream_timeout_seconds=12))
    assert client.timeout == httpx.Timeout(12)


async def test_rewritten_request_rejected_upstream_is_retried_with_original(
    storage: MemoryStorage, clock: FrozenClock, caplog: pytest.LogCaptureFixture
) -> None:
    upstream = FakeUpstream("ollama")

    async def reject_rewrites(body: bytes) -> Response:
        return Response(b"bad", 400) if b'"added"' in body else Response(b'{"ok": true}', 200)

    upstream.respond = reject_rewrites
    response = await post(make_services(upstream, storage, clock, rules=(AddKey(),)), CHAT)
    assert (response.status_code, response.content) == (200, b'{"ok": true}')
    assert [body for body, _ in upstream.received] == [
        json.dumps({**json.loads(CHAT), "added": True}).encode(),
        CHAT,
    ]
    assert "rewrite_reverted" in caplog.text
    assert storage.saved[0][0].rules == ()


async def test_accepted_rewrite_is_forwarded_and_recorded(storage: MemoryStorage, clock: FrozenClock) -> None:
    upstream = FakeUpstream("ollama")
    await post(make_services(upstream, storage, clock, rules=(AddKey(),)), CHAT)
    assert json.loads(upstream.received[0][0])["added"] is True
    assert storage.saved[0][0].rules == ("add-key",)


async def test_storage_write_failure_keeps_serving(
    tmp_path: Path, clock: FrozenClock, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    storage = SqliteStorage(tmp_path / "c.db", clock, retention_days=7)

    def disk_full(*_args: object) -> None:
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(storage, "_write", disk_full)
    upstream = FakeUpstream("llamacpp")
    services = make_services(upstream, MemoryStorage(), clock)
    storage.start()
    response = await post(Services(**{**services.__dict__, "storage": storage}), CHAT)
    await storage.stop()
    assert response.status_code == 200
    assert "storage_write_failed" in caplog.text


@pytest.mark.parametrize("fixture", ["llamacpp/error.json", "ollama/malformed.sse"])
async def test_missing_or_malformed_stats_are_unavailable(
    fixture: str, storage: MemoryStorage, clock: FrozenClock, capsys: pytest.CaptureFixture[str]
) -> None:
    upstream = FakeUpstream(fixture.split("/", maxsplit=1)[0])
    payload = (UPSTREAM / fixture).read_bytes()

    async def respond(_body: bytes) -> Response:
        return Response(payload, 200, media_type="text/event-stream")

    upstream.respond = respond
    response = await post(make_services(upstream, storage, clock), CHAT)
    assert response.content == payload
    assert storage.saved[0][0].stats.source == "unavailable"
    assert "hit n/a" in capsys.readouterr().out


async def test_rule_that_raises_forwards_original(
    storage: MemoryStorage, clock: FrozenClock, caplog: pytest.LogCaptureFixture
) -> None:
    upstream = FakeUpstream("ollama")
    await post(make_services(upstream, storage, clock, rules=(AddKey(), Broken())), CHAT)
    assert upstream.received[0][0] == CHAT
    assert any(
        r.getMessage() == "rule_failed" and r.__dict__["rule"] == "broken-rule" for r in caplog.records
    )


@pytest.mark.parametrize("spec_version", ["2.0", "2.4"])
async def test_client_disconnect_mid_stream_closes_upstream(
    spec_version: str, storage: MemoryStorage, clock: FrozenClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = FakeUpstream("ollama")

    async def chunks() -> AsyncIterator[bytes]:
        for index in range(200):
            yield f'data: {{"n": {index}}}\n\n'.encode()

    async def stream(_body: bytes) -> Response:
        return StreamingResponse(chunks(), media_type="text/event-stream")

    closed: list[bool] = []
    original_aclose = httpx.Response.aclose

    async def spy(self: httpx.Response) -> None:
        closed.append(True)
        await original_aclose(self)

    monkeypatch.setattr(httpx.Response, "aclose", spy)
    upstream.respond = stream
    app = create_app(make_services(upstream, storage, clock))
    sent = await asgi_post(app, CHAT, AsgiClient(disconnect=spec_version))
    body_messages = [m for m in sent if m["type"] == "http.response.body"]
    assert 0 < len(body_messages) < 200
    assert closed
    assert len(storage.saved) == 1


def test_invalid_config_fails_at_startup_listing_every_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CACHEKEEPER_RETENTION_DAYS", "0")
    monkeypatch.setenv("CACHEKEEPER_UPSTREAM_TIMEOUT_SECONDS", "soon")
    assert cli.main(["serve", "--port", "70000", "--rule", "nope"]) == 2
    errors = capsys.readouterr().err.splitlines()
    assert len(errors) == 4
    assert {line.split(":")[1].strip() for line in errors} == {
        "retention_days",
        "upstream_timeout_seconds",
        "port",
        "rules",
    }
