import asyncio
import dataclasses
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from starlette.responses import Response

from cachekeeper.adapters.inbound.http import create_app
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


def proxy_client(upstream: FakeUpstream, storage: MemoryStorage, clock: FrozenClock) -> httpx.AsyncClient:
    app = create_app(make_services(upstream, storage, clock))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy")


def pair_body(cause: str, name: str) -> bytes:
    return (PAIRS / cause / f"{name}.json").read_bytes()


async def test_timestamp_divergence_is_reported_with_both_lines(
    storage: MemoryStorage, clock: FrozenClock, capsys: pytest.CaptureFixture[str]
) -> None:
    async with proxy_client(FakeUpstream("llamacpp"), storage, clock) as client:
        for name in ("before", "after"):
            await client.post("/v1/chat/completions", content=pair_body("iso_datetime", name))
    first, second = (record for record, _ in storage.saved)
    assert first.cause == "new_prompt"
    assert (second.cause, second.location, second.match_id) == (
        "iso_datetime",
        "sys[0]:312",
        first.request_id,
    )
    assert second.before_line == "Current time: 2026-10-21T14:03"
    assert second.after_line == "Current time: 2026-10-21T14:05"
    assert second.stats.source == "measured"
    lines = capsys.readouterr().out.splitlines()
    assert 'miss at sys[0]:312 "Current time: 2026-10-21T14:05" (iso_datetime)' in lines[1]


async def test_two_agents_do_not_pollute_each_others_matches(
    storage: MemoryStorage, clock: FrozenClock
) -> None:
    agent_a = json.loads(pair_body("uuid", "before"))
    agent_b = json.loads(pair_body("system_replaced", "after"))
    async with proxy_client(FakeUpstream("ollama"), storage, clock) as client:
        for turn in range(3):
            for agent in (agent_a, agent_b):
                agent["messages"].append({"role": "user", "content": f"turn {turn}"})
                await client.post("/v1/chat/completions", json=agent)
    records = [record for record, _ in storage.saved]
    assert [r.cause for r in records[:2]] == ["new_prompt", "system_replaced"]
    assert all(r.cause == "prefix_match" for r in records[2:])
    assert [r.match_id for r in records[2:]] == [r.request_id for r in records[:-2]]


async def test_metrics_and_health(storage: MemoryStorage, clock: FrozenClock) -> None:
    async with proxy_client(FakeUpstream("llamacpp"), storage, clock) as client:
        await client.post("/v1/chat/completions", content=pair_body("uuid", "before"))
        await client.post("/v1/chat/completions", content=b"not json")
        metrics = (await client.get("/metrics")).text
        health = (await client.get("/health")).json()
    assert "cachekeeper_requests_total 2.0" in metrics
    assert "cachekeeper_cache_hits_total 1.0" in metrics
    assert "cachekeeper_unparsed_total 1.0" in metrics
    assert "# TYPE cachekeeper_rule_applied_total counter" in metrics
    assert health == {"status": "ok", "version": "0.1.0", "backend": "llamacpp", "backend_version": "test"}


async def test_client_headers_only_reach_upstream(storage: MemoryStorage, clock: FrozenClock) -> None:
    upstream = FakeUpstream("ollama")
    app = create_app(make_services(upstream, storage, clock))
    headers = ((b"content-type", b"application/json"), (b"authorization", b"Bearer sk-x"))
    await asgi_post(app, pair_body("uuid", "before"), AsgiClient(headers=headers))
    received = upstream.received[0][1]
    assert received["authorization"] == "Bearer sk-x"
    assert not {"accept-encoding", "user-agent", "accept"} & set(received)


async def test_upstream_error_status_and_body_reach_client(
    storage: MemoryStorage, clock: FrozenClock
) -> None:
    upstream = FakeUpstream("llamacpp")
    error_body = (UPSTREAM / "llamacpp" / "error.json").read_bytes()

    async def bad_request(_body: bytes) -> Response:
        return Response(error_body, 400, media_type="application/json")

    upstream.respond = bad_request
    async with proxy_client(upstream, storage, clock) as client:
        response = await client.post("/v1/chat/completions", content=b'{"messages": "bad"}')
    assert (response.status_code, response.content) == (400, error_body)
    assert storage.saved[0][0].stats.source == "unavailable"


async def test_non_streaming_ollama_has_no_stats_and_query_passes(
    storage: MemoryStorage, clock: FrozenClock
) -> None:
    upstream = FakeUpstream("ollama")
    body = json.loads(pair_body("uuid", "before"))
    body["stream"] = False
    async with proxy_client(upstream, storage, clock) as client:
        response = await client.post("/v1/chat/completions?x=1", json=body)
    assert response.status_code == 200
    assert storage.saved[0][0].stats.source == "unavailable"


class SlowUpstream(httpx.AsyncByteStream):
    def __init__(self, client_got_first: asyncio.Event, log: list[str]) -> None:
        self.client_got_first = client_got_first
        self.log = log

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.log.append("upstream sent 1")
        yield b'data: {"n": 1}\n\n'
        await asyncio.wait_for(self.client_got_first.wait(), timeout=2)
        self.log.append("upstream sent 2")
        yield b'data: {"n": 2}\n\n'


async def test_client_receives_first_chunk_before_upstream_sends_second(
    storage: MemoryStorage, clock: FrozenClock
) -> None:
    # MockTransport streams; the ASGITransport fake buffers whole responses, so it cannot show this.
    client_got_first, log = asyncio.Event(), list[str]()
    stream = SlowUpstream(client_got_first, log)
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
    services = make_services(FakeUpstream("ollama"), storage, clock)
    services = dataclasses.replace(
        services, client=httpx.AsyncClient(transport=transport, base_url="http://u")
    )

    async def client_side(message: dict[str, Any]) -> None:
        if message.get("body"):
            log.append(f"client got {json.loads(message['body'][6:])['n']}")
            client_got_first.set()

    sent = await asgi_post(create_app(services), pair_body("uuid", "before"), AsgiClient(on_send=client_side))
    assert log == ["upstream sent 1", "client got 1", "upstream sent 2", "client got 2"]
    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [200]
