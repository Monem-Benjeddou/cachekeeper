import json
import time

import httpx
import pytest

from cachekeeper.adapters.inbound.http import create_app
from tests.conftest import FakeUpstream, FrozenClock, MemoryStorage, make_services

pytestmark = [pytest.mark.perf, pytest.mark.anyio]

RUNS = 50
BUDGET_MS = 5.0
CHARS_PER_TOKEN = 4


def thirty_k_token_body(turn: int) -> bytes:
    system = "".join(f"Rule {i}: keep module {i % 97} small and tested.\n" for i in range(2800))
    assert len(system) >= 30_000 * CHARS_PER_TOKEN * 0.9
    messages = [{"role": "system", "content": f"Current time: 2026-10-21T14:{turn % 60:02d}\n" + system}]
    messages += [{"role": "user", "content": f"turn {turn}"}]
    return json.dumps({"model": "qwen3:4b", "messages": messages, "stream": True}).encode()


def p95(samples: list[float]) -> float:
    return sorted(samples)[int(len(samples) * 0.95) - 1]


async def timed(client: httpx.AsyncClient, body: bytes) -> float:
    started = time.perf_counter()
    response = await client.post("/v1/chat/completions", content=body)
    assert response.status_code == 200
    return (time.perf_counter() - started) * 1000


async def test_proxy_adds_under_5ms_at_p95_for_30k_tokens(clock: FrozenClock) -> None:
    upstream = FakeUpstream("llamacpp")
    app = create_app(make_services(upstream, MemoryStorage(), clock))
    direct, proxied = [], []
    async with (
        upstream.client() as upstream_client,
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy") as proxy_client,
    ):
        for turn in range(RUNS + 5):
            body = thirty_k_token_body(turn)
            direct_ms, proxied_ms = await timed(upstream_client, body), await timed(proxy_client, body)
            if turn >= 5:
                direct.append(direct_ms)
                proxied.append(proxied_ms)
    overhead = p95(proxied) - p95(direct)
    print(f"p95 direct {p95(direct):.2f} ms, proxied {p95(proxied):.2f} ms, overhead {overhead:.2f} ms")  # noqa: T201
    assert overhead < BUDGET_MS
