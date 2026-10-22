import hashlib
import json

import httpx
import pytest

from cachekeeper.adapters.inbound.http import create_app
from tests.conftest import FIXTURES, FakeUpstream, FrozenClock, MemoryStorage, make_services

pytestmark = pytest.mark.anyio

RECORDED = [
    json.loads(line).encode() for line in (FIXTURES / "requests.jsonl").read_text().split("\n") if line
]
HEADERS = {"authorization": "Bearer sk-local-test", "content-type": "application/json", "x-client": "agent"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def test_recorded_requests_are_forwarded_byte_identical(
    upstream: FakeUpstream, storage: MemoryStorage, clock: FrozenClock
) -> None:
    assert len(RECORDED) == 100
    app = create_app(make_services(upstream, storage, clock))
    direct = upstream.client()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy") as client:
        for body in RECORDED:
            proxied = await client.post("/v1/chat/completions", content=body, headers=HEADERS)
            expected = await direct.post("/v1/chat/completions", content=body, headers=HEADERS)
            assert proxied.status_code == expected.status_code
            assert proxied.content == expected.content
    sent = [received for received, _ in upstream.received[0::2]]
    assert [sha256(body) for body in sent] == [sha256(body) for body in RECORDED]
    for (_, via_proxy), (_, direct_headers) in zip(
        upstream.received[0::2], upstream.received[1::2], strict=True
    ):
        assert via_proxy["authorization"] == HEADERS["authorization"]
        hop_by_hop = {"host": "", "connection": ""}
        assert {**via_proxy, **hop_by_hop} == {**direct_headers, **hop_by_hop}
    assert len(storage.saved) == 100
    assert all(body is None for _, body in storage.saved)
