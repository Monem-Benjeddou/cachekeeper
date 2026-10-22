from collections.abc import Callable

import httpx
import pytest

from cachekeeper.adapters.inbound.http import create_app
from cachekeeper.core.report import build_report, render_report
from tests.conftest import PAIRS, FakeUpstream, FrozenClock, MemoryStorage, make_services

SCRIPT = [
    (cause, name) for cause in ("iso_datetime", "tool_order", "hex_nonce") for name in ("before", "after")
]


@pytest.mark.anyio
async def test_status_lines_and_report(
    clock: FrozenClock, capsys: pytest.CaptureFixture[str], golden: Callable[[str, str], None]
) -> None:
    storage = MemoryStorage()
    app = create_app(make_services(FakeUpstream("llamacpp"), storage, clock))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy") as client:
        for cause, name in SCRIPT:
            await client.post("/v1/chat/completions", content=(PAIRS / cause / f"{name}.json").read_bytes())
            clock.advance(1)
    golden("status.txt", capsys.readouterr().out)
    golden("report.txt", render_report(build_report([record for record, _ in storage.saved])) + "\n")
