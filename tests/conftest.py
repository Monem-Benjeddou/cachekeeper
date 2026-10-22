import itertools
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from cachekeeper.config import Settings
from cachekeeper.container import BACKENDS, Services
from cachekeeper.core.report import RequestRecord
from cachekeeper.core.rules import Rule

FIXTURES = Path(__file__).parent / "fixtures"
PAIRS = FIXTURES / "pairs"
UPSTREAM = FIXTURES / "upstream"
BACKEND_NAMES = ("ollama", "llamacpp", "lmstudio")
VERSIONS = {"ollama": {"version": "0.30.10"}, "llamacpp": {"build_info": "b11669"}, "lmstudio": {"data": []}}
Responder = Callable[[bytes], Awaitable[Response]]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


class FrozenClock:
    def __init__(self, now: float = 1792591391.0) -> None:
        self._now = now
        self._monotonic = 0.0

    def now(self) -> float:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> None:
        self._now += seconds
        self._monotonic += seconds


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--update-golden", action="store_true", help="rewrite golden files from current output")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock()


@pytest.fixture
def golden(request: pytest.FixtureRequest) -> Callable[[str, str], None]:
    update = bool(request.config.getoption("--update-golden"))

    def check(name: str, actual: str) -> None:
        path = Path(__file__).parent / "golden" / name
        if update:
            path.write_text(actual)
        assert actual == path.read_text()

    return check


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeUpstream:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.received: list[tuple[bytes, dict[str, str]]] = []
        self.respond: Responder | None = None
        version_path = BACKENDS[backend]().version_path
        self.app = Starlette(
            routes=[
                Route("/v1/chat/completions", self.chat, methods=["POST"]),
                Route("/api/chat", self.chat, methods=["POST"]),
                Route(version_path, self.version),
            ]
        )

    async def chat(self, request: Request) -> Response:
        body = await request.body()
        self.received.append((body, dict(request.headers)))
        if self.respond is not None:
            return await self.respond(body)
        streaming = b'"stream": true' in body or b'"stream":true' in body
        name = "chat.sse" if streaming else "chat.json"
        media = "text/event-stream" if streaming else "application/json"
        return Response((UPSTREAM / self.backend / name).read_bytes(), media_type=media)

    async def version(self, _request: Request) -> Response:
        return JSONResponse(VERSIONS[self.backend])

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(self.app), base_url="http://upstream")


class MemoryStorage:
    def __init__(self) -> None:
        self.saved: list[tuple[RequestRecord, bytes | None]] = []

    def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def save(self, record: RequestRecord, body: bytes | None) -> None:
        self.saved.append((record, body))

    def records(self, since: float, model: str | None) -> list[RequestRecord]:
        return [r for r, _ in self.saved if r.ts >= since and model in (None, r.model)]

    def record(self, request_id: str) -> RequestRecord | None:
        return next((r for r, _ in self.saved if r.request_id == request_id), None)

    def body(self, request_id: str) -> bytes | None:
        return next((b for r, b in self.saved if r.request_id == request_id), None)


def make_services(
    upstream: FakeUpstream, storage: MemoryStorage, clock: FrozenClock, rules: tuple[Rule, ...] = ()
) -> Services:
    ids = itertools.count(1)
    return Services(
        settings=Settings(upstream="http://upstream"),
        backend=BACKENDS[upstream.backend](),
        backend_version="test",
        storage=storage,
        clock=clock,
        client=upstream.client(),
        new_id=lambda: f"01TEST{next(ids):020d}",
        rules=rules,
    )


@pytest.fixture(params=BACKEND_NAMES)
def upstream(request: pytest.FixtureRequest) -> FakeUpstream:
    return FakeUpstream(request.param)


@pytest.fixture
def storage() -> MemoryStorage:
    return MemoryStorage()


@dataclass(frozen=True)
class AsgiClient:
    headers: tuple[tuple[bytes, bytes], ...] = ()
    disconnect: str = ""
    on_send: Callable[[dict[str, Any]], Awaitable[None]] | None = None


PLAIN_CLIENT = AsgiClient()


async def asgi_post(
    app: Callable[..., Awaitable[None]], body: bytes, client: AsgiClient = PLAIN_CLIENT
) -> list[dict[str, Any]]:
    headers, disconnect, on_send = list(client.headers), client.disconnect, client.on_send
    import asyncio  # noqa: PLC0415
    import contextlib  # noqa: PLC0415

    from starlette.requests import ClientDisconnect  # noqa: PLC0415

    sent: list[dict[str, Any]] = []
    pending = [{"type": "http.request", "body": body, "more_body": False}]
    gone = asyncio.Event()

    async def receive() -> dict[str, Any]:
        if pending:
            return pending.pop()
        await (gone if disconnect else asyncio.Event()).wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if disconnect >= "2.4" and gone.is_set():
            raise OSError("client went away")
        sent.append(message)
        if on_send is not None:
            await on_send(message)
        if disconnect and message["type"] == "http.response.body":
            gone.set()
        await asyncio.sleep(0)

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": disconnect or "2.0"},
        "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions", "query_string": b"", "root_path": "", "headers": headers,
        "server": ("proxy", 80), "client": ("agent", 1),
    }  # fmt: skip
    with contextlib.suppress(ClientDisconnect):
        await app(scope, receive, send)
    return sent
