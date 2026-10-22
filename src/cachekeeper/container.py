import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from cachekeeper.adapters.backends.llamacpp import LlamaCppBackend
from cachekeeper.adapters.backends.lmstudio import LmStudioBackend
from cachekeeper.adapters.backends.ollama import OllamaBackend
from cachekeeper.config import Settings
from cachekeeper.core.errors import UnsupportedBackendError
from cachekeeper.core.ports import Backend, Clock, Storage
from cachekeeper.core.rules import RULES, Rule

BACKENDS: dict[str, Callable[[], Backend]] = {
    "ollama": OllamaBackend,
    "llamacpp": LlamaCppBackend,
    "lmstudio": LmStudioBackend,
}
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
DETECT_TIMEOUT_SECONDS = 5
LOG_KEYS = ("request_id", "model", "backend")
_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "color_message"}


class SystemClock:
    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()


@dataclass(frozen=True)
class Services:
    settings: Settings
    backend: Backend
    backend_version: str | None
    storage: Storage
    clock: Clock
    client: httpx.AsyncClient
    new_id: Callable[[], str]
    rules: tuple[Rule, ...] = ()


def ulid(clock: Clock) -> str:
    value = (int(clock.now() * 1000) << 80) | int.from_bytes(os.urandom(10), "big")
    return "".join(CROCKFORD[(value >> shift) & 31] for shift in range(125, -1, -5))


def enabled_rules(names: tuple[str, ...]) -> tuple[Rule, ...]:
    return tuple(rule for rule in RULES if rule.name in names)


def upstream_client(settings: Settings) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=settings.upstream, timeout=settings.upstream_timeout_seconds)


async def detect_backend(client: httpx.AsyncClient, name: str) -> tuple[Backend, str | None]:
    candidates = list(BACKENDS) if name == "auto" else [name]
    for candidate in candidates:
        backend = BACKENDS[candidate]()
        try:
            response = await client.get(backend.version_path, timeout=DETECT_TIMEOUT_SECONDS)
            payload = response.json() if response.is_success else None
        except (httpx.HTTPError, ValueError):
            payload = None
        if payload is not None:
            return backend, backend.version(payload)
        if name != "auto":
            return backend, None
    raise UnsupportedBackendError(f"no Ollama, llama.cpp or LM Studio answered at {client.base_url}")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = {key: value for key, value in record.__dict__.items() if key not in _RECORD_FIELDS}
        line = {
            "ts": record.created,
            "level": record.levelname.lower(),
            "event": record.getMessage(),
            **{key: fields.pop(key, None) for key in LOG_KEYS},
            **fields,
        }
        return json.dumps(line, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
