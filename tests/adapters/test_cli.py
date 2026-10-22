import argparse
from pathlib import Path

import httpx
import pytest

from cachekeeper import container
from cachekeeper.adapters.inbound import cli
from cachekeeper.adapters.storage.sqlite import SqliteStorage
from cachekeeper.config import Settings
from cachekeeper.core.report import RequestRecord
from cachekeeper.core.stats import measured_stats
from tests.conftest import FakeUpstream


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "c.db"
    monkeypatch.setenv("CACHEKEEPER_DB_PATH", str(path))
    return path


async def fill(path: Path, bodies: dict[str, bytes]) -> None:
    storage = SqliteStorage(path, container.SystemClock(), retention_days=7)
    storage.start()
    first = RequestRecord(
        request_id="A", ts=container.SystemClock().now(), model="m", backend="ollama", cause="new_prompt"
    )
    second = RequestRecord(
        request_id="B", ts=first.ts, model="m", backend="ollama", cause="iso_datetime",
        stats=measured_stats(0, 100, 2000.0), match_id="A", location="sys[0]:312",
        before_line="Current time: 14:03", after_line="Current time: 14:05", wasted_ms=1800.0,
    )  # fmt: skip
    for record in (first, second):
        storage.save(record, bodies.get(record.request_id))
    await storage.stop()
    storage.close()


@pytest.mark.anyio
async def test_report_and_diff(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    await fill(db, {"A": b'{"messages": ["t 14:03"]}', "B": b'{"messages": ["t 14:05"]}'})
    assert cli.main(["report", "--since", "1h"]) == 0
    report = capsys.readouterr().out
    assert "requests        2" in report
    assert "1. iso_datetime: 1 requests, 1.8 s wasted" in report
    assert cli.main(["diff", "B"]) == 0
    diff = capsys.readouterr().out
    assert "cause     iso_datetime at sys[0]:312 (chars)" in diff
    assert '-    "t 14:03"' in diff
    assert '+    "t 14:05"' in diff


@pytest.mark.anyio
async def test_diff_without_bodies_and_unknown_id(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    await fill(db, {"B": b"raw\nbody"})
    assert cli.main(["diff", "A"]) == 0
    assert "matched   none" in capsys.readouterr().out
    assert cli.main(["diff", "nope"]) == 1
    assert cli.main(["report", "--model", "other"]) == 0
    assert "requests        0" in capsys.readouterr().out


def test_duration_parsing() -> None:
    assert cli.duration_seconds("90s") == 90
    assert cli.duration_seconds("7d") == 7 * 86400
    with pytest.raises(argparse.ArgumentTypeError):
        cli.duration_seconds("soon")


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert capsys.readouterr().out.strip() == "cachekeeper 0.1.0"


@pytest.mark.parametrize("backend", ["ollama", "llamacpp"])
def test_probe_prints_the_measurement_table(
    backend: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    upstream = FakeUpstream(backend)
    monkeypatch.setattr(container, "upstream_client", lambda _settings: upstream.client())
    assert cli.main(["probe", "--model", "qwen3:4b"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"{backend} ")
    for label in ("Stable prompt, cold", "Stable prompt, warm", "turn 1", "turn 2 (timestamp changed)"):
        assert label in out
    systems = [body for body, _ in upstream.received]
    assert len(systems) == 4
    assert systems[0] == systems[1]
    assert b"Current time: " in systems[2]


def test_probe_reports_upstream_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = httpx.MockTransport(refuse)
    monkeypatch.setattr(
        container, "upstream_client", lambda _s: httpx.AsyncClient(transport=transport, base_url="http://u")
    )
    assert cli.main(["probe", "--model", "m", "--backend", "ollama"]) == 1
    assert "probe failed" in capsys.readouterr().err


def test_probe_reports_http_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(404, json={"error": "model not found"}))
    monkeypatch.setattr(
        container, "upstream_client", lambda _s: httpx.AsyncClient(transport=transport, base_url="http://u")
    )
    assert cli.main(["probe", "--model", "m", "--backend", "llamacpp"]) == 1
    assert "model not found" in capsys.readouterr().err


def test_serve_fails_at_startup_when_no_backend_answers(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(404))
    monkeypatch.setattr(
        container, "upstream_client", lambda _s: httpx.AsyncClient(transport=transport, base_url="http://u")
    )
    monkeypatch.setattr(container, "configure_logging", lambda: None)
    assert cli.main(["serve"]) == 1
    assert "startup_failed" in caplog.text


def test_settings_cli_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CACHEKEEPER_UPSTREAM", "http://env:1")
    monkeypatch.setenv("CACHEKEEPER_PORT", "")
    args = cli.parser().parse_args(["serve", "--backend", "ollama"])
    settings = cli.load_settings(args)
    assert settings == Settings(upstream="http://env:1", backend="ollama")
