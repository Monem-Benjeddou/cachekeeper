import sqlite3
from pathlib import Path

import pytest

from cachekeeper.adapters.storage import sqlite as storage_module
from cachekeeper.adapters.storage.sqlite import SqliteStorage
from cachekeeper.core.report import RequestRecord
from cachekeeper.core.stats import measured_stats
from tests.conftest import FrozenClock

pytestmark = pytest.mark.anyio


def record(request_id: str, ts: float, model: str = "m") -> RequestRecord:
    return RequestRecord(
        request_id=request_id, ts=ts, model=model, backend="llamacpp", cause="iso_datetime",
        stats=measured_stats(10, 90, 123.0), location="sys[0]:312", before_line="a", after_line="b",
        match_id="prev", divergence_offset=320, ttft_ms=5.0, total_ms=9.0, wasted_ms=100.0, rules=("x",),
        system_hash="abc",
    )  # fmt: skip


async def test_round_trip_and_bodies(tmp_path: Path, clock: FrozenClock) -> None:
    storage = SqliteStorage(tmp_path / "db" / "c.db", clock, retention_days=7)
    storage.start()
    first = record("1", clock.now())
    storage.save(first, b'{"model": "m"}')
    storage.save(record("2", clock.now(), model="other"), None)
    await storage.stop()
    assert storage.record("1") == first
    assert storage.record("missing") is None
    assert storage.body("1") == b'{"model": "m"}'
    assert storage.body("2") is None
    assert [r.request_id for r in storage.records(0, None)] == ["1", "2"]
    assert [r.request_id for r in storage.records(0, "other")] == ["2"]
    storage.close()


async def test_save_before_start_is_dropped(tmp_path: Path, clock: FrozenClock) -> None:
    storage = SqliteStorage(tmp_path / "c.db", clock, retention_days=7)
    storage.save(record("1", clock.now()), None)
    await storage.stop()
    assert storage.records(0, None) == []


async def test_retention_deletes_old_rows_and_bodies(tmp_path: Path, clock: FrozenClock) -> None:
    path = tmp_path / "c.db"
    storage = SqliteStorage(path, clock, retention_days=7)
    storage.start()
    storage.save(record("old", clock.now()), b"secret")
    await storage.stop()
    storage.close()
    clock.advance(8 * 86400)
    reopened = SqliteStorage(path, clock, retention_days=7)
    assert reopened.record("old") is None
    assert reopened.body("old") is None


async def test_write_failure_keeps_serving_and_logs_once_per_minute(
    tmp_path: Path, clock: FrozenClock, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    storage = SqliteStorage(tmp_path / "c.db", clock, retention_days=7)

    def locked(*_args: object) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(storage, "_write", locked)
    storage.start()
    for request_id in ("1", "2"):
        storage.save(record(request_id, clock.now()), None)
    await storage.stop()
    clock.advance(61)
    storage.start()
    storage.save(record("3", clock.now()), None)
    await storage.stop()
    assert [r.getMessage() for r in caplog.records] == ["storage_write_failed", "storage_write_failed"]


async def test_full_queue_drops_record(
    tmp_path: Path, clock: FrozenClock, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(storage_module, "QUEUE_SIZE", 1)
    storage = SqliteStorage(tmp_path / "c.db", clock, retention_days=7)
    storage.start()
    storage.save(record("1", clock.now()), None)
    storage.save(record("2", clock.now()), None)
    await storage.stop()
    assert [r.request_id for r in storage.records(0, None)] == ["1"]
    assert "storage_queue_full" in caplog.text


def test_database_from_before_system_hash_is_upgraded(tmp_path: Path, clock: FrozenClock) -> None:
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(storage_module.SCHEMA.replace(", system_hash TEXT", ""))
    old.execute("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("old", clock.now(), "m", "ollama", "", 0, "", None, None, "", "new_prompt", "", "", 0, 0,
                 "unavailable", None, None, None, 0.0, ""))  # fmt: skip
    old.commit()
    old.close()
    record_from_before = SqliteStorage(path, clock, retention_days=7).record("old")
    assert record_from_before is not None
    assert record_from_before.system_hash == ""
