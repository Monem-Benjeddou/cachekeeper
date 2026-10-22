import asyncio
import contextlib
import logging
import sqlite3
from pathlib import Path

from cachekeeper.core.ports import Clock
from cachekeeper.core.report import RequestRecord
from cachekeeper.core.stats import CacheStats

LOG = logging.getLogger("cachekeeper")
QUEUE_SIZE = 1000
FAILURE_LOG_SECONDS = 60
RETENTION_CHECK_SECONDS = 3600
SECONDS_PER_DAY = 86400

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY, ts REAL, model TEXT, backend TEXT, client_hint TEXT, prompt_chars INTEGER,
    prompt_hash TEXT, match_id TEXT, divergence_offset INTEGER, location TEXT, cause TEXT,
    before_line TEXT, after_line TEXT, reused INTEGER, processed INTEGER, stats_source TEXT,
    prefill_ms REAL, ttft_ms REAL, total_ms REAL, wasted_ms REAL, rules TEXT, system_hash TEXT);
CREATE INDEX IF NOT EXISTS requests_ts ON requests (ts);
CREATE TABLE IF NOT EXISTS changes (request_id TEXT, rule TEXT, before_excerpt TEXT, after_excerpt TEXT);
CREATE TABLE IF NOT EXISTS bodies (request_id TEXT PRIMARY KEY, original BLOB, rewritten BLOB);
"""

INSERT_REQUEST = "INSERT OR REPLACE INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"

Pending = tuple[RequestRecord, bytes | None]


class SqliteStorage:
    def __init__(self, path: Path, clock: Clock, retention_days: int) -> None:
        self._clock = clock
        self._retention_days = retention_days
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        if "system_hash" not in {row["name"] for row in self._db.execute("PRAGMA table_info(requests)")}:
            self._db.execute("ALTER TABLE requests ADD COLUMN system_hash TEXT")
        self._queue: asyncio.Queue[Pending] | None = None
        self._writer: asyncio.Task[None] | None = None
        self._last_failure_log = float("-inf")
        self._last_retention = float("-inf")
        self._enforce_retention()

    def start(self) -> None:
        self._queue = asyncio.Queue(QUEUE_SIZE)
        self._writer = asyncio.create_task(self._drain(self._queue))

    async def stop(self) -> None:
        if self._queue is None or self._writer is None:
            return
        await self._queue.join()
        self._writer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._writer

    def close(self) -> None:
        self._db.close()

    def save(self, record: RequestRecord, body: bytes | None) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait((record, body))
        except asyncio.QueueFull:
            self._failed("storage_queue_full", record.request_id)

    def records(self, since: float, model: str | None) -> list[RequestRecord]:
        query = "SELECT * FROM requests WHERE ts >= ? AND (? IS NULL OR model = ?) ORDER BY ts"
        return [_record(row) for row in self._db.execute(query, (since, model, model))]

    def record(self, request_id: str) -> RequestRecord | None:
        row = self._db.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
        return None if row is None else _record(row)

    def body(self, request_id: str) -> bytes | None:
        row = self._db.execute("SELECT original FROM bodies WHERE request_id = ?", (request_id,)).fetchone()
        return None if row is None else bytes(row["original"])

    async def _drain(self, queue: asyncio.Queue[Pending]) -> None:
        while True:
            record, body = await queue.get()
            try:
                await asyncio.to_thread(self._write, record, body)
            except (sqlite3.Error, OSError):
                self._failed("storage_write_failed", record.request_id)
            finally:
                queue.task_done()

    def _write(self, record: RequestRecord, body: bytes | None) -> None:
        with self._db:
            self._db.execute(INSERT_REQUEST, _row(record))
            if body is not None:
                self._db.execute(
                    "INSERT OR REPLACE INTO bodies VALUES (?, ?, NULL)", (record.request_id, body)
                )
        self._enforce_retention()

    def _enforce_retention(self) -> None:
        now = self._clock.now()
        if now - self._last_retention < RETENTION_CHECK_SECONDS:
            return
        self._last_retention = now
        cutoff = now - self._retention_days * SECONDS_PER_DAY
        with self._db:
            self._db.execute(
                "DELETE FROM bodies WHERE request_id IN (SELECT id FROM requests WHERE ts < ?)", (cutoff,)
            )
            self._db.execute(
                "DELETE FROM changes WHERE request_id IN (SELECT id FROM requests WHERE ts < ?)", (cutoff,)
            )
            self._db.execute("DELETE FROM requests WHERE ts < ?", (cutoff,))

    def _failed(self, event: str, request_id: str) -> None:
        now = self._clock.now()
        if now - self._last_failure_log >= FAILURE_LOG_SECONDS:
            self._last_failure_log = now
            LOG.error(event, extra={"request_id": request_id})


def _row(r: RequestRecord) -> tuple[object, ...]:
    s = r.stats
    return (
        r.request_id, r.ts, r.model, r.backend, r.client_hint, r.prompt_chars, r.prompt_hash, r.match_id,
        r.divergence_offset, r.location, r.cause, r.before_line, r.after_line, s.reused, s.processed,
        s.source, s.prefill_ms, r.ttft_ms, r.total_ms, r.wasted_ms, ",".join(r.rules), r.system_hash,
    )  # fmt: skip


def _record(row: sqlite3.Row) -> RequestRecord:
    return RequestRecord(
        request_id=row["id"], ts=row["ts"], model=row["model"], backend=row["backend"], cause=row["cause"],
        stats=CacheStats(row["reused"], row["processed"], row["prefill_ms"], row["stats_source"]),
        client_hint=row["client_hint"], prompt_chars=row["prompt_chars"], prompt_hash=row["prompt_hash"],
        match_id=row["match_id"], divergence_offset=row["divergence_offset"], location=row["location"],
        before_line=row["before_line"], after_line=row["after_line"], ttft_ms=row["ttft_ms"],
        total_ms=row["total_ms"], wasted_ms=row["wasted_ms"],
        rules=tuple(filter(None, row["rules"].split(","))), system_hash=row["system_hash"] or "",
    )  # fmt: skip
