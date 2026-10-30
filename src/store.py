"""SQLite edit log. Saves every glue change so edits survive a restart."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edge_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    edit_id INTEGER NOT NULL,
    ts      REAL    NOT NULL,
    user    TEXT    NOT NULL,
    op      TEXT    NOT NULL CHECK (op IN ('add', 'remove')),
    brick_a INTEGER NOT NULL,
    brick_b INTEGER NOT NULL,
    undone  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_edit ON edge_events (edit_id);
"""


def _connect(path: Path) -> sqlite3.Connection:
    # One connection shared between threads; EditService serialises access with a lock.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def _normalise(fingerprint: dict) -> str:
    return json.dumps(json.loads(json.dumps(fingerprint)), sort_keys=True)


class EditStore:
    def __init__(self, conn: sqlite3.Connection, path: Path):
        self._conn = conn
        self.path = path

    @classmethod
    def open(cls, path: Path, fingerprint: dict) -> tuple["EditStore", str]:
        """Open the log. Returns (store, status).

        status is "new" (nothing saved yet), "match" (saved edits fit this data) or
        "reset" (saved edits were for different data, so the old file was moved aside).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        wanted = _normalise(fingerprint)
        status = "new"
        conn = _connect(path)
        row = conn.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
        has_events = conn.execute("SELECT COUNT(*) FROM edge_events").fetchone()[0] > 0
        if row is not None and row[0] != wanted:
            conn.close()
            backup = path.with_name(f"{path.name}.mismatch-{int(time.time())}")
            path.rename(backup)
            for suffix in ("-wal", "-shm"):
                leftover = Path(str(path) + suffix)
                if leftover.exists():
                    leftover.unlink()
            conn = _connect(path)
            status = "reset"
        elif row is not None and has_events:
            status = "match"
        with conn:
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('fingerprint', ?)", (wanted,))
        return cls(conn, path), status

    def append_edit(self, user: str, events: list[tuple[str, int, int]]) -> int:
        """Save one edit (one or more glue changes) as a single all-or-nothing step."""
        if not events:
            raise ValueError("An edit needs at least one glue change")
        with self._conn:
            edit_id = self._conn.execute("SELECT COALESCE(MAX(edit_id), 0) + 1 FROM edge_events").fetchone()[0]
            now = time.time()
            self._conn.executemany(
                "INSERT INTO edge_events (edit_id, ts, user, op, brick_a, brick_b) VALUES (?, ?, ?, ?, ?, ?)",
                [(edit_id, now, user, op, int(a), int(b)) for op, a, b in events],
            )
        return int(edit_id)

    def live_events(self) -> list[tuple[int, str, int, int]]:
        """All glue changes that are not undone, in the order they happened: (edit_id, op, a, b)."""
        rows = self._conn.execute(
            "SELECT edit_id, op, brick_a, brick_b FROM edge_events WHERE undone=0 ORDER BY id"
        ).fetchall()
        return [(int(e), str(op), int(a), int(b)) for e, op, a, b in rows]

    def undo_last(self) -> tuple[int, str] | None:
        """Mark the newest edit as undone. Returns (edit_id, user) or None if nothing to undo."""
        with self._conn:
            row = self._conn.execute(
                "SELECT edit_id, user FROM edge_events WHERE undone=0 ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            self._conn.execute("UPDATE edge_events SET undone=1 WHERE edit_id=?", (row[0],))
        return int(row[0]), str(row[1])

    def undo_all(self) -> int:
        """Mark every edit as undone (the log is kept for the record). Returns how many edits that was."""
        with self._conn:
            count = self.edit_count()
            self._conn.execute("UPDATE edge_events SET undone=1 WHERE undone=0")
        return count

    def edit_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(DISTINCT edit_id) FROM edge_events WHERE undone=0").fetchone()[0])

    def close(self) -> None:
        self._conn.close()
