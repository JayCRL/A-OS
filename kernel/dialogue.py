"""对话史。

只负责把说过的话按时间存下来、按窗口取出来。所有轮次永久保留；
进不进模型上下文由上下文构建器按预算决定。
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Turn:
    id: int
    role: str
    text: str
    ts: float


class Dialogue:
    def __init__(self, db_path: str | Path, *, session: str = "main") -> None:
        self.session = session
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                session TEXT NOT NULL DEFAULT 'main',
                role    TEXT NOT NULL,
                text    TEXT NOT NULL,
                ts      REAL NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session, id)")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)"
        )
        self._conn.commit()

    def add(self, role: str, text: str) -> Turn:
        ts = time.time()
        cur = self._conn.execute(
            "INSERT INTO turns (session, role, text, ts) VALUES (?,?,?,?)",
            (self.session, role, text, ts),
        )
        self._conn.commit()
        return Turn(int(cur.lastrowid), role, text, ts)

    def recent(self, *, limit: int = 14, before_id: int | None = None) -> list[Turn]:
        """按时间正序返回最近 limit 条。"""

        if before_id is None:
            rows = self._conn.execute(
                "SELECT id, role, text, ts FROM turns WHERE session=? ORDER BY id DESC LIMIT ?",
                (self.session, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, role, text, ts FROM turns WHERE session=? AND id<? ORDER BY id DESC LIMIT ?",
                (self.session, before_id, limit),
            ).fetchall()
        return [Turn(*r) for r in reversed(rows)]

    def history(self, *, limit: int = 80) -> list[Turn]:
        rows = self._conn.execute(
            "SELECT id, role, text, ts FROM turns WHERE session=? ORDER BY id DESC LIMIT ?",
            (self.session, limit),
        ).fetchall()
        return [Turn(*r) for r in reversed(rows)]

    def all(self) -> list[Turn]:
        rows = self._conn.execute(
            "SELECT id, role, text, ts FROM turns WHERE session=? ORDER BY id", (self.session,)
        ).fetchall()
        return [Turn(*r) for r in rows]

    def count(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) FROM turns WHERE session=?", (self.session,)).fetchone()[0]
        )

    def oldest_unsummarized(self, *, keep_recent: int = 16, limit: int = 30) -> list[Turn]:
        upto = int(self.get_meta("summary_upto", "0") or 0)
        total = self.count()
        if total <= keep_recent + 4:
            return []
        rows = self._conn.execute(
            "SELECT id, role, text, ts FROM turns WHERE session=? AND id>? ORDER BY id LIMIT ?",
            (self.session, upto, limit),
        ).fetchall()
        rows = [r for r in rows if r[0] <= (self._max_id() - keep_recent)]
        return [Turn(*r) for r in rows]

    def _max_id(self) -> int:
        row = self._conn.execute("SELECT MAX(id) FROM turns WHERE session=?", (self.session,)).fetchone()
        return int(row[0] or 0)

    def _meta_key(self, key: str) -> str:
        return f"{self.session}:{key}"

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT v FROM meta WHERE k=?", (self._meta_key(key),)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (k, v) VALUES (?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (self._meta_key(key), value),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
