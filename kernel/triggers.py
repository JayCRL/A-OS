"""Trigger 队列：Agent 层的“中断控制器”。

外部事件先落库、去重、排序，再由内核决定唤醒/派生哪个 worker。
不是 MCP 工具，也不让 Router 直接阻塞在事件源上。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

STATUSES = ("queued", "running", "done", "failed", "ignored")


@dataclass
class Trigger:
    id: int
    source: str
    kind: str
    payload: dict
    thread_id: str
    priority: int
    status: str
    dedupe_key: str
    result: str
    created_at: float
    updated_at: float


class TriggerStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS triggers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT NOT NULL,
                kind        TEXT NOT NULL,
                payload     TEXT NOT NULL DEFAULT '{}',
                thread_id   TEXT NOT NULL DEFAULT '',
                priority    INTEGER NOT NULL DEFAULT 0,
                status      TEXT NOT NULL DEFAULT 'queued',
                dedupe_key  TEXT NOT NULL DEFAULT '',
                result      TEXT NOT NULL DEFAULT '',
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_trigger_dedupe ON triggers(dedupe_key)"
            " WHERE dedupe_key != ''"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trigger_status ON triggers(status, priority)"
        )
        self._conn.commit()

    def _row(self, row) -> Trigger:
        return Trigger(
            id=row[0], source=row[1], kind=row[2], payload=json.loads(row[3] or "{}"),
            thread_id=row[4], priority=row[5], status=row[6], dedupe_key=row[7],
            result=row[8], created_at=row[9], updated_at=row[10],
        )

    def push(self, *, source: str, kind: str, payload: dict | None = None,
             thread_id: str = "", priority: int = 0,
             dedupe_key: str = "") -> tuple[Trigger, bool]:
        payload = dict(payload or {})
        if not dedupe_key:
            raw = json.dumps([source, kind, thread_id, payload], ensure_ascii=False, sort_keys=True)
            dedupe_key = hashlib.sha1(raw.encode()).hexdigest()
        now = time.time()
        try:
            cur = self._conn.execute(
                "INSERT INTO triggers (source,kind,payload,thread_id,priority,status,"
                "dedupe_key,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (source, kind, json.dumps(payload, ensure_ascii=False), thread_id,
                 int(priority), "queued", dedupe_key, now, now),
            )
            self._conn.commit()
            return self.get(int(cur.lastrowid)), True
        except sqlite3.IntegrityError:
            row = self._conn.execute(
                "SELECT id,source,kind,payload,thread_id,priority,status,dedupe_key,"
                "result,created_at,updated_at FROM triggers WHERE dedupe_key=?",
                (dedupe_key,),
            ).fetchone()
            return self._row(row), False

    def get(self, tid: int) -> Trigger | None:
        row = self._conn.execute(
            "SELECT id,source,kind,payload,thread_id,priority,status,dedupe_key,"
            "result,created_at,updated_at FROM triggers WHERE id=?", (tid,)
        ).fetchone()
        return self._row(row) if row else None

    def claim(self, tid: int, pid: int) -> bool:
        now = time.time()
        cur = self._conn.execute(
            "UPDATE triggers SET status='running', updated_at=? WHERE id=? AND status='queued'",
            (now, tid),
        )
        self._conn.commit()
        return cur.rowcount == 1

    def finish(self, tid: int, *, status: str, result: str = "") -> None:
        if status not in STATUSES:
            status = "failed"
        self._conn.execute(
            "UPDATE triggers SET status=?, result=?, updated_at=? WHERE id=?",
            (status, result[:4000], time.time(), tid),
        )
        self._conn.commit()

    def list(self, *, status: str | None = None, limit: int = 50) -> list[Trigger]:
        sql = ("SELECT id,source,kind,payload,thread_id,priority,status,dedupe_key,"
               "result,created_at,updated_at FROM triggers")
        args: list = []
        if status:
            sql += " WHERE status=?"
            args.append(status)
        sql += " ORDER BY priority DESC, id ASC LIMIT ?"
        args.append(max(1, min(200, limit)))
        return [self._row(r) for r in self._conn.execute(sql, args).fetchall()]

    def stats(self) -> dict:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) FROM triggers GROUP BY status"
        ).fetchall()
        return {status: int(n) for status, n in rows}

    def close(self) -> None:
        self._conn.close()
