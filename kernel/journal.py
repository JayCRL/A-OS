"""任务日志与崩溃恢复。

任务状态、断点、租约都落 SQLite。进程被杀后，靠租约过期判断哪些任务可以接手。
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskRecord:
    task_id: str
    goal: str
    node_id: int | None
    status: str
    round: int
    steps: int
    lease_until: float
    attempts: int
    created_at: float


class Journal:
    def __init__(self, db_path: str | Path, *, lease_seconds: float = 60.0) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id     TEXT PRIMARY KEY,
                goal        TEXT NOT NULL,
                node_id     INTEGER,
                status      TEXT NOT NULL,
                round       INTEGER NOT NULL DEFAULT 0,
                steps       INTEGER NOT NULL DEFAULT 0,
                lease_until REAL NOT NULL DEFAULT 0,
                attempts    INTEGER NOT NULL DEFAULT 0,
                created_at  REAL NOT NULL
            )
            """
        )
        self._conn.commit()
        self.lease_seconds = lease_seconds

    # ---------- 生命周期 ----------

    def register(self, goal: str, node_id: int | None) -> str:
        task_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            "INSERT INTO tasks (task_id, goal, node_id, status, round, steps, lease_until, attempts, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (task_id, goal, node_id, "ready", 0, 0, 0.0, 0, time.time()),
        )
        self._conn.commit()
        return task_id

    def claim(self, task_id: str) -> bool:
        """抢任务：只有租约过期才能抢到，防止两个进程同时跑。"""

        now = time.time()
        cur = self._conn.execute(
            "UPDATE tasks SET status='running', lease_until=?, attempts=attempts+1"
            " WHERE task_id=? AND (status!='running' OR lease_until<?)",
            (now + self.lease_seconds, task_id, now),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def renew(self, task_id: str) -> None:
        self._conn.execute(
            "UPDATE tasks SET lease_until=? WHERE task_id=?",
            (time.time() + self.lease_seconds, task_id),
        )
        self._conn.commit()

    def checkpoint(self, task_id: str, *, node_id: int | None, round_no: int, steps: int) -> None:
        self._conn.execute(
            "UPDATE tasks SET node_id=?, round=?, steps=?, lease_until=? WHERE task_id=?",
            (node_id, round_no, steps, time.time() + self.lease_seconds, task_id),
        )
        self._conn.commit()

    def finish(self, task_id: str, status: str = "done") -> None:
        self._conn.execute(
            "UPDATE tasks SET status=?, lease_until=0 WHERE task_id=?",
            (status, task_id),
        )
        self._conn.commit()

    # ---------- 恢复 ----------

    def recover(self) -> list[TaskRecord]:
        """启动时调用：把可接手的任务挑出来。"""

        now = time.time()
        self._conn.execute(
            "UPDATE tasks SET status='ready' WHERE status='running' AND lease_until<?", (now,)
        )
        self._conn.execute(
            "UPDATE tasks SET status='failed' WHERE status='ready' AND attempts>=3"
        )
        self._conn.commit()

        rows = self._conn.execute(
            "SELECT task_id, goal, node_id, status, round, steps, lease_until, attempts, created_at"
            " FROM tasks WHERE status='ready' ORDER BY created_at"
        ).fetchall()
        return [TaskRecord(*r) for r in rows]

    def get(self, task_id: str) -> TaskRecord | None:
        r = self._conn.execute(
            "SELECT task_id, goal, node_id, status, round, steps, lease_until, attempts, created_at"
            " FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        return TaskRecord(*r) if r else None

    def stats(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status").fetchall()
        return {k: v for k, v in rows}

    def close(self) -> None:
        self._conn.close()
