"""A-OS 状态树：CoW 分支 + 后验剪枝。

不存在 messages 数组，只有节点；每次组装上下文都从活跃路径重新拼。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class NodeStatus(str, Enum):
    ACTIVE = "active"      # 在主干活跃路径上
    ARCHIVED = "archived"  # 被剪掉的旁支
    SUCCESS = "success"    # 人类确认成功
    FAILED = "failed"      # 明确失败


@dataclass
class Node:
    node_id: int
    parent_id: int | None
    content: str
    status: NodeStatus = NodeStatus.ACTIVE
    kind: str = "step"
    created_at: float = field(default_factory=time.time)
    meta: dict | None = None
    pid: int = 0


class StateDAG:
    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id  INTEGER,
                content    TEXT NOT NULL,
                status     TEXT NOT NULL,
                kind       TEXT NOT NULL DEFAULT 'step',
                created_at REAL NOT NULL,
                meta       TEXT,
                pid        INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(nodes)")}
        if "pid" not in cols:
            self._conn.execute("ALTER TABLE nodes ADD COLUMN pid INTEGER NOT NULL DEFAULT 0")
        self._conn.commit()
        self._nodes: dict[int, Node] = {}
        self._load()

    def _load(self) -> None:
        for r in self._conn.execute(
            "SELECT node_id, parent_id, content, status, kind, created_at, meta, pid FROM nodes"
        ):
            self._nodes[r[0]] = Node(
                node_id=r[0], parent_id=r[1], content=r[2],
                status=NodeStatus(r[3]), kind=r[4], created_at=r[5],
                meta=(json.loads(r[6]) if r[6] else None), pid=r[7],
            )

    # ---------- 基本操作 ----------

    def add(self, content: str, *, parent_id: int | None = None, kind: str = "step",
            meta: dict | None = None, pid: int = 0) -> Node:
        cur = self._conn.execute(
            "INSERT INTO nodes (parent_id, content, status, kind, created_at, meta, pid)"
            " VALUES (?,?,?,?,?,?,?)",
            (parent_id, content, NodeStatus.ACTIVE.value, kind, time.time(),
             json.dumps(meta, ensure_ascii=False) if meta else None, pid),
        )
        self._conn.commit()
        node = Node(int(cur.lastrowid), parent_id, content, NodeStatus.ACTIVE, kind, time.time(), meta, pid=pid)
        self._nodes[node.node_id] = node
        return node

    def get(self, node_id: int) -> Node | None:
        return self._nodes.get(node_id)

    def children(self, node_id: int) -> list[Node]:
        return [n for n in self._nodes.values() if n.parent_id == node_id]

    def head(self, pid: int | None = None) -> Node | None:
        """当前活跃路径的末端：最深的 ACTIVE / SUCCESS 节点。"""
        live = [n for n in self._nodes.values()
                if n.status in (NodeStatus.ACTIVE, NodeStatus.SUCCESS)
                and (pid is None or n.pid == pid)]
        if not live:
            return None
        return max(live, key=lambda n: (self._depth(n), n.created_at))

    def _depth(self, node: Node) -> int:
        d = 0
        cur = node
        while cur.parent_id is not None and cur.parent_id in self._nodes:
            cur = self._nodes[cur.parent_id]
            d += 1
        return d

    # ---------- CoW ----------

    def cow(self, branch_from: int, content: str, *, kind: str = "human_interrupt",
            pid: int = 0) -> Node:
        """人类干预：从指定节点派生新分支，旧分支整条归档。"""

        base = self._nodes[branch_from]
        self._archive_subtree(base.node_id)
        base.status = NodeStatus.ARCHIVED
        self._settle(base)

        node = self.add(content, parent_id=base.parent_id, kind=kind, pid=pid)
        return node

    def _archive_subtree(self, root_id: int) -> None:
        stack = [root_id]
        while stack:
            nid = stack.pop()
            node = self._nodes[nid]
            if node.status is NodeStatus.ACTIVE:
                node.status = NodeStatus.ARCHIVED
                self._settle(node)
            stack.extend(c.node_id for c in self.children(nid))

    # ---------- 剪枝 ----------

    def prune(self, success_id: int, pid: int | None = None) -> Node:
        """成功后验剪枝：成功路径保留，其余 ACTIVE 旁支归档并压成经验。"""

        winner = self._nodes[success_id]
        winner.status = NodeStatus.SUCCESS
        self._settle(winner)

        path = {n.node_id for n in self.path_to_root(success_id)}
        lessons: list[str] = []

        for node in list(self._nodes.values()):
            if node.node_id in path:
                continue
            if pid is not None and node.pid != pid:
                continue
            if node.status is not NodeStatus.ACTIVE:
                continue
            lessons.append(node.content[:160])
            node.status = NodeStatus.FAILED
            self._settle(node)

        if not lessons:
            return winner

        digest = "经验：" + " / ".join(lessons)
        return self.add(digest, parent_id=winner.parent_id, kind="lesson", pid=winner.pid)

    # ---------- 组装 ----------

    def path_to_root(self, node_id: int) -> list[Node]:
        out: list[Node] = []
        cur: Node | None = self._nodes.get(node_id)
        while cur is not None:
            out.append(cur)
            cur = self._nodes.get(cur.parent_id) if cur.parent_id is not None else None
        return list(reversed(out))

    def active_path(self, pid: int | None = None) -> list[Node]:
        head = self.head(pid)
        return self.path_to_root(head.node_id) if head else []

    def render(self, *, limit: int = 12, max_chars: int = 4000,
               pid: int | None = None) -> str:
        lines: list[str] = []
        for node in self.active_path(pid)[-limit:]:
            mark = "*" if node.status is NodeStatus.SUCCESS else "-"
            lines.append(f"{mark} [{node.kind}] {node.content[:max_chars]}")
        return "\n".join(lines)

    def stats(self) -> dict[str, int]:
        c = {s.value: 0 for s in NodeStatus}
        for n in self._nodes.values():
            c[n.status.value] += 1
        c["total"] = len(self._nodes)
        return c

    def update_content(self, node_id: int, content: str) -> None:
        node = self._nodes.get(node_id)
        if not node:
            return
        node.content = content
        self._conn.execute("UPDATE nodes SET content=? WHERE node_id=?", (content, node_id))
        self._conn.commit()

    def _settle(self, node: Node) -> None:
        self._conn.execute(
            "UPDATE nodes SET status=? WHERE node_id=?", (node.status.value, node.node_id)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
