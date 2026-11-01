"""A-OS 图记忆：节点、边、别名、试错流水。

设计原则：
- 图里只放高价值实体（project/module/task/decision/lesson/artifact/concept）
- 原始聊天进 episodes，不进图
- 失败过程进 trials，不进图；成功才折叠成 task 的 artifact/lesson 属性
- 节点数和边数都有硬上限，超限合并/上卷/归档
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

NODE_TYPES = ("project", "module", "task", "decision", "lesson", "artifact", "concept")
EDGE_KINDS = ("contains", "modifies", "depends_on", "similar", "fixed_by", "contradicts")
DEFAULT_IMPORTANCE = {
    "project": 1.0, "module": 0.85, "task": 0.75,
    "decision": 0.8, "lesson": 0.7, "artifact": 0.8, "concept": 0.75,
}


def slug(text: str, *, limit: int = 48) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text, flags=re.U).strip("-")
    return (text or "x")[:limit]


def _norm(vec: Sequence[float] | None) -> np.ndarray | None:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32)
    if arr.size == 0:
        return None
    n = float(np.linalg.norm(arr))
    return arr / n if n > 0 else arr


def _cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or len(a) != len(b):
        return 0.0
    return float(np.dot(a, b))


@dataclass
class Node:
    id: int
    key: str
    type: str
    title: str
    summary: str = ""
    attrs: dict = field(default_factory=dict)
    vec: np.ndarray | None = None
    parent_id: int | None = None
    importance: float = 0.5
    status: str = "active"
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    spectral: list[float] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        # 图节点渲染进上下文时按字符粗略算，够用
        return max(8, len(self.title) + len(self.summary) + 40)


class GraphStore:
    def __init__(self, db_path: str | Path, encoder) -> None:
        self.db_path = str(db_path)
        self.encoder = encoder
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                attrs TEXT NOT NULL DEFAULT '{}',
                vec BLOB,
                parent_id INTEGER,
                importance REAL NOT NULL DEFAULT 0.5,
                status TEXT NOT NULL DEFAULT 'active',
                created REAL NOT NULL,
                updated REAL NOT NULL,
                spectral TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS edges (
                src INTEGER NOT NULL,
                dst INTEGER NOT NULL,
                kind TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY (src, dst, kind)
            );
            CREATE TABLE IF NOT EXISTS aliases (
                alias TEXT NOT NULL,
                node_id INTEGER NOT NULL,
                PRIMARY KEY (alias, node_id)
            );
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_key TEXT NOT NULL,
                step INTEGER NOT NULL DEFAULT 0,
                action TEXT NOT NULL DEFAULT '',
                result TEXT NOT NULL DEFAULT '',
                ok INTEGER NOT NULL DEFAULT 0,
                elapsed REAL NOT NULL DEFAULT 0,
                ts REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
            CREATE INDEX IF NOT EXISTS idx_trials_task ON trials(task_key);
            CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
            """
        )
        self._conn.commit()
        self._nodes: dict[int, Node] = {}
        self._by_key: dict[str, Node] = {}
        self._aliases: dict[str, int] = {}
        self._adj: dict[int, list[tuple[int, str, float]]] = {}
        self._spectral: dict[int, np.ndarray] = {}
        self._vector_version = 0
        self._load()

    # ---------- 持久化 ----------

    def _load(self) -> None:
        for r in self._conn.execute(
            "SELECT id,key,type,title,summary,attrs,vec,parent_id,importance,status,"
            "created,updated,spectral FROM nodes"
        ):
            node = Node(
                id=r[0], key=r[1], type=r[2], title=r[3], summary=r[4],
                attrs=json.loads(r[5] or "{}"),
                vec=np.frombuffer(r[6], dtype=np.float32).copy() if r[6] else None,
                parent_id=r[7], importance=r[8], status=r[9],
                created=r[10], updated=r[11],
                spectral=json.loads(r[12] or "[]"),
            )
            self._nodes[node.id] = node
            self._by_key[node.key] = node
            if node.spectral:
                self._spectral[node.id] = np.asarray(node.spectral, dtype=np.float32)
        for alias, nid in self._conn.execute("SELECT alias,node_id FROM aliases"):
            self._aliases[alias] = nid
        for src, dst, kind, weight in self._conn.execute(
            "SELECT src,dst,kind,weight FROM edges"
        ):
            self._adj.setdefault(src, []).append((dst, kind, weight))
            self._adj.setdefault(dst, []).append((src, kind, weight))

    def _persist(self, node: Node) -> None:
        self._conn.execute(
            "INSERT INTO nodes (id,key,type,title,summary,attrs,vec,parent_id,importance,"
            "status,created,updated,spectral) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET key=excluded.key,type=excluded.type,"
            "title=excluded.title,summary=excluded.summary,attrs=excluded.attrs,"
            "vec=excluded.vec,parent_id=excluded.parent_id,importance=excluded.importance,"
            "status=excluded.status,updated=excluded.updated,spectral=excluded.spectral",
            (
                node.id, node.key, node.type, node.title, node.summary,
                json.dumps(node.attrs, ensure_ascii=False),
                node.vec.astype(np.float32).tobytes() if node.vec is not None else None,
                node.parent_id, node.importance, node.status,
                node.created, node.updated, json.dumps(node.spectral),
            ),
        )
        self._conn.commit()
        self._vector_version += 1

    # ---------- 节点 ----------

    def get(self, key_or_id: str | int | Node) -> Node | None:
        if isinstance(key_or_id, Node):
            return self._nodes.get(key_or_id.id)
        if isinstance(key_or_id, int):
            return self._nodes.get(key_or_id)
        return self._by_key.get(key_or_id)

    def add_node(
        self,
        key: str,
        node_type: str,
        title: str,
        *,
        summary: str = "",
        attrs: dict | None = None,
        parent: str | int | None = None,
        importance: float | None = None,
        vec: Sequence[float] | None = None,
        status: str = "active",
        merge_similar: bool = True,
        merge_threshold: float = 0.93,
    ) -> Node:
        if node_type not in NODE_TYPES:
            raise ValueError(f"未知节点类型：{node_type}")
        existing = self._by_key.get(key)
        if existing is not None:
            self._update(existing, title=title, summary=summary, attrs=attrs,
                         parent=parent, importance=importance, vec=vec, status=status)
            return existing

        v = _norm(vec) if vec is not None else _norm(self.encoder.encode_one(f"{title}\n{summary}"))
        if merge_similar and v is not None:
            near = self._nearest(node_type, v, parent, merge_threshold)
            if near is not None:
                self._update(near, title=title, summary=summary, attrs=attrs,
                             parent=parent, importance=importance, vec=v, status=status)
                self._by_key[key] = near
                self.add_alias(key, near)
                return near

        parent_id = self._resolve_parent(parent)
        now = time.time()
        node = Node(
            id=0, key=key, type=node_type, title=title, summary=summary,
            attrs=dict(attrs or {}), vec=v, parent_id=parent_id,
            importance=DEFAULT_IMPORTANCE.get(node_type, 0.5) if importance is None else importance,
            status=status, created=now, updated=now,
        )
        cur = self._conn.execute(
            "INSERT INTO nodes (key,type,title,summary,attrs,vec,parent_id,importance,"
            "status,created,updated,spectral) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                node.key, node.type, node.title, node.summary,
                json.dumps(node.attrs, ensure_ascii=False),
                node.vec.astype(np.float32).tobytes() if node.vec is not None else None,
                node.parent_id, node.importance, node.status,
                node.created, node.updated, "[]",
            ),
        )
        self._conn.commit()
        node.id = int(cur.lastrowid)
        self._nodes[node.id] = node
        self._by_key[node.key] = node
        self.add_alias(title, node)
        self.add_alias(key, node)
        if node.parent_id is not None:
            self.link(node, node.parent_id, "contains", 1.0)
        self._vector_version += 1
        return node

    def _update(self, node: Node, *, title: str, summary: str, attrs: dict | None,
                parent: str | int | None, importance: float | None,
                vec: Sequence[float] | None, status: str) -> None:
        node.title = title or node.title
        if summary:
            node.summary = _merge_text(node.summary, summary)
        if attrs:
            node.attrs = _merge_attrs(node.attrs, attrs)
        if parent is not None:
            node.parent_id = self._resolve_parent(parent)
        if importance is not None:
            node.importance = max(node.importance, importance)
        if vec is not None:
            node.vec = _norm(vec)
        if status:
            node.status = status
        node.updated = time.time()
        self._persist(node)

    def _resolve_parent(self, parent: str | int | Node | None) -> int | None:
        if parent is None:
            return None
        if isinstance(parent, Node):
            return parent.id if parent.id in self._nodes else None
        if isinstance(parent, int):
            return parent if parent in self._nodes else None
        node = self._by_key.get(parent)
        return node.id if node else None

    def _nearest(self, node_type: str, vec: np.ndarray, parent, threshold: float) -> Node | None:
        best: tuple[float, Node] | None = None
        parent_id = self._resolve_parent(parent)
        for node in self._nodes.values():
            if node.status != "active" or node.type != node_type or node.vec is None:
                continue
            if parent_id is not None and node.parent_id != parent_id:
                continue
            sim = _cos(vec, node.vec)
            if best is None or sim > best[0]:
                best = (sim, node)
        if best and best[0] >= threshold:
            return best[1]
        return None

    def add_alias(self, alias: str, node: Node | int) -> None:
        alias = (alias or "").strip().lower()
        if not alias:
            return
        if isinstance(node, int):
            node = self._nodes.get(node)
        if node is None:
            return
        self._aliases[alias] = node.id
        self._conn.execute("INSERT OR IGNORE INTO aliases (alias,node_id) VALUES (?,?)",
                           (alias, node.id))
        self._conn.commit()

    def children(self, node: Node | int, *, include_archived: bool = False) -> list[Node]:
        nid = node.id if isinstance(node, Node) else node
        return [n for n in self._nodes.values()
                if n.parent_id == nid and (include_archived or n.status == "active")]

    def parent_chain(self, node: Node | int) -> list[Node]:
        nid = node.id if isinstance(node, Node) else node
        chain: list[Node] = []
        seen: set[int] = set()
        cur = self._nodes.get(nid)
        while cur is not None and cur.id not in seen:
            chain.append(cur)
            seen.add(cur.id)
            cur = self._nodes.get(cur.parent_id) if cur.parent_id is not None else None
        return list(reversed(chain))

    # ---------- 边 ----------

    def link(self, src: str | int, dst: str | int, kind: str, weight: float = 1.0) -> None:
        a = self.get(src)
        b = self.get(dst)
        if a is None or b is None or a.id == b.id:
            return
        weight = max(0.0, min(1.0, float(weight)))
        self._conn.execute(
            "INSERT INTO edges (src,dst,kind,weight) VALUES (?,?,?,?)"
            " ON CONFLICT(src,dst,kind) DO UPDATE SET weight=max(weight,excluded.weight)",
            (a.id, b.id, kind, weight),
        )
        self._conn.commit()
        self._add_adj(a.id, b.id, kind, weight)
        self._add_adj(b.id, a.id, kind, weight)

    def _add_adj(self, nid: int, other: int, kind: str, weight: float) -> None:
        edges = self._adj.setdefault(nid, [])
        for i, (oid, okind, oweight) in enumerate(edges):
            if oid == other and okind == kind:
                edges[i] = (oid, okind, max(oweight, weight))
                return
        edges.append((other, kind, weight))

    def neighbors(self, node: Node | int, *, kinds: Iterable[str] | None = None,
                  limit: int = 8) -> list[tuple[Node, str, float]]:
        nid = node.id if isinstance(node, Node) else node
        allow = set(kinds) if kinds else None
        out = []
        for other_id, kind, weight in self._adj.get(nid, []):
            if allow is not None and kind not in allow:
                continue
            other = self._nodes.get(other_id)
            if other is None or other.status != "active":
                continue
            out.append((other, kind, weight))
        out.sort(key=lambda x: (-x[2], x[0].id))
        return out[:limit]

    def link_similar(self, node: Node, *, threshold: float = 0.72, limit: int = 3) -> int:
        """同一父模块内，按向量相似度连少量局部边。"""

        if node.vec is None:
            return 0
        scored: list[tuple[float, Node]] = []
        for other in self._nodes.values():
            if other.id == node.id or other.status != "active" or other.vec is None:
                continue
            if other.parent_id != node.parent_id:
                continue
            sim = _cos(node.vec, other.vec)
            if sim >= threshold:
                scored.append((sim, other))
        scored.sort(key=lambda x: -x[0])
        n = 0
        for sim, other in scored[:limit]:
            self.link(node, other, "similar", sim)
            n += 1
        return n

    def _remove_edge(self, a: int, b: int, kind: str | None = None) -> None:
        if kind:
            self._conn.execute("DELETE FROM edges WHERE src=? AND dst=? AND kind=?", (a, b, kind))
            self._conn.execute("DELETE FROM edges WHERE src=? AND dst=? AND kind=?", (b, a, kind))
        else:
            self._conn.execute("DELETE FROM edges WHERE (src=? AND dst=?) OR (src=? AND dst=?)",
                               (a, b, b, a))
        self._conn.commit()
        for nid, other in ((a, b), (b, a)):
            self._adj[nid] = [e for e in self._adj.get(nid, [])
                              if not (e[0] == other and (kind is None or e[1] == kind))]

    # ---------- 试错流水 ----------

    def add_trials(self, task_key: str, rows: Sequence[dict]) -> int:
        now = time.time()
        data = [
            (task_key, int(r.get("step", 0)), str(r.get("action", ""))[:200],
             str(r.get("result", ""))[:400], 1 if r.get("ok") else 0,
             float(r.get("elapsed", 0.0)), float(r.get("ts", now)))
            for r in rows
        ]
        if data:
            self._conn.executemany(
                "INSERT INTO trials (task_key,step,action,result,ok,elapsed,ts)"
                " VALUES (?,?,?,?,?,?,?)", data,
            )
            self._conn.commit()
        return len(data)

    def trials_count(self, task_key: str | None = None) -> int:
        if task_key:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM trials WHERE task_key=?", (task_key,)).fetchone()[0])
        return int(self._conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])

    # ---------- 检索 ----------

    def alias_lookup(self, query: str, *, limit: int = 6) -> list[tuple[float, Node]]:
        q = (query or "").strip().lower()
        if not q:
            return []
        scored: dict[int, float] = {}
        for alias, nid in self._aliases.items():
            node = self._nodes.get(nid)
            if node is None or node.status != "active":
                continue
            if alias == q:
                score = 3.0
            elif len(alias) >= 2 and alias in q:
                score = 1.0 + min(1.0, len(alias) / max(2, len(q)))
            elif len(q) >= 2 and q in alias:
                score = 0.8
            else:
                continue
            scored[nid] = max(scored.get(nid, 0.0), score)
        out = [(score, self._nodes[nid]) for nid, score in scored.items()]
        out.sort(key=lambda x: (-x[0], -x[1].importance))
        return out[:limit]

    def vector_search(self, vec: Sequence[float], *, k: int = 6,
                      types: Iterable[str] | None = None) -> list[tuple[float, Node]]:
        q = _norm(vec)
        if q is None:
            return []
        allow = set(types) if types else None
        scored = []
        for node in self._nodes.values():
            if node.status != "active" or node.vec is None:
                continue
            if allow is not None and node.type not in allow:
                continue
            scored.append((_cos(q, node.vec), node))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    # ---------- 谱 ----------

    def spectral(self, *, k: int = 8) -> dict[int, np.ndarray]:
        """归一化拉普拉斯的最小非零特征向量，作为节点的低频坐标。

        小特征值对应社区/慢变模式，是"主干"；大特征值是高频噪声，不用于建树。
        """

        nodes = [n for n in self._nodes.values() if n.status == "active"]
        n = len(nodes)
        if n < 3:
            return {}
        index = {node.id: i for i, node in enumerate(nodes)}
        A = np.zeros((n, n), dtype=np.float32)
        for a in nodes:
            for b, _kind, weight in self._adj.get(a.id, []):
                j = index.get(b)
                if j is not None and j > index[a.id]:
                    A[index[a.id], j] = weight
                    A[j, index[a.id]] = weight
        deg = A.sum(axis=1)
        deg[deg == 0] = 1.0
        d = 1.0 / np.sqrt(deg)
        L = np.eye(n, dtype=np.float32) - (A * d[:, None] * d[None, :])
        kk = min(k, n - 1)
        try:
            _vals, vecs = np.linalg.eigh(L)
            coords = vecs[:, 1:kk + 1]
        except np.linalg.LinAlgError:
            return {}
        coords = coords / np.maximum(np.linalg.norm(coords, axis=1, keepdims=True), 1e-9)
        out: dict[int, np.ndarray] = {}
        for node in nodes:
            c = coords[index[node.id]].astype(np.float32)
            out[node.id] = c
            node.spectral = [round(float(x), 5) for x in c]
            self._conn.execute("UPDATE nodes SET spectral=? WHERE id=?",
                               (json.dumps(node.spectral), node.id))
        self._conn.commit()
        self._spectral = out
        return out

    def spectral_affinity(self, a: Node, b: Node) -> float:
        ca = self._spectral.get(a.id)
        cb = self._spectral.get(b.id)
        if ca is None or cb is None:
            return 0.0
        return max(0.0, float(np.dot(ca, cb)))

    # ---------- 上限与合并 ----------

    def enforce_caps(self, *, max_nodes: int = 3000, max_per_module: int = 200,
                     max_degree: int = 12) -> dict:
        archived = 0
        active = [n for n in self._nodes.values() if n.status == "active"]
        if len(active) > max_nodes:
            for node in sorted(active, key=lambda n: (n.importance, n.updated)):
                if node.type in ("project", "module"):
                    continue
                node.status = "archived"
                self._persist(node)
                archived += 1
                if len([n for n in self._nodes.values() if n.status == "active"]) <= max_nodes:
                    break

        for module in [n for n in self._nodes.values()
                       if n.type == "module" and n.status == "active"]:
            kids = [n for n in self.children(module) if n.type == "task"]
            if len(kids) > max_per_module:
                for node in sorted(kids, key=lambda n: (n.importance, n.updated))[: len(kids) - max_per_module]:
                    node.status = "archived"
                    self._persist(node)
                    archived += 1

        trimmed = 0
        for nid in list(self._adj):
            edges = self._adj.get(nid, [])
            if len(edges) <= max_degree:
                continue
            keep = sorted(edges, key=lambda e: -e[2])[:max_degree]
            keep_keys = {(e[0], e[1]) for e in keep}
            for other, kind, _w in list(edges):
                if (other, kind) not in keep_keys:
                    self._remove_edge(nid, other, kind)
                    trimmed += 1
        return {"archived": archived, "trimmed_edges": trimmed}

    # ---------- 观察 ----------

    def stats(self) -> dict:
        active = [n for n in self._nodes.values() if n.status == "active"]
        by_type: dict[str, int] = {}
        for node in active:
            by_type[node.type] = by_type.get(node.type, 0) + 1
        degrees = [len(self._adj.get(n.id, [])) for n in active]
        return {
            "active": len(active),
            "total": len(self._nodes),
            "archived": len(self._nodes) - len(active),
            "edges": int(self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]),
            "trials": self.trials_count(),
            "by_type": by_type,
            "max_degree": max(degrees) if degrees else 0,
            "avg_degree": round(sum(degrees) / len(degrees), 2) if degrees else 0,
            "aliases": len(self._aliases),
        }

    def render(self, node: Node, *, include_children: bool = True) -> str:
        lines = [f"[{node.type}] {node.title}"]
        if node.summary:
            lines.append(node.summary[:500])
        lessons = node.attrs.get("lessons") or []
        if lessons:
            lines.append("经验：" + "；".join(str(x)[:80] for x in lessons[:3]))
        artifact = node.attrs.get("artifact")
        if artifact:
            lines.append("产物：" + str(artifact)[:120])
        if include_children:
            kids = self.children(node)
            if kids:
                lines.append("子节点：" + "、".join(k.title for k in kids[:8]))
        return "\n".join(lines)

    def close(self) -> None:
        self._conn.close()


def _merge_text(old: str, new: str, *, limit: int = 320) -> str:
    old = (old or "").strip()
    new = (new or "").strip()
    if not new:
        return old
    if not old:
        return new[:limit]
    if new in old:
        return old
    merged = f"{old}；{new}"
    return merged[-limit:]


def _merge_attrs(old: dict, new: dict) -> dict:
    out = dict(old or {})
    for key, value in (new or {}).items():
        if key == "lessons":
            items = list(out.get(key) or [])
            for item in value or []:
                item = str(item).strip()
                if item and item not in items:
                    items.append(item)
            out[key] = items[:3]
        elif key == "attempts":
            out[key] = max(int(out.get(key) or 0), int(value or 0))
        elif key == "status":
            if value == "done" or out.get(key) != "done":
                out[key] = value
        elif key not in out or not out[key]:
            out[key] = value
    return out
