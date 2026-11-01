"""长期记忆。

设计上只做两件事：把值得记的写进去，把相关的捞出来。
不换页、不遗忘、不设容量上限；上下文放不下时只裁剪"这次要带进去的"，
库里的记忆一条都不删。
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from kernel.tokens import estimate_tokens

KINDS = ("identity", "preference", "fact", "plan", "episode", "summary")

# 不同类型的分值上限：身份和偏好几乎不会过期，闲聊片段会。
BASE_IMPORTANCE = {
    "identity": 0.95,
    "preference": 0.9,
    "plan": 0.8,
    "fact": 0.7,
    "summary": 0.65,
    "episode": 0.45,
}

def _normalize(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def _grams(text: str) -> set[str]:
    """中文字符二元组 + 英文词，用来补向量检索对专名/缩写的不敏感。"""

    low = (text or "").lower()
    chars = re.sub(r"\s+", "", low)
    grams = {chars[i: i + 2] for i in range(max(0, len(chars) - 1))}
    grams.update(re.findall(r"[a-z0-9_]{2,}", low))
    return grams


def _lexical(a: str, b: str) -> float:
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    return inter / math.sqrt(len(ga) * len(gb))


@dataclass
class Memory:
    id: int
    text: str
    kind: str
    ts: float
    importance: float
    refs: int = 0
    source: str = ""
    vec: np.ndarray | None = None

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


class MemoryStore:
    """SQLite 落盘 + 内存向量矩阵。万级条目内暴力余弦足够快。"""

    def __init__(self, db_path: str | Path, encoder, *, max_text: int = 400) -> None:
        self.db_path = str(db_path)
        self.encoder = encoder
        self.max_text = max_text
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                text        TEXT NOT NULL,
                kind        TEXT NOT NULL DEFAULT 'fact',
                ts          REAL NOT NULL,
                importance  REAL NOT NULL DEFAULT 0.5,
                refs        INTEGER NOT NULL DEFAULT 0,
                source      TEXT NOT NULL DEFAULT '',
                vec         BLOB
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_kind ON memories(kind)")
        self._conn.commit()
        self._cache: list[Memory] = []
        self._index: dict[int, Memory] = {}
        self._mat_full: np.ndarray | None = None
        self._mat_cap = 0
        self._mat_n = 0
        self._mat_dim = 0
        self._mat_ids: list[int] = []
        self._load()

    # ---------- 读写 ----------

    def _load(self) -> None:
        rows = self._conn.execute(
            "SELECT id, text, kind, ts, importance, refs, source, vec FROM memories ORDER BY id"
        ).fetchall()
        self._cache = [
            Memory(
                id=r[0], text=r[1], kind=r[2], ts=r[3], importance=r[4], refs=r[5],
                source=r[6], vec=np.frombuffer(r[7], dtype=np.float32).copy() if r[7] else None,
            )
            for r in rows
        ]
        self._index = {m.id: m for m in self._cache}
        self._reset_matrix()

    def _reset_matrix(self) -> None:
        self._mat_full = None
        self._mat_n = 0
        self._mat_ids = []

    def _matrix(self) -> np.ndarray:
        if self._mat_full is None:
            vecs: list[np.ndarray] = []
            ids: list[int] = []
            for m in self._cache:
                if m.vec is None or not len(m.vec):
                    continue
                if vecs and len(m.vec) != len(vecs[0]):
                    continue
                vecs.append(m.vec)
                ids.append(m.id)
            self._mat_dim = len(vecs[0]) if vecs else int(getattr(self.encoder, "dim", 1))
            self._mat_cap = max(64, len(vecs) * 2)
            self._mat_full = np.zeros((self._mat_cap, self._mat_dim), dtype=np.float32)
            for i, vec in enumerate(vecs):
                self._mat_full[i] = vec
            self._mat_n = len(vecs)
            self._mat_ids = ids
        return self._mat_full[: self._mat_n]

    def _append_vec(self, mem: Memory) -> None:
        if mem.vec is None or not len(mem.vec):
            return
        self._matrix()
        if len(mem.vec) != self._mat_dim:
            self._reset_matrix()
            return
        if self._mat_n >= self._mat_cap:
            self._mat_cap *= 2
            grown = np.zeros((self._mat_cap, self._mat_dim), dtype=np.float32)
            grown[: self._mat_n] = self._mat_full[: self._mat_n]
            self._mat_full = grown
        self._mat_full[self._mat_n] = mem.vec
        self._mat_n += 1
        self._mat_ids.append(mem.id)

    def add(
        self,
        text: str,
        *,
        kind: str = "fact",
        importance: float | None = None,
        source: str = "",
        dedupe: bool = True,
    ) -> Memory | None:
        text = (text or "").strip()
        if not text or kind not in KINDS:
            return None
        if len(text) > self.max_text:
            text = text[: self.max_text].rstrip()
        imp = BASE_IMPORTANCE.get(kind, 0.5) if importance is None else max(0.0, min(1.0, importance))

        vec = _normalize(self.encoder.encode_one(text))
        if dedupe:
            near = self._nearest(text, vec, kind)
            if near is not None and near[1] >= (0.965 if kind == "episode" else 0.93):
                _, score = near
                old = self._by_id(near[0])
                if old is not None:
                    old.ts = time.time()
                    old.refs += 1
                    old.importance = max(old.importance, imp)
                    if len(text) > len(old.text) * 1.25:
                        old.text = text
                        old.vec = vec
                        self._reset_matrix()
                    self._conn.execute(
                        "UPDATE memories SET text=?, ts=?, importance=?, refs=?, vec=? WHERE id=?",
                        (old.text, old.ts, old.importance, old.refs,
                         old.vec.astype(np.float32).tobytes() if old.vec is not None else None, old.id),
                    )
                    self._conn.commit()
                    return old

        ts = time.time()
        cur = self._conn.execute(
            "INSERT INTO memories (text, kind, ts, importance, refs, source, vec) VALUES (?,?,?,?,?,?,?)",
            (text, kind, ts, imp, 0, source, vec.astype(np.float32).tobytes()),
        )
        self._conn.commit()
        mem = Memory(int(cur.lastrowid), text, kind, ts, imp, 0, source, vec)
        self._cache.append(mem)
        self._index[mem.id] = mem
        self._append_vec(mem)
        return mem

    def add_many(self, items: Iterable[dict], *, source: str = "") -> int:
        n = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            kind = str(item.get("kind", "fact")).strip()
            if kind not in KINDS:
                kind = "fact"
            try:
                imp = float(item.get("importance", BASE_IMPORTANCE[kind]))
            except (TypeError, ValueError):
                imp = BASE_IMPORTANCE[kind]
            if text and self.add(text, kind=kind, importance=imp, source=source):
                n += 1
        return n

    def _by_id(self, mid: int) -> Memory | None:
        return self._index.get(mid)

    def _nearest(self, text: str, vec: np.ndarray, kind: str) -> tuple[int, float] | None:
        if not self._cache:
            return None
        mat = self._matrix()
        if mat.size == 0 or mat.shape[1] != vec.shape[0]:
            return None
        ids = getattr(self, "_mat_ids", [])
        same = [i for i, mid in enumerate(ids)
                if self._by_id(mid) is not None and self._by_id(mid).kind == kind]
        if not same:
            return None
        sims = mat[same] @ vec
        order = np.argsort(-sims)[:24]
        best: tuple[int, float] | None = None
        for j in order:
            mem = self._by_id(ids[same[int(j)]])
            if mem is None:
                continue
            score = float(sims[int(j)])
            if kind in ("identity", "preference", "fact", "plan"):
                score = max(score, 0.62 * score + 0.38 * _lexical(text, mem.text))
            if best is None or score > best[1]:
                best = (mem.id, score)
        return best

    # ---------- 召回 ----------

    def recall(
        self,
        query: str,
        *,
        k: int = 8,
        max_tokens: int = 1400,
        kinds: Sequence[str] | None = None,
        exclude_ids: set[int] | None = None,
        min_score: float = 0.0,
    ) -> list[tuple[float, Memory]]:
        if not self._cache or not query.strip():
            return []
        q = _normalize(self.encoder.encode_one(query))
        mat = self._matrix()
        if mat.size == 0 or mat.shape[1] != q.shape[0]:
            return []
        ids = getattr(self, "_mat_ids", [])
        sims = mat @ q
        now = time.time()
        exclude_ids = exclude_ids or set()

        # 高频词不加分，专名/编号这类稀有片段加权：向量检索对数字不敏感，靠这个补
        qgrams = _grams(query)
        df = {g: 0 for g in qgrams}
        for mem in self._cache:
            low = mem.text.lower()
            for g in qgrams:
                if g in low:
                    df[g] += 1
        rare_cut = max(2, int(len(self._cache) * 0.02))
        rare = [g for g in qgrams if 0 < df.get(g, 0) <= rare_cut]

        cands: list[tuple[float, float, Memory]] = []
        for idx, mid in enumerate(ids):
            mem = self._by_id(mid)
            if mem is None or mem.kind not in (kinds or ("episode", "summary", "fact", "plan", "preference", "identity")):
                continue
            if mem.id in exclude_ids:
                continue
            cos = float(sims[idx])
            lex = _lexical(query, mem.text)
            if cos < 0.42 and lex < 0.18:
                continue
            age_days = max(0.0, (now - mem.ts) / 86400.0)
            recency = 1.0 / (1.0 + math.log1p(age_days / 7.0))
            bonus = 0.0
            if rare:
                low = mem.text.lower()
                bonus = 0.30 * sum(1 for g in rare if g in low) / len(rare)
            score = 0.58 * cos + 0.14 * lex + 0.08 * recency + 0.20 * mem.importance + bonus
            if score >= min_score:
                cands.append((score, cos, mem))
        if not cands:
            return []

        cands.sort(key=lambda x: x[0], reverse=True)
        cands = cands[: max(k * 5, 24)]

        # MMR：挑相关但不重复的几条，避免一口气带进 5 条同义记忆
        picked: list[tuple[float, Memory]] = []
        used_tokens = 0
        chosen_vecs: list[np.ndarray] = []
        for score, _cos, mem in cands:
            if mem.tokens + used_tokens > max_tokens:
                continue
            if mem.vec is not None and chosen_vecs:
                overlap = max(float(mem.vec @ v) for v in chosen_vecs)
                mmr = 0.78 * score - 0.22 * overlap
            else:
                mmr = score
            picked.append((mmr, mem))
            used_tokens += mem.tokens
            if mem.vec is not None:
                chosen_vecs.append(mem.vec)
            if len(picked) >= k:
                break

        picked.sort(key=lambda x: x[0], reverse=True)
        for _, mem in picked:
            mem.refs += 1
        if picked:
            self._conn.executemany(
                "UPDATE memories SET refs=? WHERE id=?", [(m.refs, m.id) for _, m in picked]
            )
            self._conn.commit()
        return picked

    def profile(self, *, limit: int = 8, max_tokens: int = 600) -> list[Memory]:
        """稳定信息：身份、偏好。每次对话都带上，不靠检索碰运气。"""

        rows = [m for m in self._cache if m.kind in ("identity", "preference")]
        rows.sort(key=lambda m: (m.importance, m.ts), reverse=True)
        out: list[Memory] = []
        used = 0
        for m in rows:
            if m.tokens + used > max_tokens or len(out) >= limit:
                break
            out.append(m)
            used += m.tokens
        return out

    def recent(self, *, limit: int = 10, kinds: Sequence[str] = ("fact", "plan", "preference", "identity")) -> list[Memory]:
        rows = [m for m in self._cache if m.kind in kinds]
        rows.sort(key=lambda m: m.ts, reverse=True)
        return rows[:limit]

    def get(self, mid: int) -> Memory | None:
        return self._by_id(mid)

    def all(self) -> list[Memory]:
        return list(self._cache)

    # ---------- 维护 ----------

    def stats(self) -> dict:
        by_kind: dict[str, int] = {}
        for m in self._cache:
            by_kind[m.kind] = by_kind.get(m.kind, 0) + 1
        return {"total": len(self._cache), "by_kind": by_kind}

    def close(self) -> None:
        self._conn.close()
