"""A-OS 虚拟上下文管理器（MMU）。

页表持久化在 SQLite，内存里只留索引；页内容是语义块，不是定长行。
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


class Segment(str, Enum):
    SYSTEM = "system"      # 宪法、工具定义：最高特权，永不换出
    HUMAN = "human"        # 人类指令：本轮必带
    DAG_STATE = "dag"      # 任务状态：常驻
    DATA = "data"          # 代码/文档：可淘汰


class PageState(str, Enum):
    ACTIVE = "active"        # 常驻热层
    SWAPPED = "swapped_out"  # 已换出，正文在冷层


@dataclass
class Page:
    page_id: int
    content: str
    state: PageState = PageState.ACTIVE
    pinned: bool = False
    importance: float = 0.0
    refs: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    vec: list[float] | None = None
    tag: str = ""
    tokens: int = 0
    source: str = ""
    stale: bool = False
    segment: Segment = Segment.DATA

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = max(1, len(self.content) // 3)


DEFAULT_WEIGHTS = {
    "sim": 1.0,
    "recency": 0.35,
    "refs": 0.25,
    "importance": 0.4,
}


class PageTable:
    """页表：维护页的生命周期，并按预算组装上下文。"""

    def __init__(self, db_path: str | Path, *, weights: dict[str, float] | None = None,
                 max_active: int = 5, top_k: int = 2) -> None:
        self.db_path = str(db_path)
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.max_active = max_active
        self.top_k = top_k
        self._pages: dict[int, Page] = {}
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                page_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                content     TEXT NOT NULL,
                state       TEXT NOT NULL,
                pinned      INTEGER NOT NULL DEFAULT 0,
                importance  REAL NOT NULL DEFAULT 0,
                refs        INTEGER NOT NULL DEFAULT 0,
                created_at  REAL NOT NULL,
                last_used   REAL NOT NULL,
                vec         TEXT,
                tag         TEXT NOT NULL DEFAULT '',
                tokens      INTEGER NOT NULL DEFAULT 0,
                source      TEXT NOT NULL DEFAULT '',
                stale       INTEGER NOT NULL DEFAULT 0,
                segment     TEXT NOT NULL DEFAULT 'data' 
            )
            """
        )
        self._migrate()
        self._conn.commit()
        self._load()
        self._enforce_capacity()

    def _migrate(self) -> None:
        """老库补列，避免升级后直接崩。"""

        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(pages)")}
        for name, decl in (("source", "TEXT NOT NULL DEFAULT ''"),
                           ("stale", "INTEGER NOT NULL DEFAULT 0"),
                           ("segment", "TEXT NOT NULL DEFAULT 'data'")):
            if name not in cols:
                self._conn.execute(f"ALTER TABLE pages ADD COLUMN {name} {decl}")
        self._conn.commit()

    # ---------- 持久化 ----------

    def _load(self) -> None:
        rows = self._conn.execute(
            "SELECT page_id, content, state, pinned, importance, refs,"
            " created_at, last_used, vec, tag, tokens, source, stale, segment FROM pages"
        ).fetchall()
        for r in rows:
            self._pages[r[0]] = Page(
                page_id=r[0],
                content=r[1],
                state=PageState(r[2]),
                pinned=bool(r[3]),
                importance=r[4],
                refs=r[5],
                created_at=r[6],
                last_used=r[7],
                vec=json.loads(r[8]) if r[8] else None,
                tag=r[9],
                tokens=r[10],
                source=r[11] if len(r) > 11 else "",
                stale=bool(r[12]) if len(r) > 12 else False,
                segment=Segment(r[13]) if len(r) > 13 else Segment.DATA,
            )

    def _persist(self, page: Page) -> None:
        self._conn.execute(
            "INSERT INTO pages (page_id, content, state, pinned, importance, refs,"
            " created_at, last_used, vec, tag, tokens, source, stale, segment)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(page_id) DO UPDATE SET content=excluded.content,"
            " state=excluded.state, pinned=excluded.pinned,"
            " importance=excluded.importance, refs=excluded.refs,"
            " last_used=excluded.last_used, vec=excluded.vec,"
            " tag=excluded.tag, tokens=excluded.tokens",
            (
                page.page_id, page.content, page.state.value, int(page.pinned),
                page.importance, page.refs, page.created_at, page.last_used,
                json.dumps(page.vec) if page.vec else None, page.tag, page.tokens,
                page.source, int(page.stale), page.segment.value,
            ),
        )
        self._conn.commit()

    # ---------- 页的增删改 ----------

    def add(
        self,
        content: str,
        *,
        vec: Sequence[float] | None = None,
        importance: float = 0.0,
        tag: str = "",
        pinned: bool = False,
        source: str = "",
        segment: Segment | str = Segment.DATA,
    ) -> Page:
        seg = Segment(segment) if isinstance(segment, str) else segment
        if seg in (Segment.SYSTEM, Segment.HUMAN, Segment.DAG_STATE):
            pinned = True
        from kernel.sanitize import strip_json

        content = strip_json(content)
        if not content:
            content = "（空）"
        page = Page(
            page_id=0,
            content=content,
            vec=list(vec) if vec else None,
            importance=importance,
            tag=tag,
            pinned=pinned,
            source=source,
            segment=seg,
        )
        cur = self._conn.execute(
            "INSERT INTO pages (content, state, pinned, importance, refs,"
            " created_at, last_used, vec, tag, tokens, source, stale, segment)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                page.content, page.state.value, int(page.pinned), page.importance,
                0, page.created_at, page.last_used,
                json.dumps(page.vec) if page.vec else None, page.tag, page.tokens,
                page.source, int(page.stale), page.segment.value,
            ),
        )
        self._conn.commit()
        page.page_id = int(cur.lastrowid)
        self._pages[page.page_id] = page
        self._enforce_capacity()
        return page

    def _enforce_capacity(self) -> list[int]:
        """活跃页超过上限，把最久没用且没钉住的换出去。"""

        active = [p for p in self._pages.values()
                  if p.state is PageState.ACTIVE and not p.pinned
                  and p.segment is Segment.DATA]
        if len(active) <= self.max_active:
            return []
        active.sort(key=lambda p: p.last_used)
        evicted: list[int] = []
        for page in active[: len(active) - self.max_active]:
            page.state = PageState.SWAPPED
            self._persist(page)
            evicted.append(page.page_id)
        return evicted

    def get(self, page_id: int) -> Page | None:
        return self._pages.get(page_id)

    def touch(self, page_id: int) -> None:
        page = self._pages.get(page_id)
        if not page:
            return
        page.last_used = time.time()
        page.refs += 1
        self._persist(page)

    def pin(self, page_id: int) -> None:
        page = self._pages[page_id]
        page.pinned = True
        page.state = PageState.ACTIVE
        self._persist(page)

    def unpin(self, page_id: int) -> None:
        page = self._pages[page_id]
        page.pinned = False
        self._persist(page)

    def swap_out(self, page_id: int) -> None:
        page = self._pages.get(page_id)
        if not page or page.pinned:
            return
        page.state = PageState.SWAPPED
        self._persist(page)

    def compact(self, *, tag: str = "interrupt", keep: int = 1) -> Page | None:
        """把同类旧页压成一句摘要，只留最后一个。"""

        targets = [p for p in self._pages.values()
                   if p.tag == tag and p.state is PageState.ACTIVE and not p.pinned]
        if len(targets) <= keep:
            return None
        targets.sort(key=lambda p: p.page_id)
        old = targets[:-keep]
        text = "；".join(p.content.strip()[:40] for p in old)
        for page in old:
            page.state = PageState.SWAPPED
            page.stale = True
            self._persist(page)
        return self.add(f"历史摘要：{text[:200]}", importance=0.4, tag="summary")

    def swap_out_segment(self, segment: Segment) -> int:
        """整段换出（用于新意图与代码无关时清空 DATA 段）。"""

        count = 0
        for page in self._pages.values():
            if page.segment is segment and page.state is PageState.ACTIVE and not page.pinned:
                page.state = PageState.SWAPPED
                self._persist(page)
                count += 1
        return count

    def invalidate_source(self, source: str) -> int:
        """源文件变了：相关页标记 stale 并换出，下次查询重新取。"""

        count = 0
        for page in self._pages.values():
            if page.source == source and not page.stale:
                page.stale = True
                page.state = PageState.SWAPPED
                self._persist(page)
                count += 1
        return count

    def drop(self, page_id: int) -> bool:
        page = self._pages.pop(page_id, None)
        if page is None:
            return False
        self._conn.execute("DELETE FROM pages WHERE page_id=?", (page_id,))
        self._conn.commit()
        return True

    def stale_sources(self) -> list[str]:
        return sorted({p.source for p in self._pages.values() if p.stale and p.source})

    def drop_stale(self, source: str | None = None) -> int:
        targets = [p for p in self._pages.values()
                   if p.stale and (source is None or p.source == source)]
        for page in targets:
            self._conn.execute("DELETE FROM pages WHERE page_id=?", (page.page_id,))
            self._pages.pop(page.page_id, None)
        self._conn.commit()
        return len(targets)

    def swap_in(self, page_id: int) -> None:
        page = self._pages.get(page_id)
        if not page:
            return
        page.state = PageState.ACTIVE
        page.last_used = time.time()
        self._persist(page)

    # ---------- 打分与选择 ----------

    def score(self, page: Page, *, query_vec: Sequence[float] | None = None, now: float | None = None) -> float:
        w = self.weights
        now = now or time.time()
        sim = _cosine(query_vec, page.vec) if query_vec and page.vec else 0.0
        age_hours = max(0.0, (now - page.created_at) / 3600.0)
        recency = 1.0 / (1.0 + math.log1p(age_hours))
        refs = math.log1p(page.refs)
        return sim * w["sim"] + recency * w["recency"] + refs * w["refs"] + page.importance * w["importance"]

    def ranked(
        self,
        *,
        query_vec: Sequence[float] | None = None,
        only_active: bool = False,
        segment: Segment | None = None,
    ) -> list[tuple[float, Page]]:
        now = time.time()
        pool: Iterable[Page] = self._pages.values()
        if only_active:
            pool = [p for p in pool if p.state is PageState.ACTIVE]
        if segment is not None:
            pool = [p for p in pool if p.segment is segment]
        scored = [(self.score(p, query_vec=query_vec, now=now), p) for p in pool]
        scored.sort(key=lambda x: (x[1].pinned, x[0]), reverse=True)
        return scored

    # ---------- 组装上下文 ----------

    def compose(
        self,
        *,
        query_vec: Sequence[float] | None = None,
        budget_tokens: int = 6000,
        segments: dict[str, str] | None = None,
        top_k: int | None = None,
    ) -> tuple[str, list[int]]:
        """段页式组装：固定段在前，DATA 段按相似度取 Top-K。

        segments: {"system": 文本, "human": 文本, "dag": 文本}，按此顺序拼接。
        """

        segments = segments or {}
        limit = self.top_k if top_k is None else top_k

        data_pages: list[Page] = []
        picked = 0
        used = 0
        budget = budget_tokens

        for _, page in self.ranked(query_vec=query_vec, only_active=True, segment=Segment.DATA):
            if page.stale or page.pinned:
                continue
            if picked >= limit or used + page.tokens > budget:
                if page.state is PageState.ACTIVE:
                    self.swap_out(page.page_id)
                continue
            if page.state is PageState.SWAPPED:
                self.swap_in(page.page_id)
            data_pages.append(page)
            picked += 1
            used += page.tokens

        resident = [p for p in self._pages.values()
                    if p.segment in (Segment.SYSTEM, Segment.HUMAN, Segment.DAG_STATE)
                    and p.state is PageState.ACTIVE]

        for page in data_pages + resident:
            self.touch(page.page_id)

        parts: list[str] = []
        titles = {"system": "系统段", "human": "人类指令段", "dag": "状态段"}
        for key in ("system", "human", "dag"):
            body = (segments.get(key) or "").strip()
            if body:
                parts.append(f"===【{titles[key]}】===\n{body}")
            extra = [p for p in resident
                     if p.segment.value == key and p.content not in body]
            if extra:
                merged = "\n".join(f"[p{p.page_id}] {p.content}" for p in extra)
                if not body:
                    parts.append(f"===【{titles[key]}】===\n{merged}")
                else:
                    parts.append(merged)

        if data_pages:
            body = "\n\n".join(f"[p{p.page_id}] {p.content}" for p in data_pages)
            parts.append(f"===【数据段】===\n{body}")

        chosen = [p.page_id for p in data_pages] + [p.page_id for p in resident]
        return "\n\n".join(parts), chosen

    # ---------- 缺页中断 ----------

    def search(self, keyword: str, roots: Sequence[str] = (), *, max_hits: int = 5, max_chars: int = 1200) -> list[Page]:
        """模型输出 [SEARCH: kw] 时调用：抓文件片段，作为新页插进上下文。"""

        hits: list[Page] = []
        for root in roots:
            base = Path(root)
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if len(hits) >= max_hits:
                    break
                if not path.is_file() or path.stat().st_size > 2_000_000:
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                idx = text.find(keyword)
                if idx < 0:
                    continue
                snippet = text[max(0, idx - 200): idx + max_chars]
                hits.append(self.add(f"{path}:{idx}\n{snippet}", importance=0.6, tag="fault"))
        return hits

    def stats(self) -> dict[str, int]:
        active = sum(1 for p in self._pages.values() if p.state is PageState.ACTIVE)
        return {
            "pages": len(self._pages),
            "active": active,
            "swapped": len(self._pages) - active,
            "pinned": sum(1 for p in self._pages.values() if p.pinned),
        }

    def close(self) -> None:
        self._conn.close()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / math.sqrt(na * nb)
