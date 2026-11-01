"""A-OS 虚拟上下文（MMU）。

上下文不是一段不断变长的 messages，而是一组页：
- 分页：文件 / 记忆 / 旧对话按需换入，不预先全塞
- 分段：SYSTEM / HUMAN / DAG_STATE / DATA，低优先级段才允许淘汰
- 换页：超出预算从低分页开始换出，正文回磁盘
- 缺页：模型发信号，内核去磁盘或向量索引里把页换进来
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

from kernel.tokens import estimate_tokens


class Segment(str, Enum):
    SYSTEM = "system"       # 宪法、工具协议：永不换出
    HUMAN = "human"         # 当前人类指令：本轮必带
    DAG_STATE = "dag"       # 任务状态与最近对话：常驻但可降级
    DATA = "data"           # 文件 / 记忆 / 旧片段：可淘汰


class PageState(str, Enum):
    ACTIVE = "active"
    SWAPPED = "swapped_out"


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
            self.tokens = estimate_tokens(self.content)

    def set_content(self, text: str, vec: Sequence[float] | None = None) -> None:
        self.content = text
        self.tokens = estimate_tokens(text)
        if vec is not None:
            self.vec = list(vec)


DEFAULT_WEIGHTS = {"sim": 1.0, "recency": 0.3, "refs": 0.15, "importance": 0.35}


class PageTable:
    def __init__(self, db_path: str | Path, *, weights: dict[str, float] | None = None,
                 max_active: int = 10) -> None:
        self.db_path = str(db_path)
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.max_active = max_active
        self._pages: dict[int, Page] = {}
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
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

    def _migrate(self) -> None:
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
            "SELECT page_id, content, state, pinned, importance, refs, created_at,"
            " last_used, vec, tag, tokens, source, stale, segment FROM pages"
        ).fetchall()
        for r in rows:
            page = Page(
                page_id=r[0], content=r[1], state=PageState(r[2]), pinned=bool(r[3]),
                importance=r[4], refs=r[5], created_at=r[6], last_used=r[7],
                vec=json.loads(r[8]) if r[8] else None, tag=r[9],
                tokens=r[10] or estimate_tokens(r[1]), source=r[11],
                stale=bool(r[12]), segment=Segment(r[13]),
            )
            self._pages[page.page_id] = page

    def _persist(self, page: Page) -> None:
        self._conn.execute(
            "INSERT INTO pages (page_id, content, state, pinned, importance, refs,"
            " created_at, last_used, vec, tag, tokens, source, stale, segment)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(page_id) DO UPDATE SET content=excluded.content,"
            " state=excluded.state, pinned=excluded.pinned,"
            " importance=excluded.importance, refs=excluded.refs,"
            " last_used=excluded.last_used, vec=excluded.vec,"
            " tag=excluded.tag, tokens=excluded.tokens,"
            " source=excluded.source, stale=excluded.stale, segment=excluded.segment",
            (
                page.page_id, page.content, page.state.value, int(page.pinned),
                page.importance, page.refs, page.created_at, page.last_used,
                json.dumps(page.vec) if page.vec else None, page.tag, page.tokens,
                page.source, int(page.stale), page.segment.value,
            ),
        )
        self._conn.commit()

    # ---------- 增删改 ----------

    def add(self, content: str, *, vec: Sequence[float] | None = None,
            importance: float = 0.0, tag: str = "", pinned: bool = False,
            source: str = "", segment: Segment | str = Segment.DATA) -> Page:
        seg = Segment(segment) if isinstance(segment, str) else segment
        if seg in (Segment.SYSTEM, Segment.HUMAN, Segment.DAG_STATE):
            pinned = True
        page = Page(
            page_id=0, content=content or "（空）", pinned=pinned,
            importance=importance, vec=list(vec) if vec else None,
            tag=tag, source=source, segment=seg,
        )
        cur = self._conn.execute(
            "INSERT INTO pages (content, state, pinned, importance, refs, created_at,"
            " last_used, vec, tag, tokens, source, stale, segment)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (page.content, page.state.value, int(page.pinned), page.importance, 0,
             page.created_at, page.last_used,
             json.dumps(page.vec) if page.vec else None, page.tag, page.tokens,
             page.source, 0, page.segment.value),
        )
        self._conn.commit()
        page.page_id = int(cur.lastrowid)
        self._pages[page.page_id] = page
        self._enforce_data_cap()
        return page

    def upsert(self, content: str, *, source: str, vec: Sequence[float] | None = None,
               importance: float = 0.6, tag: str = "", segment: Segment | str = Segment.DATA) -> Page:
        """按 source 去重：同一份记忆/文件只保留一页。"""

        page = self.find_by_source(source)
        seg = Segment(segment) if isinstance(segment, str) else segment
        if page is None:
            return self.add(content, vec=vec, importance=importance, tag=tag,
                            source=source, segment=seg)
        changed = page.content != content
        if changed:
            page.set_content(content, vec)
        page.segment = seg
        page.pinned = seg in (Segment.SYSTEM, Segment.HUMAN, Segment.DAG_STATE)
        page.stale = False
        page.state = PageState.ACTIVE
        page.last_used = time.time()
        page.importance = max(page.importance, importance)
        page.tag = tag or page.tag
        if changed or page.tokens != estimate_tokens(page.content):
            page.tokens = estimate_tokens(page.content)
        self._persist(page)
        return page

    def find_by_source(self, source: str, *, include_stale: bool = False) -> Page | None:
        for page in self._pages.values():
            if page.source == source and (include_stale or not page.stale):
                return page
        return None

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

    def swap_in(self, page_id: int) -> None:
        page = self._pages.get(page_id)
        if not page:
            return
        page.state = PageState.ACTIVE
        page.last_used = time.time()
        self._persist(page)

    def compact(self, *, tag: str = "turn", keep: int = 2) -> Page | None:
        """把同标签的旧页压成一句摘要，只留最近 keep 个。"""

        targets = [p for p in self._pages.values()
                   if p.tag == tag and p.segment is Segment.DATA
                   and p.state is PageState.ACTIVE]
        if len(targets) <= keep:
            return None
        targets.sort(key=lambda p: p.page_id)
        old = targets[:-keep]
        text = "；".join(p.content.strip()[:50] for p in old)
        for page in old:
            page.state = PageState.SWAPPED
            page.stale = True
            self._persist(page)
        return self.add(f"（早期对话摘要）{text[:220]}", importance=0.45,
                        tag="summary", segment=Segment.DATA)

    def invalidate_source(self, source: str) -> int:
        count = 0
        for page in self._pages.values():
            if page.source == source and not page.stale:
                page.stale = True
                page.state = PageState.SWAPPED
                page.pinned = False
                self._persist(page)
                count += 1
        return count

    def drop(self, page_id: int) -> bool:
        if page_id not in self._pages:
            return False
        self._pages.pop(page_id, None)
        self._conn.execute("DELETE FROM pages WHERE page_id=?", (page_id,))
        self._conn.commit()
        return True

    def swap_out_segment(self, segment: Segment) -> int:
        count = 0
        for page in self._pages.values():
            if page.segment is segment and page.state is PageState.ACTIVE and not page.pinned:
                page.state = PageState.SWAPPED
                self._persist(page)
                count += 1
        return count

    def _enforce_data_cap(self, *, keep: set[int] | None = None, budget: int | None = None) -> list[int]:
        """DATA 段超上限时，从最久没用的换出；保留本轮选中的页。"""

        keep = keep or set()
        cap_tokens = max(1800, int((budget or 6000) * 0.55))
        active = [p for p in self._pages.values()
                  if p.segment is Segment.DATA and p.state is PageState.ACTIVE
                  and not p.pinned and p.page_id not in keep]
        total = sum(p.tokens for p in active)
        evicted: list[int] = []
        for page in sorted(active, key=lambda p: p.last_used):
            if total <= cap_tokens and len(active) <= self.max_active:
                break
            page.state = PageState.SWAPPED
            self._persist(page)
            total -= page.tokens
            active.remove(page)
            evicted.append(page.page_id)
        return evicted

    # ---------- 打分 ----------

    def score(self, page: Page, *, query_vec: Sequence[float] | None = None,
              now: float | None = None) -> float:
        w = self.weights
        now = now or time.time()
        sim = _cosine(query_vec, page.vec) if query_vec and page.vec else 0.0
        age_hours = max(0.0, (now - page.created_at) / 3600.0)
        recency = 1.0 / (1.0 + math.log1p(age_hours))
        return (sim * w["sim"] + recency * w["recency"]
                + math.log1p(page.refs) * w["refs"] + page.importance * w["importance"])

    def ranked(self, *, query_vec: Sequence[float] | None = None,
               only_active: bool = False, segment: Segment | None = None) -> list[tuple[float, Page]]:
        now = time.time()
        pool: Iterable[Page] = self._pages.values()
        if only_active:
            pool = [p for p in pool if p.state is PageState.ACTIVE]
        if segment is not None:
            pool = [p for p in pool if p.segment is segment]
        scored = [(self.score(p, query_vec=query_vec, now=now), p) for p in pool]
        scored.sort(key=lambda x: (x[1].pinned, x[0]), reverse=True)
        return scored

    # ---------- 段页式组装 ----------

    def compose(self, *, query_vec: Sequence[float] | None = None,
                query_text: str = "", budget_tokens: int = 8000,
                segments: dict[str, str] | None = None,
                must_include: set[int] | None = None,
                max_data: int = 6) -> tuple[str, list[int]]:
        """固定段在前，最近状态常驻，DATA 段按相关度取 Top-K，总预算硬约束。"""

        segments = segments or {}
        must = set(must_include or ())
        budget = max(800, budget_tokens)
        fixed = {k: (segments.get(k) or "").strip() for k in ("system", "human", "dag")}
        used = sum(estimate_tokens(v) for v in fixed.values())

        resident = sorted(
            (p for p in self._pages.values()
             if p.state is PageState.ACTIVE
             and p.segment in (Segment.SYSTEM, Segment.HUMAN, Segment.DAG_STATE)),
            key=lambda p: p.page_id,
        )
        used += sum(p.tokens for p in resident)
        data_budget = max(0, budget - used - 200)

        cands: list[tuple[float, float, float, Page]] = []
        for page in self._pages.values():
            if page.segment is not Segment.DATA or page.stale or page.pinned:
                continue
            sim = _cosine(query_vec, page.vec) if (query_vec and page.vec) else 0.0
            lex = _lexical(query_text, page.content) if query_text else 0.0
            cands.append((sim, lex, self.score(page, query_vec=query_vec), page))

        chosen: list[Page] = []
        chosen_ids: set[int] = set()
        chosen_tokens = 0
        by_id = {p.page_id: p for _, _, _, p in cands}
        for pid in must:
            page = by_id.get(pid)
            if page is None:
                continue
            chosen.append(page)
            chosen_ids.add(pid)
            chosen_tokens += page.tokens

        for sim, lex, score, page in sorted(cands, key=lambda x: x[2], reverse=True):
            if page.page_id in chosen_ids:
                continue
            if sim < 0.50 and lex < 0.18:
                continue
            if len(chosen) >= max_data:
                break
            if chosen_tokens + page.tokens > data_budget:
                continue
            chosen.append(page)
            chosen_ids.add(page.page_id)
            chosen_tokens += page.tokens

        for page in chosen:
            if page.state is PageState.SWAPPED:
                page.state = PageState.ACTIVE
            page.last_used = time.time()
            page.refs += 1
            self._persist(page)
        self._enforce_data_cap(keep=chosen_ids, budget=budget)

        parts: list[str] = []
        titles = {"system": "系统段", "human": "人类指令段", "dag": "状态段"}
        for key in ("system", "human", "dag"):
            body = fixed[key]
            pages = [p for p in resident if p.segment.value == key]
            if body:
                parts.append(f"===【{titles[key]}】===\n{body}")
            if pages:
                text = "\n".join(_render_page(p) for p in pages)
                parts.append(text if body else f"===【{titles[key]}】===\n{text}")

        if chosen:
            body = "\n\n".join(_render_page(p) for p in chosen)
            parts.append(f"===【数据段（按需换入）】===\n{body}")

        ids = [p.page_id for p in chosen] + [p.page_id for p in resident]
        return "\n\n".join(parts), ids

    # ---------- 观察 ----------

    def stats(self) -> dict:
        active = [p for p in self._pages.values() if p.state is PageState.ACTIVE]
        seg = {s.value: 0 for s in Segment}
        for page in active:
            seg[page.segment.value] += 1
        return {
            "pages": len(self._pages),
            "active": len(active),
            "swapped": len(self._pages) - len(active),
            "active_tokens": sum(p.tokens for p in active),
            "segments": seg,
            "faults": sum(1 for p in self._pages.values() if p.tag == "fault"),
            "memories": sum(1 for p in self._pages.values() if p.tag == "memory"),
        }

    def close(self) -> None:
        self._conn.close()


def _render_page(page: Page) -> str:
    label = f"[p{page.page_id}"
    if page.source:
        label += f" {page.source}"
    label += "]"
    return f"{label} {page.content}"


def _cosine(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _grams(text: str) -> set[str]:
    low = (text or "").lower()
    chars = re.sub(r"\s+", "", low)
    grams = {chars[i: i + 2] for i in range(max(0, len(chars) - 1))}
    grams.update(re.findall(r"[a-z0-9_]{2,}", low))
    return grams


def _lexical(a: str, b: str) -> float:
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / math.sqrt(len(ga) * len(gb))
