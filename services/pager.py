"""pager 服务：拥有段页式 MMU、长期记忆、图索引。

内核只负责缺页陷阱和 IPC；这里决定"换哪几页、换多少"。
图节点只是背衬存储，最终都映射成 DATA 页。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from kernel import persona
from kernel.dialogue import Dialogue
from kernel.graph import GraphStore
from kernel.memory import MemoryStore
from kernel.page_table import PageTable, Segment
from kernel.retriever import GraphRetriever


class PagerService:
    name = "pager"

    def __init__(self, home: str | Path, encoder) -> None:
        self.home = Path(home)
        self.encoder = encoder
        self.graph = GraphStore(self.home / "graph.db", encoder)
        self.retriever = GraphRetriever(self.graph, encoder)
        self.mmu = PageTable(self.home / "pages.db", max_active=10)
        self.memory = MemoryStore(self.home / "brain.db", encoder)
        self.dialogue = Dialogue(self.home / "brain.db", session="main")
        self.faults = 0

    async def boot(self) -> None:
        self.mmu.upsert(persona.system_prompt(), source="sys:persona",
                        vec=self.encoder.encode_one(persona.system_prompt()),
                        importance=1.0, tag="system", segment=Segment.SYSTEM)
        if self.graph.stats()["active"] >= 3:
            try:
                await asyncio.to_thread(self.graph.spectral, k=8)
            except Exception:
                pass

    async def handle(self, method: str, params: dict) -> Any:
        if method == "begin_turn":
            return self.begin_turn(int(params["pid"]), str(params.get("text", "")))
        if method == "compose":
            return self.compose(
                int(params["pid"]), str(params.get("query", "")),
                str(params.get("dag", "")), list(params.get("profile") or []),
                set(params.get("must_include") or []), int(params.get("budget", 8000)),
            )
        if method == "fault":
            return self.fault(int(params["pid"]), list(params.get("targets") or []),
                              set(params.get("hot") or []))
        if method == "invalidate":
            n = self.mmu.invalidate_source(str(params.get("source", "")))
            return {"invalidated": n}
        if method == "finish_turn":
            return self.finish_turn(int(params["pid"]), str(params.get("user", "")),
                                    str(params.get("answer", "")))
        if method == "resolve":
            return self.retriever.context(str(params.get("query", "")),
                                          k=int(params.get("k", 5)))
        if method == "stats":
            return {
                "graph": self.graph.stats(),
                "pages": self.mmu.stats(pid=params.get("pid")),
                "memory": self.memory.stats(),
                "turns": self.dialogue.count(),
                "faults": self.faults,
            }
        if method == "ping":
            return "pong"
        raise ValueError(f"pager 没有方法：{method}")

    # ---------- 一轮的页 ----------

    def begin_turn(self, pid: int, text: str) -> dict:
        text = (text or "").strip()
        self.dialogue.add("user", text)
        self._roll_pages(pid)
        shown = text if len(text) <= 4000 else text[:4000] + "…（已截断）"
        self.mmu.add(f"用户：{shown}", segment=Segment.HUMAN, pinned=True,
                     tag="turn", importance=1.0, pid=pid)

        # 系统页每轮刷新时间，内容仍是同一页
        sys_text = persona.system_prompt()
        self.mmu.upsert(sys_text, source="sys:persona",
                        vec=self.encoder.encode_one(sys_text),
                        importance=1.0, tag="system", segment=Segment.SYSTEM, pid=0)

        profile = self.memory.profile()
        wants_past = any(w in text for w in ("上次", "之前", "聊过", "说过", "那天", "刚才", "那个"))
        kinds = None if wants_past else ("identity", "preference", "fact", "plan", "summary")
        seen = {p.id for p in profile}
        recalled = [m for _, m in self.memory.recall(text, k=8, kinds=kinds)
                    if m.id not in seen]
        for mem in recalled:
            self.mmu.upsert(f"（长期记忆）{mem.text}", source=f"mem:{mem.id}",
                            vec=self.encoder.encode_one(mem.text),
                            importance=min(0.9, mem.importance),
                            tag="memory", segment=Segment.DATA, pid=pid)
        return {
            "pid": pid,
            "profile": [m.text for m in profile],
            "recalled": [m.text for m in recalled],
            "pages": self.mmu.stats(),
        }

    def compose(self, pid: int, query: str, dag: str, profile: list[str],
                must_include: set[int], budget: int) -> dict:
        profile_text = ""
        if profile:
            profile_text = "关于用户：\n" + "\n".join(f"- {t}" for t in profile)
        dag_text = (profile_text + ("\n最近状态：\n" + dag if dag else "")).strip()
        prompt, ids = self.mmu.compose(
            query_vec=self.encoder.encode_one(query),
            query_text=query,
            budget_tokens=budget,
            segments={"system": "", "human": "", "dag": dag_text},
            must_include=must_include,
            max_data=6,
            pid=pid,
        )
        return {"prompt": prompt, "pages": ids, "budget": budget}

    def fault(self, pid: int, targets: list[str], hot: set[int] | list[int]) -> dict:
        """图优先：query → 3~5 个图节点 → DATA 页。"""

        hot = set(hot or [])
        messages: list[str] = []
        for target in targets[:2]:
            target = target.strip().lstrip("./")
            if not target:
                continue
            nodes = self.retriever.retrieve(target, k=3)
            added = 0
            for node in nodes:
                page = self.mmu.upsert(
                    self.graph.render(node), source=f"graph:{node.id}",
                    vec=node.vec if node.vec is not None else self.encoder.encode_one(node.title),
                    importance=node.importance, tag="graph", segment=Segment.DATA,
                    pid=pid,
                )
                hot.add(page.page_id)
                added += 1
            if added:
                self.faults += added
                messages.append(f"换入 {added} 个图节点（{target}）")
                continue
            hits = self.memory.recall(target, k=2)
            if hits:
                for _score, mem in hits:
                    page = self.mmu.upsert(f"（长期记忆）{mem.text}", source=f"mem:{mem.id}",
                                           vec=self.encoder.encode_one(mem.text),
                                           importance=min(0.9, mem.importance),
                                           tag="memory", segment=Segment.DATA, pid=pid)
                    hot.add(page.page_id)
                    added += 1
                messages.append(f"图里没有，换入 {added} 条长期记忆")
            else:
                messages.append(f"图里没有：{target}")
        return {"messages": "；".join(messages) or "没有可换入的内容", "hot": sorted(hot)}

    def finish_turn(self, pid: int, user: str, answer: str) -> dict:
        self.dialogue.add("assistant", answer)
        self.mmu.add(f"A-OS：{answer}", segment=Segment.DAG_STATE, pinned=True,
                     tag="answer", importance=0.7, source=f"turn:{pid}", pid=pid)
        self.mmu.compact(tag="turn", keep=2, pid=pid)
        if user:
            self.memory.add(user, kind="episode", importance=0.4, source="chat")
        return {"pages": self.mmu.stats(), "graph": self.graph.stats()}

    def _roll_pages(self, pid: int) -> None:
        for page in list(self.mmu._pages.values()):
            if page.pid != pid:
                continue
            if page.segment is Segment.HUMAN and page.tag == "turn":
                page.segment = Segment.DAG_STATE
                page.pinned = True
                self.mmu._persist(page)

        recent = sorted(
            (p for p in self.mmu._pages.values()
             if p.pid == pid and p.segment is Segment.DAG_STATE
             and p.state.value == "active"),
            key=lambda p: p.page_id,
        )
        for page in recent[:-6]:
            page.segment = Segment.DATA
            page.pinned = False
            self.mmu._persist(page)

        recent = recent[-6:]
        total = sum(p.tokens for p in recent)
        while len(recent) > 2 and total > 2400:
            page = recent.pop(0)
            page.segment = Segment.DATA
            page.pinned = False
            total -= page.tokens
            self.mmu._persist(page)
        self.mmu.compact(tag="turn", keep=2, pid=pid)

    async def close(self) -> None:
        self.mmu.close()
        self.memory.close()
        self.dialogue.close()
        self.graph.close()
