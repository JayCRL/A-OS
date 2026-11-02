"""pager 服务：拥有段页式 MMU、长期记忆、图索引。

内核只负责缺页陷阱和 IPC；这里决定"换哪几页、换多少"。
图节点只是背衬存储，最终都映射成 DATA 页。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from kernel import persona
from kernel.consolidator import Consolidator, Episode
from kernel.dialogue import Dialogue
from kernel.graph import GraphStore
from kernel.memory import MemoryStore
from kernel.page_table import PageTable, Segment
from kernel.retriever import GraphRetriever


class PagerService:
    name = "pager"

    def __init__(self, home: str | Path, encoder, bus=None) -> None:
        self.home = Path(home)
        self.encoder = encoder
        self.bus = bus
        self.graph = GraphStore(self.home / "graph.db", encoder)
        self.retriever = GraphRetriever(self.graph, encoder)
        self.mmu = PageTable(self.home / "pages.db", max_active=10)
        self.memory = MemoryStore(self.home / "brain.db", encoder)
        self.dialogue = Dialogue(self.home / "brain.db", session="main")
        self.faults = 0
        self.consolidator: Consolidator | None = None
        self.memory._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                pid INTEGER NOT NULL DEFAULT 0,
                query TEXT NOT NULL DEFAULT '',
                page_ids TEXT NOT NULL DEFAULT '[]',
                used_ids TEXT NOT NULL DEFAULT '[]',
                outcome INTEGER NOT NULL DEFAULT 0,
                tokens INTEGER NOT NULL DEFAULT 0,
                latency REAL NOT NULL DEFAULT 0
            )
            """
        )
        self.memory._conn.commit()

    async def boot(self) -> None:
        # 旧版页表里 pid=0 的 HUMAN/DAG/DATA 是历史遗留，不再参与新进程地址空间
        for page in list(self.mmu._pages.values()):
            if page.pid == 0 and page.segment is not Segment.SYSTEM and page.state.value == "active":
                page.pinned = False
                page.state = page.state.__class__.SWAPPED
                self.mmu._persist(page)
        self._reap_old_pids(0, keep=0)
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
        if method == "commit_memory":
            return {"added": self.commit_memory(list(params.get("items") or []))}
        if method == "prefetch":
            return await self.prefetch(int(params["pid"]), str(params.get("query", "")))
        if method == "fault":
            return await self.fault(int(params["pid"]), list(params.get("targets") or []),
                                    set(params.get("hot") or []))
        if method == "commit_episode":
            return self.commit_episode(dict(params))
        if method == "recent_history":
            return [{"id": t.id, "role": t.role, "text": t.text, "ts": t.ts}
                    for t in self.dialogue.recent(limit=int(params.get("limit", 20)))]
        if method == "supersede_facts":
            return {"changed": self.memory.supersede(str(params.get("text", "")))}
        if method == "decay_memory":
            return {"changed": self.memory.decay(float(params.get("factor", 0.995)))}
        if method == "commit_summary":
            return self.commit_summary(str(params.get("text", "")), str(params.get("scope", "session")))
        if method == "log_event":
            return self.log_event(
                int(params.get("pid", 0)), str(params.get("query", "")),
                list(params.get("page_ids") or []), list(params.get("used_ids") or []),
                int(params.get("outcome", 0)), int(params.get("tokens", 0)),
                float(params.get("latency", 0.0)),
            )
        if method == "event_stats":
            return self.event_stats()
        if method == "history":
            return [{"role": t.role, "text": t.text, "ts": t.ts}
                    for t in self.dialogue.history(limit=int(params.get("limit", 100)))]
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
                "events": self.event_stats(),
            }
        if method == "ping":
            return "pong"
        raise ValueError(f"pager 没有方法：{method}")

    # ---------- 一轮的页 ----------

    def commit_memory(self, items: list[dict]) -> int:
        if not items:
            return 0
        return self.memory.add_many(items, source="chat")

    def commit_summary(self, text: str, scope: str = "session") -> dict:
        text = (text or "").strip()
        if not text:
            return {"added": False}
        mem = self.memory.add(text, kind="summary", importance=0.75,
                              source=f"consolidation:{scope}")
        return {"added": bool(mem), "total": self.memory.stats()["total"]}

    def log_event(self, pid: int, query: str, page_ids: list[int],
                  used_ids: list[int], outcome: int, tokens: int, latency: float) -> dict:
        import json as _json
        import time as _time
        self.memory._conn.execute(
            "INSERT INTO memory_events (ts,pid,query,page_ids,used_ids,outcome,tokens,latency)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (_time.time(), pid, query[:1000], _json.dumps(page_ids), _json.dumps(used_ids),
             outcome, tokens, latency),
        )
        self.memory._conn.commit()
        return {"ok": True}

    def event_stats(self) -> dict:
        row = self.memory._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(tokens),0), COALESCE(SUM(outcome),0) FROM memory_events"
        ).fetchone()
        return {"total": int(row[0]), "tokens": int(row[1]), "reward": int(row[2])}

    def commit_episode(self, data: dict) -> dict:
        if self.consolidator is None:
            self.consolidator = Consolidator(self.graph)
        episode = Episode(
            text=str(data.get("text", "")),
            module=str(data.get("module", "")),
            task=str(data.get("task", "")),
            success=bool(data.get("success", False)),
            lesson=str(data.get("lesson", "")),
            artifact=str(data.get("artifact", "")),
            attempts=int(data.get("attempts", 0) or 0),
            action=str(data.get("action", "")),
            result=str(data.get("result", "")),
            depends_on=list(data.get("depends_on") or []),
        )
        self.consolidator.ingest(episode)
        if data.get("flush"):
            self.consolidator.flush()
        return self.graph.stats()

    def begin_turn(self, pid: int, text: str) -> dict:
        text = (text or "").strip()
        self._reap_old_pids(pid, keep=2)
        turn = self.dialogue.add("user", text)
        self._roll_pages(pid)
        shown = text if len(text) <= 4000 else text[:4000] + "…（已截断）"
        self.mmu.add(f"用户：{shown}", segment=Segment.HUMAN, pinned=True,
                     tag="turn", importance=1.0, pid=pid)
        # 只有新进程才需要导入历史；同一会话进程的 DAG 页还在
        has_dag = any(p.pid == pid and p.segment is Segment.DAG_STATE
                      and p.state.value == "active" for p in self.mmu._pages.values())
        if not has_dag:
            self._import_history(pid, before_id=turn.id, limit=8)

        # 系统页每轮刷新时间，内容仍是同一页
        sys_text = persona.system_prompt()
        self.mmu.upsert(sys_text, source="sys:persona",
                        vec=self.encoder.encode_one(sys_text),
                        importance=1.0, tag="system", segment=Segment.SYSTEM, pid=0)

        profile = self.memory.profile()
        wants_past = any(w in text for w in ("上次", "之前", "聊过", "说过", "那天", "刚才", "那个"))
        kinds = None if wants_past else ("identity", "preference", "fact", "plan", "summary")
        seen = {p.id for p in profile}
        # 多路探测：当前这句话 + 最近两轮用户话，避免笼统指代召回失败
        probes = [text]
        for turn in self.dialogue.recent(limit=4, before_id=turn.id):
            if turn.role == "user" and turn.text not in probes:
                probes.append(turn.text)
        recalled = []
        for probe in probes[:3]:
            for _score, mem in self.memory.recall(probe, k=8, kinds=kinds):
                if mem.id not in seen:
                    seen.add(mem.id)
                    recalled.append(mem)
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

    async def prefetch(self, pid: int, query: str) -> dict:
        """自动预取：图节点 + VFS 文件块，不需要模型发缺页。"""

        if not query.strip():
            return {"pages": 0}
        added = 0
        nodes = self.retriever.retrieve(query, k=3)
        if any(n.type != "project" for n in nodes):
            for node in nodes:
                self.mmu.upsert(
                    self.graph.render(node), source=f"graph:{node.id}",
                    vec=node.vec if node.vec is not None else self.encoder.encode_one(node.title),
                    importance=node.importance, tag="graph", segment=Segment.DATA, pid=pid,
                )
                added += 1
        if self.bus is not None:
            try:
                hits = await self.bus.call("vfs", "semantic", query=query, k=3)
            except Exception:
                hits = []
            for hit in hits[:2]:
                if float(hit.get("score", 0)) < 0.30:
                    continue
                body = str(hit.get("text", ""))[:1800]
                self.mmu.upsert(
                    f"文件 {hit.get('path', '')}：\n{body}",
                    source=str(hit.get("source", "")),
                    vec=self.encoder.encode_one(body[:800]),
                    importance=0.85, tag="file", segment=Segment.DATA, pid=pid,
                )
                added += 1
        return {"pages": added}

    async def fault(self, pid: int, targets: list[str], hot: set[int] | list[int]) -> dict:
        """图优先：图节点 → VFS 文件块 → 长期记忆，全部映射成 DATA 页。"""

        hot = set(hot or [])
        messages: list[str] = []
        for target in targets[:2]:
            target = target.strip().lstrip("./")
            if not target:
                continue
            nodes = self.retriever.retrieve(target, k=3)
            # 只有 project 不算命中；否则会永远挡住 VFS/记忆回退
            if not any(n.type != "project" for n in nodes):
                nodes = []
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

            file_hits = []
            if self.bus is not None:
                try:
                    file_hits = await self.bus.call("vfs", "semantic", query=target, k=3)
                except Exception:
                    file_hits = []
            for hit in file_hits[:2]:
                if float(hit.get("score", 0.0)) < 0.30:
                    continue
                body = str(hit.get("text", ""))[:1800]
                page = self.mmu.upsert(
                    f"文件 {hit.get('path', '')}：\n{body}",
                    source=str(hit.get("source", "")),
                    vec=self.encoder.encode_one(body[:800]),
                    importance=0.85, tag="file", segment=Segment.DATA, pid=pid,
                )
                hot.add(page.page_id)
                added += 1
                self.faults += 1
                messages.append(f"图里没有，换入文件块 {hit.get('path', '')}（{hit.get('score', 0):.2f}）")
            if added:
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

    def _import_history(self, pid: int, *, before_id: int, limit: int = 8) -> None:
        """把最近几轮对话映射成当前进程的 DAG_STATE 页。"""

        for turn in self.dialogue.recent(limit=limit, before_id=before_id):
            who = "用户" if turn.role == "user" else "A-OS"
            content = f"{who}：{turn.text[:1200]}"
            self.mmu.upsert(
                content, source=f"hist:{turn.id}",
                vec=self.encoder.encode_one(content[:800]),
                importance=0.65, tag="history",
                segment=Segment.DAG_STATE, pid=pid,
            )

    def _reap_old_pids(self, current: int, *, keep: int = 2) -> None:
        """进程退出后，旧 pid 的工作集降温；只保留最近 keep 个进程活跃。"""

        pids = sorted({p.pid for p in self.mmu._pages.values()
                       if p.pid not in (0, current) and p.state.value == "active"})
        targets = pids[:-keep] if keep > 0 else pids
        for old in targets:
            for page in self.mmu._pages.values():
                if page.pid == old and page.state.value == "active":
                    page.pinned = False
                    page.state = page.state.__class__.SWAPPED
                    self.mmu._persist(page)

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
