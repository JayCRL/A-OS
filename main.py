"""A-OS 引导与主循环。

终端单入口：输入即事件，事件进 spooler，合并后由内核对状态树与页表做一次推进。
模型调用走 mock，接口留好。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel.io_spooler import ActionKind, HumanAction, IOSpooler, InterruptEvent
from kernel.kernel_scheduler import Job, KernelScheduler
from kernel.page_table import PageTable, Segment
from kernel.state_dag import NodeStatus, StateDAG
from drivers.model import DeepSeekKernel, OfflineKernel
from drivers import protocol
from drivers.embedding import load_encoder
from kernel.executor import Executor
from kernel.shell_sandbox import ShellSandbox
from kernel.depgraph import DepGraph
from kernel.lock import PathLocks
from kernel.memindex import MemoryIndex
from kernel import sanitize
from drivers.mcp_client import McpRegistry
from kernel.progress import ProgressTracker
from kernel.journal import Journal

DATA = Path(os.environ.get("AOS_HOME", "/opt/aos/data"))
SEARCH_RE = re.compile(r"\[SEARCH:\s*([^\]]+)\]")


class AOS:
    def __init__(self, home: Path = DATA, *, idle: float = 2.0) -> None:
        self.home = home
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "memory").mkdir(exist_ok=True)

        self.mmu = PageTable(self.home / "pages.db")
        self.dag = StateDAG(self.home / "dag.db")
        self.sched = KernelScheduler(max_workers=3)
        self.spooler = IOSpooler(idle_seconds=idle)
        self.kernel = DeepSeekKernel() if DeepSeekKernel().ready else OfflineKernel()
        self.guard_factory = ProgressTracker
        self.sandbox = ShellSandbox(self.home / "workspace", allow_shell := True) if False else ShellSandbox(self.home / "workspace")
        self.mcp = McpRegistry(Path("/opt/aos/mcp.json"))
        self.depgraph = DepGraph.build(self.home / "workspace")
        self.path_locks = self._make_locks()
        self.executor = Executor(self.home / "workspace", allow_shell=True,
                                 sandbox=self.sandbox, mcp=self.mcp,
                                 on_write=self._on_file_written,
                                 path_locks=self.path_locks)
        self.max_rounds = int(os.environ.get("AOS_MAX_ROUNDS", "6"))
        self.encoder = load_encoder()
        self.memindex = MemoryIndex(self.home / "chunks.db", self.encoder, self.home / "workspace")
        self.journal = Journal(self.home / "tasks.db")
        self.broker = None
        self.budget = int(os.environ.get("AOS_CTX_BUDGET", "8000"))
        self.log: list[str] = []
        self._running = True
        self._head = self.dag.head()

    # ---------- 启动 ----------

    async def boot(self) -> None:
        await self.sched.start()
        started = await self.mcp.load()
        if started:
            self.emit(f"MCP 已挂载：{', '.join(started)}")
        if self._head is None:
            self._head = self.dag.add("系统就绪", kind="boot")
        self._sanitize_pages()
        self._migrate_vectors()
        indexed = self.memindex.refresh()
        if indexed:
            self.emit(f"记忆索引：{indexed} 个文件入库 {self.memindex.stats()}")
        self._recover()
        self.emit(f"boot: 节点 {self._head.node_id}，页 {self.mmu.stats()['pages']} 条，"
                  f"模型 {'在线' if self.kernel.ready else '离线'}，编码器 {self.encoder.name}({self.encoder.dim}d)")

    def _sanitize_pages(self) -> None:
        """一次性清洗历史页里残留的 JSON 日志。"""

        fixed = 0
        for page in list(self.mmu._pages.values()):
            raw = page.content
            clean = sanitize.strip_json(raw)
            if clean and clean != raw:
                page.content = clean[:1500]
                self.mmu._persist(page)
                fixed += 1
        dag_fixed = 0
        for node in list(self.dag._nodes.values()):
            clean = sanitize.strip_json(node.content)
            if clean and clean != node.content:
                self.dag.update_content(node.node_id, clean[:200])
                dag_fixed += 1

        dropped = 0
        for page in list(self.mmu._pages.values()):
            text = page.content.replace("人类指令：", "").strip()
            if page.segment is Segment.DATA and len(text) <= 12 and page.tag in ("interrupt", "summary"):
                self.mmu.drop(page.page_id)
                dropped += 1

        if fixed:
            self.emit(f"[净化] 清理 {fixed} 页里的原始指令日志", cls="sys")
        if dropped:
            self.emit(f"[净化] 丢弃 {dropped} 页闲聊垃圾", cls="sys")
        if dag_fixed:
            self.emit(f"[净化] 清理 {dag_fixed} 个状态节点里的原始日志", cls="sys")

    def _recover(self) -> None:
        pending = self.journal.recover()
        for rec in pending:
            if not self.journal.claim(rec.task_id):
                continue
            self.emit(f"恢复任务 {rec.task_id}：{rec.goal[:30]}（第 {rec.round} 轮续跑）")
            node = self.dag.get(rec.node_id) if rec.node_id else None
            if node is None:
                self.journal.finish(rec.task_id, "failed")
                continue
            self.sched.submit(Job(
                priority=1, seq=0, name=f"resume:{rec.task_id}",
                run=lambda r=rec, n=node: self._step(n, None, task_id=r.task_id, start_round=r.round),
            ))

    def _migrate_vectors(self) -> None:
        fixed = 0
        for page in list(self.mmu._pages.values()):
            if page.vec is None or len(page.vec) != self.encoder.dim:
                page.vec = self.encoder.encode_one(page.content)
                self.mmu._persist(page)
                fixed += 1
        if fixed:
            self.emit(f"向量迁移：{fixed} 页重新编码")

    def emit(self, line: str, *, cls: str = "") -> None:
        self.log.append({"text": line, "cls": cls})
        if len(self.log) > 400:
            del self.log[:-300]
        print(line, flush=True)

    # ---------- 终端 ----------

    async def terminal(self) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except (EOFError, RuntimeError):
                break
            if not line:
                break
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.strip() in ("exit", "quit", ":q"):
                break
            await self._handle_input(line)

    async def _handle_input(self, line: str) -> None:
        if line.startswith("!"):
            await self.spooler.feed(HumanAction(ActionKind.SIGNAL, line[1:] or "强制"))
        elif line.startswith("@"):
            path, _, text = line[1:].partition(" ")
            await self.spooler.feed(HumanAction(ActionKind.CONTENT, text or path, path=path))
        else:
            await self.spooler.feed(HumanAction(ActionKind.INTENT, line))

    # ---------- 事件循环 ----------

    async def event_loop(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self.spooler.next_event(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            await self._dispatch(event)

    async def _dispatch(self, event: InterruptEvent) -> None:
        victims = await self.sched.preempt()
        if victims:
            self.emit(f"中断：挂起 {len(victims)} 个后台任务")

        if event.is_force and self._head is not None:
            node = self.dag.cow(self._head.node_id, f"[{event.summary}]", kind="human_interrupt")
        else:
            node = self.dag.add(event.summary, parent_id=self._head.node_id if self._head else None,
                                kind="goal" if self._head is None else "step")
        self._head = node
        for old in list(self.mmu._pages.values()):
            if old.segment is Segment.HUMAN:
                old.segment = Segment.DATA
                old.pinned = False
                self.mmu._persist(old)
        page = self.mmu.add(f"人类指令：{event.summary}", importance=1.0,
                            tag="interrupt", segment=Segment.HUMAN)

        if sanitize.needs_code(event.summary):
            self.mmu.swap_in(0) if False else None
        else:
            cleared = self.mmu.swap_out_segment(Segment.DATA)
            if cleared:
                self.emit(f"[清段] 本轮与代码无关，DATA 段换出 {cleared} 页", cls="sys")

        task_id = self.journal.register(event.summary, node.node_id)
        self.journal.claim(task_id)
        job_id = self.sched.submit(Job(
            priority=1, seq=0, name="step",
            run=lambda: self._step(node, event, task_id=task_id),
            on_cancel=lambda jid: self._checkpoint(node, jid),
        ))
        self.emit(f"入队 {job_id} → {event.summary}")

    async def _step(self, node, event, *, task_id: str | None = None, start_round: int = 0) -> None:
        summary = event.summary if event is not None else (node.content or "恢复任务")
        if self._is_doc_task(summary):
            await self._write_doc(summary, node, task_id=task_id)
            return

        trace: list[str] = []
        current = node
        done = False
        guard = self.guard_factory()
        self.executor.guard = guard

        for rnd in range(start_round + 1, self.max_rounds + 1):
            sections = {
                "system": protocol.SYSTEM,
                "human": f"任务目标：{summary}",
                "dag": (f"文件树：\n{self._tree()}\n"
                        f"依赖：\n{self.depgraph.render(limit=8)}\n"
                        f"台账：{guard.ledger()}\n"
                        f"最近路径：\n{self.dag.render(limit=4)}"),
            }
            prompt, pages = self.mmu.compose(
                query_vec=self._vec(summary),
                budget_tokens=self.budget,
                segments=sections,
            )
            if trace:
                prompt += "\n\n最近两步：\n" + "\n".join(trace[-2:])

            action = None
            fault_targets: list[str] = []
            fault_seen: set[str] = locals().get("fault_seen") or set()
            out_tokens = int(os.environ.get("AOS_OUT_TOKENS", "3000"))
            for attempt in range(2):
                raw = await self.kernel.chat(
                    [
                        {"role": "user", "content": prompt + (
                            "\n[系统] 上一次输出被截断或不是合法 JSON。这次要么用 append 分块写，"
                            "要么只输出一个短 JSON。" if attempt else "")},
                    ],
                    max_tokens=out_tokens if attempt == 0 else 1200,
                )
                wanted = protocol.faults(raw.text)
                if wanted:
                    fault_targets = wanted
                    break

                action = protocol.parse(raw.text)
                if not (action.name == "done" and str(action.args.get("summary", "")).startswith("无法解析")):
                    break
                salvaged = protocol.salvage(raw.text)
                if salvaged is not None:
                    self.emit("[续写] 输出被截断，已追加，下一轮从断点继续")
                    action = salvaged
                    break
                self.emit(f"[重试] 输出不合法，第 {attempt + 2} 次")
            if fault_targets:
                await self._page_fault(fault_targets, guard, fault_seen)
                continue

            if action is None:
                continue

            if action.name == "invalid":
                bad = action.args.get("name") or "非 JSON"
                hint = "[系统] 动作不合法：" + str(bad)
                if str(bad) == "read":
                    hint += "。read 已废弃，需要文件内容请输出 <|PAGE_FAULT: 文件名|>"
                else:
                    hint += "。只输出一个合法 JSON 动作，或 done"
                trace.append(hint)
                self.emit(f"[退回] {hint[:70]}", cls="err")
                continue

            obs = await self.executor.run(action.name, action.args)
            guard.observe(action.name, action.args, obs.ok)


            trace.append(f"#{rnd} {action.name} -> {str(obs)[:300]}")

            nudge = guard.intervention()
            if nudge:
                trace.append(nudge)
                self.emit(f"[看门狗] {nudge[:60]}")

            step_node = self.dag.add(
                sanitize.summarize(action.name, action.args, obs.ok, str(obs)),
                parent_id=current.node_id, kind="act",
            )
            current = step_node
            self._head = current
            if task_id:
                self.journal.checkpoint(task_id, node_id=current.node_id, round_no=rnd, steps=len(trace))
            self.emit(f"[{rnd}] {action.name} → {str(obs)[:100]}")

            if action.name == "done":
                done = True
                if task_id:
                    self.journal.finish(task_id)
                self.dag.prune(current.node_id)
                break

        if not done:
            if task_id:
                self.journal.finish(task_id, "failed")
            self.emit("轮次用尽，收尾")

        u = self.kernel.usage
        self.emit(f"任务结束（{len(trace)} 步，累计 {u.total} tokens）")

    async def _page_fault(self, targets: list[str], guard, seen: set[str] | None = None) -> None:
        """缺页中断：把目标文件换入内存，LRU 自动踢掉最老的页。"""

        seen = seen if seen is not None else set()
        for target in targets[:3]:
            key = target.strip().lower()
            if key in seen:
                self.emit(f"[缺页] {target} 本轮已寻过，跳过（换个文件或直接动手）", cls="err")
                continue
            seen.add(key)

            looks_like_path = ("." in target or "/" in target) and " " not in target
            active = [p for p in self.mmu._pages.values()
                      if p.source == target and p.state.value == "active" and not p.stale]
            if active:
                self.emit(f"[缺页] {target} 已在内存（页 #{active[0].page_id}），不重复换入", cls="sys")
                continue
            obs = self.executor.read_internal(target)
            if not obs.ok and looks_like_path:
                hits = []
                self.emit(f"[缺页] 文件不存在：{target}（不猜，换目标或动手）", cls="err")
                continue
            if not obs.ok:
                hits = self.memindex.query(target, k=2)
                hits = [h for h in hits if h.score > 0.35]
                if hits:
                    for h in hits[:1]:
                        page = self.mmu.add(h.text[:1500], vec=self._vec(h.text[:800]),
                                            importance=0.9, tag="pagefault", source=h.path)
                        self.emit(f"[缺页中断] 语义寻址「{target}」→ {h.path}:{h.start} "
                                  f"(相似度 {h.score:.2f}) → 页 #{page.page_id}", cls="fault")
                    guard.observe("fault", {"path": target}, True)
                    continue
                self.emit(f"[缺页] 寻址失败：{target}", cls="err")
                continue
            page = self.mmu.add(obs.text[:1500], vec=self._vec(obs.text[:800]),
                                importance=0.95, tag="pagefault", source=target)
            guard.observe("fault", {"path": target}, True)
            self.emit(f"[缺页中断] 换入 {target} → 页 #{page.page_id}", cls="fault")

    def _is_doc_task(self, summary: str) -> bool:
        return ("readme" in summary.lower() or ".md" in summary.lower()) and ("写" in summary or "生成" in summary)

    async def _write_doc(self, summary: str, node, *, task_id: str | None = None) -> None:
        """两阶段：先定大纲钉住，再逐节填，最后合并落盘。"""

        outline_prompt = (
            f"{summary}\n\n项目结构：\n{self._tree()}\n\n"
            '只输出一个 JSON：{"sections": ["章节名1", "章节名2", ...]}，至少 5 节，中文，不要解释。'
        )
        sections: list[str] = []
        for attempt in range(2):
            raw = await self.kernel.chat(
                [{"role": "system", "content": "你是文档规划器，只输出 JSON。"},
                 {"role": "user", "content": outline_prompt}],
                max_tokens=300,
            )
            obj = protocol.parse(raw.text)
            if obj.name == "done" and isinstance(obj.args.get("summary"), str):
                pass
            try:
                import json as _json
                data = _json.loads(raw.text[raw.text.find("{"): raw.text.rfind("}") + 1])
                sections = [str(x) for x in data.get("sections", []) if str(x).strip()]
            except Exception:
                sections = []
            if len(sections) >= 4:
                break
            self.emit(f"[大纲] 只有 {len(sections)} 节，重来")

        if not sections:
            sections = ["项目定位", "整体架构", "核心模块", "运行方式", "目录说明"]
        self.emit(f"[大纲] {len(sections)} 节：{' / '.join(sections[:6])}")

        parts: list[str] = []
        for idx, sec in enumerate(sections, 1):
            tail = parts[-1][-300:] if parts else ""
            prompt = (
                f"【系统段】{protocol.SYSTEM}\n\n"
                f"【人类指令段】任务：{summary}\n"
                f"【状态段】大纲：{' / '.join(sections)}\n上文结尾：{tail}\n\n"
                f"现在只写第 {idx} 节「{sec}」，用 Markdown 二级标题开头，写满 400 字以上，"
                "给代码细节，不要空话，不要写其他章节。"
            )
            raw = await self.kernel.chat(
                [{"role": "system", "content": "你是技术文档作者，只输出该节正文。"},
                 {"role": "user", "content": prompt}],
                max_tokens=1200,
            )
            body = raw.text.strip()
            parts.append(body)
            n = self.dag.add(f"写 {sec}（{len(body)} 字符）", parent_id=node.node_id, kind="doc")
            node = n
            self._head = n
            if task_id:
                self.journal.checkpoint(task_id, node_id=n.node_id, round_no=idx, steps=idx)
            self.emit(f"[{idx}/{len(sections)}] {sec} → {len(body)} 字符")

        content = "# " + summary[:20].strip("。") + "\n\n" + "\n\n".join(parts) + "\n"
        target = self.executor.root / "README.md"
        target.write_text(content, encoding="utf-8")
        self.mmu.add(sanitize.strip_json(content)[:1200], importance=0.9,
                     tag="deliverable", segment=Segment.DAG_STATE)

        done_node = self.dag.add(f"README.md 完成（{len(content)} 字符）", parent_id=node.node_id, kind="done")
        self._head = done_node
        if task_id:
            self.journal.finish(task_id)
        self.dag.prune(done_node.node_id)
        u = self.kernel.usage
        self.emit(f"完成：README.md {len(content)} 字符 / {len(sections)} 节（累计 {u.total} tokens）")

    async def _checkpoint(self, node, job_id: str) -> None:
        self.dag.add(f"checkpoint {job_id}", parent_id=node.node_id, kind="checkpoint")
        self.emit(f"checkpoint：{job_id}")

    _tree_cache: str | None = None

    def _make_locks(self):
        try:
            from cluster.broker import Broker
            import redis.asyncio as aioredis
            client = aioredis.from_url("redis://127.0.0.1:6379/0", decode_responses=True)
            return PathLocks(redis_client=client, lock_dir=self.home / "locks", owner=self.home.name)
        except Exception:
            return PathLocks(lock_dir=self.home / "locks", owner=self.home.name)

    def snapshot(self) -> dict:
        nodes = sorted(self.dag._nodes.values(), key=lambda n: n.node_id)[-40:]
        order = {"system": 0, "human": 1, "dag": 2, "data": 3}
        pages = sorted((p for p in self.mmu._pages.values() if p.state.value == "active"),
                       key=lambda p: (order.get(p.segment.value, 9), -p.page_id))[:30]
        return {
            "dag": [{"id": n.node_id, "pid": n.parent_id, "kind": n.kind,
                     "status": n.status.value, "text": n.content[:80]} for n in nodes],
            "pages": [{"id": p.page_id, "preview": p.content[:60].replace("\n", " "),
                       "source": p.source, "stale": p.stale, "pinned": p.pinned,
                       "state": p.state.value, "segment": p.segment.value}
                      for p in pages],
            "log": self.log[-120:],
            "tokens": self.kernel.usage.total,
            "running": len(self.sched.stats()["running"]),
            "workers": getattr(self, "_workers", 0),
            "locks": self.path_locks.stats(),
        }

    async def watch_workers(self) -> None:
        """后台刷新集群 worker 数量，供控制台显示。"""

        from cluster.broker import Broker
        self.broker = Broker()
        while True:
            try:
                self._workers = len(await self.broker.workers())
            except Exception:  # noqa: BLE001
                self._workers = 0
            await asyncio.sleep(10)

    async def inject(self, text: str) -> None:
        self.emit(f"> {text}", cls="sys")
        await self._handle_input(text)

    def _on_file_written(self, rel: str) -> None:
        """写文件 → 依赖它的页面失效，避免拿旧内容回答。"""

        n = self.mmu.invalidate_source(rel)
        affected = self.depgraph.affected_by(rel)
        for item in affected:
            n += self.mmu.invalidate_source(item)
        merged = self.mmu.compact(tag="interrupt", keep=1)
        if merged:
            self.emit(f"[合并] 闲聊页压缩成摘要 #${merged.page_id}")
        if n:
            self.emit(f"[依赖失效] {rel} 改动，作废 {n} 页（波及 {len(affected)} 个文件）")
        self.depgraph.refresh()

    def _tree(self, limit: int = 60) -> str:
        if self._tree_cache is None:
            rows: list[str] = []
            root = self.home / "workspace"
            for child in sorted(root.rglob("*")):
                if len(rows) >= limit:
                    rows.append("...")
                    break
                rel = child.relative_to(root)
                rows.append(("d " if child.is_dir() else "f ") + str(rel))
            self._tree_cache = "\n".join(rows) or "（空）"
        return self._tree_cache

    def _vec(self, text: str) -> list[float]:
        return self.encoder.encode_one(text)

    # ---------- 退出 ----------

    async def shutdown(self) -> None:
        self._running = False
        await self.sched.stop()
        self.mmu.close()
        self.dag.close()
        self.journal.close()
        await self.mcp.stop_all()
        self.emit(f"bye: {self.mmu.stats()} {self.dag.stats()}")


async def main() -> None:
    import sys as _sys
    aos = AOS()
    await aos.boot()

    if "--web" in _sys.argv:
        from web.server import serve
        port = 8799
        for i, arg in enumerate(_sys.argv):
            if arg == "--port" and i + 1 < len(_sys.argv):
                port = int(_sys.argv[i + 1])
        events = asyncio.create_task(aos.event_loop())
        asyncio.create_task(aos.watch_workers())
        try:
            await serve(aos, "0.0.0.0", port)
        finally:
            events.cancel()
            await aos.shutdown()
        return

    terminal = asyncio.create_task(aos.terminal())
    events = asyncio.create_task(aos.event_loop())
    await terminal
    await aos.spooler.flush()
    deadline = time.time() + float(os.environ.get("AOS_DRAIN", "180"))
    while time.time() < deadline:
        if aos.spooler.stats["queued"] == 0 and not aos.sched.stats()["running"]:
            break
        await asyncio.sleep(0.05)
    events.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await events
    await aos.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
