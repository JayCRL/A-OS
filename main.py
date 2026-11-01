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
from kernel.page_table import PageTable
from kernel.state_dag import NodeStatus, StateDAG
from drivers.model import DeepSeekKernel, OfflineKernel
from drivers import protocol
from drivers.embedding import load_encoder
from kernel.executor import Executor
from kernel.shell_sandbox import ShellSandbox
from kernel.depgraph import DepGraph
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
        self.executor = Executor(self.home / "workspace", allow_shell=True,
                                 sandbox=self.sandbox, mcp=self.mcp,
                                 on_write=self._on_file_written)
        self.max_rounds = int(os.environ.get("AOS_MAX_ROUNDS", "6"))
        self.encoder = load_encoder()
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
        self._migrate_vectors()
        self._recover()
        self.emit(f"boot: 节点 {self._head.node_id}，页 {self.mmu.stats()['pages']} 条，"
                  f"模型 {'在线' if self.kernel.ready else '离线'}，编码器 {self.encoder.name}({self.encoder.dim}d)")

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
        page = self.mmu.add(f"人类指令：{event.summary}", importance=1.0, tag="interrupt", pinned=True)
        self.mmu.unpin(page.page_id)  # 只在本轮钉住

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
            header = (f"任务目标：{summary}\n"
                      f"文件树（已缓存，别 list）：\n{self._tree()}\n"
                      f"依赖（谁 import 谁）：\n{self.depgraph.render(limit=8)}\n"
                      f"台账：{guard.ledger()}")
            prompt, pages = self.mmu.compose(
                query_vec=self._vec(summary),
                budget_tokens=self.budget,
                header=header,
            )
            if trace:
                prompt += "\n\n最近两步：\n" + "\n".join(trace[-2:])

            action = None
            out_tokens = int(os.environ.get("AOS_OUT_TOKENS", "3000"))
            for attempt in range(2):
                raw = await self.kernel.chat(
                    [
                        {"role": "system", "content": protocol.SYSTEM},
                        {"role": "user", "content": prompt + (
                            "\n[系统] 上一次输出被截断或不是合法 JSON。这次要么用 append 分块写，"
                            "要么只输出一个短 JSON。" if attempt else "")},
                    ],
                    max_tokens=out_tokens if attempt == 0 else 1200,
                )
                action = protocol.parse(raw.text)
                if not (action.name == "done" and str(action.args.get("summary", "")).startswith("无法解析")):
                    break
                salvaged = protocol.salvage(raw.text)
                if salvaged is not None:
                    self.emit("[续写] 输出被截断，已追加，下一轮从断点继续")
                    action = salvaged
                    break
                self.emit(f"[重试] 输出不合法，第 {attempt + 2} 次")
            obs = await self.executor.run(action.name, action.args)
            guard.observe(action.name, action.args, obs.ok)

            if action.name == "read" and obs.ok:
                src = str(action.args.get("path", ""))
                self.mmu.add(f"{src}\n{obs.text}"[:1500], vec=self._vec(obs.text[:800]),
                             importance=0.6, tag="file", source=src)
            trace.append(f"#{rnd} {action.name} -> {str(obs)[:300]}")

            nudge = guard.intervention()
            if nudge:
                trace.append(nudge)
                self.emit(f"[看门狗] {nudge[:60]}")

            step_node = self.dag.add(
                f"{action.name}: {str(obs)[:200]}", parent_id=current.node_id, kind="act"
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
                f"任务：{summary}\n"
                f"大纲：{' / '.join(sections)}\n"
                f"上文结尾：{tail}\n\n"
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
        self.mmu.add(content[:1200], importance=0.9, tag="deliverable")

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

    def snapshot(self) -> dict:
        nodes = sorted(self.dag._nodes.values(), key=lambda n: n.node_id)[-40:]
        pages = sorted(self.mmu._pages.values(), key=lambda p: p.page_id)[-30:]
        return {
            "dag": [{"id": n.node_id, "kind": n.kind, "status": n.status.value,
                     "text": n.content[:80]} for n in nodes],
            "pages": [{"id": p.page_id, "preview": p.content[:60].replace("\n", " "),
                       "source": p.source, "stale": p.stale, "pinned": p.pinned}
                      for p in pages],
            "log": self.log[-120:],
            "tokens": self.kernel.usage.total,
            "running": len(self.sched.stats()["running"]),
            "workers": getattr(self, "_workers", 0),
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
