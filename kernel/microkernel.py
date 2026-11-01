"""A-OS 微内核：进程表 + 调度 + 消息总线 + 服务装配。

内核只做机制：进程、调度、IPC、缺页陷阱（后续）。
pager / librarian 是内核外的服务。
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from drivers.embedding import load_encoder
from drivers.model import DeepSeekKernel, OfflineKernel
from kernel.ipc import MessageBus
from kernel.journal import Journal
from kernel.kernel_scheduler import Job, KernelScheduler
from kernel.process import ProcState, Process, ProcessTable
from kernel.spool import Spool
from kernel.state_dag import StateDAG
from services.librarian import LibrarianService
from services.pager import PagerService
from services.vfs import VFSService

Routine = Callable[[Process, "Microkernel"], Awaitable[Any]]


class Microkernel:
    def __init__(self, home: str | Path = "/opt/aos/data", *, model=None,
                 encoder=None) -> None:
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder or load_encoder()
        if model is not None:
            self.model = model
        elif os.environ.get("AOS_MODEL", "auto") == "offline":
            self.model = OfflineKernel()
        else:
            self.model = DeepSeekKernel() if DeepSeekKernel().ready else OfflineKernel()

        self.bus = MessageBus()
        self.table = ProcessTable()
        self.sched = KernelScheduler(max_workers=3)
        self.pager = PagerService(self.home, self.encoder)
        self.librarian = LibrarianService(self.pager.graph, model=self.model)
        self.vfs = VFSService(self.home, self.encoder)
        self.spool = Spool(self.home)
        self._last: dict = {}
        self._last_pid: int | None = None
        self.dag = StateDAG(self.home / "dag.db")
        self.journal = Journal(self.home / "tasks.db")
        self.budget = int(os.environ.get("AOS_CTX_BUDGET", "8000"))
        self._state = "booting"
        self.started_at = time.time()

    # ---------- 启动 ----------

    async def boot(self) -> None:
        self.bus.register("kernel", self)
        self.bus.register("pager", self.pager)
        self.bus.register("librarian", self.librarian)
        self.bus.register("vfs", self.vfs)
        await self.bus.start()
        await self.pager.boot()
        await self.librarian.start()
        await self.vfs.boot()
        await self.sched.start()
        self._state = "running"

    async def close(self) -> None:
        self._state = "stopping"
        await self.librarian.close()
        await self.sched.stop()
        await self.bus.stop()
        await self.vfs.close()
        await self.pager.close()
        self.dag.close()
        self.journal.close()
        self._state = "stopped"

    # ---------- 进程 ----------

    async def spawn(self, name: str, routine: Routine | None = None, *,
                    parent: int | None = None, priority: int = 0,
                    payload: dict | None = None) -> Process:
        proc = self.table.new(name, parent=parent, priority=priority, payload=payload)

        async def run() -> None:
            self.table.set_state(proc.pid, ProcState.RUNNING)
            try:
                result = await routine(proc, self) if routine else proc.payload
                self.table.set_state(proc.pid, ProcState.DONE, result=result)
            except asyncio.CancelledError:
                self.table.set_state(proc.pid, ProcState.KILLED, error="cancelled")
                raise
            except Exception as exc:  # noqa: BLE001
                self.table.set_state(proc.pid, ProcState.KILLED, error=f"{type(exc).__name__}: {exc}")

        async def on_cancel(_jid: str) -> None:
            self.table.set_state(proc.pid, ProcState.KILLED, error="preempted")

        self.sched.submit(Job(priority=priority, seq=0, name=name, run=run,
                              on_cancel=on_cancel))
        return proc

    async def wait(self, pid: int, *, timeout: float = 60.0) -> Any:
        proc = self.table.get(pid)
        if proc is None:
            raise LookupError(f"没有进程 {pid}")
        return await asyncio.wait_for(asyncio.shield(proc.future), timeout)

    async def run(self, name: str, routine: Routine, *, timeout: float = 60.0) -> Any:
        proc = await self.spawn(name, routine)
        return await self.wait(proc.pid, timeout=timeout)

    async def ask(self, text: str, *, channel: str = "cli", timeout: float = 90.0) -> str:
        """一个用户请求 = 一个 agent 进程。"""

        from kernel.agent_proc import AgentProcess

        text = (text or "").strip()
        if not text:
            return "在呢，你说。"
        self.spool.input(text, channel=channel)

        async def routine(proc, mk):
            return await AgentProcess(mk).run(proc)

        started = time.time()
        proc = await self.spawn("chat", routine, payload={"text": text, "channel": channel})
        answer = await self.wait(proc.pid, timeout=timeout)
        self.spool.output(pid=proc.pid, channel=channel, text=answer)
        self._last_pid = proc.pid
        self._last = {
            "elapsed": round(time.time() - started, 2),
            "process": proc.pid,
            "answer": answer[:120],
        }
        return answer

    async def reply(self, text: str) -> str:
        return await self.ask(text)

    @property
    def memory(self):
        return self.pager.memory

    def history(self, limit: int = 100) -> list[dict]:
        return [{"role": t.role, "text": t.text, "ts": t.ts}
                for t in self.pager.dialogue.history(limit=limit)]

    # ---------- 内核端点 ----------

    async def handle(self, method: str, params: dict) -> Any:
        if method == "state":
            return self.debug_snapshot()
        if method == "processes":
            return self.table.list(limit=int(params.get("limit", 30)))
        if method == "stats":
            return {
                "state": self._state,
                "uptime": round(time.time() - self.started_at, 1),
                "processes": self.table.stats(),
                "scheduler": self.sched.stats(),
                "bus": self.bus.stats(),
            }
        if method == "fault":
            return await self._fault(params)
        if method == "compose":
            return await self.bus.call("pager", "compose", **params)
        if method == "state_add":
            pid = int(params.get("pid", 0))
            head = self.dag.head(pid)
            node = self.dag.add(
                str(params.get("content", ""))[:200],
                parent_id=head.node_id if head else None,
                kind=str(params.get("kind", "step")), pid=pid,
            )
            return {"id": node.node_id}
        if method == "state_render":
            return {"text": self.dag.render(limit=int(params.get("limit", 5)),
                                            pid=int(params.get("pid", 0)))}
        if method == "state_prune":
            self.dag.prune(int(params["node_id"]), pid=int(params.get("pid", 0)))
            return {"ok": True}
        if method == "state_stats":
            return self.dag.stats()
        if method == "journal_register":
            task_id = self.journal.register(str(params.get("text", "")),
                                            int(params.get("node_id", 0)))
            self.journal.claim(task_id)
            return {"task_id": task_id}
        if method == "journal_checkpoint":
            self.journal.checkpoint(
                str(params["task_id"]),
                node_id=int(params.get("node_id", 0)),
                round_no=int(params.get("round", 0)),
                steps=int(params.get("steps", 0)),
            )
            return {"ok": True}
        if method == "journal_finish":
            self.journal.finish(str(params["task_id"]), str(params.get("status", "done")))
            return {"ok": True}
        if method == "ping":
            return "pong"
        raise ValueError(f"kernel 没有方法：{method}")

    async def _fault(self, params: dict) -> dict:
        pid = int(params["pid"])
        proc = self.table.get(pid)
        if proc is not None:
            self.table.set_state(pid, ProcState.WAITING)
        try:
            return await self.bus.call(
                "pager", "fault", pid=pid,
                targets=list(params.get("targets") or []),
                hot=list(params.get("hot") or []),
            )
        finally:
            if proc is not None:
                self.table.set_state(pid, ProcState.RUNNING)

    def debug_snapshot(self) -> dict:
        return {
            "state": self._state,
            "uptime": round(time.time() - self.started_at, 1),
            "processes": self.table.list(),
            "process_stats": self.table.stats(),
            "scheduler": self.sched.stats(),
            "bus": self.bus.stats(),
            "services": self.bus.endpoints(),
            "pager": self.pager.graph.stats(),
            "pages": self.pager.mmu.stats(),
            "memory": self.pager.memory.stats(),
            "dag": self.dag.stats(),
            "journal": self.journal.stats(),
            "librarian": self.librarian.stats(),
            "vfs": self.vfs.stats(),
            "spool": self.spool.stats(),
            "memories": self.pager.memory.stats(),
            "turns": self.pager.dialogue.count(),
            "model": "online" if getattr(self.model, "ready", False) else "offline",
            "last": self._last,
            "tools": self.vfs.stats().get("tools", []),
            "events": [],
            "kernel": {
                "nodes": self.dag.stats(),
                "pages": self.pager.mmu.stats(),
                "scheduler": self.sched.stats(),
                "journal": self.journal.stats(),
                "faults": self.pager.faults,
                "steps": self._last.get("steps", 0),
                "path": self.dag.render(limit=6, pid=self._last_pid)
                if self._last_pid is not None else "",
            },
        }
