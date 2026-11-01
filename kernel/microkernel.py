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
from kernel.kernel_scheduler import Job, KernelScheduler
from kernel.process import ProcState, Process, ProcessTable
from services.librarian import LibrarianService
from services.pager import PagerService

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
        self._state = "booting"
        self.started_at = time.time()

    # ---------- 启动 ----------

    async def boot(self) -> None:
        self.bus.register("kernel", self)
        self.bus.register("pager", self.pager)
        self.bus.register("librarian", self.librarian)
        await self.bus.start()
        await self.pager.boot()
        await self.librarian.start()
        await self.sched.start()
        self._state = "running"

    async def close(self) -> None:
        self._state = "stopping"
        await self.librarian.close()
        await self.sched.stop()
        await self.bus.stop()
        await self.pager.close()
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

    async def ask(self, text: str, *, timeout: float = 90.0) -> str:
        """一个用户请求 = 一个 agent 进程。"""

        from kernel.agent_proc import AgentProcess

        async def routine(proc, mk):
            return await AgentProcess(mk).run(proc)

        proc = await self.spawn("chat", routine, payload={"text": text})
        return await self.wait(proc.pid, timeout=timeout)

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
        if method == "ping":
            return "pong"
        raise ValueError(f"kernel 没有方法：{method}")

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
            "librarian": self.librarian.stats(),
        }
