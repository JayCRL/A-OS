"""A-OS 微内核：进程表 + 调度 + 消息总线 + 服务装配。

内核只做机制：进程、调度、IPC、缺页陷阱（后续）。
pager / librarian 是内核外的服务。
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from drivers.embedding import load_encoder
from drivers.model import DeepSeekKernel, OfflineKernel
from kernel.dispatcher import Dispatcher
from kernel.ipc import MessageBus
from kernel.journal import Journal
from kernel.kernel_scheduler import Job, KernelScheduler
from kernel.process import ProcState, Process, ProcessTable
from kernel.spool import Spool
from kernel.state_dag import StateDAG
from kernel.unix_ipc import RemoteBus, UnixClient
from services.librarian import LibrarianService
from services.pager import PagerService
from services.vfs import VFSService

Routine = Callable[[Process, "Microkernel"], Awaitable[Any]]


class Microkernel:
    def __init__(self, home: str | Path = "/opt/aos/data", *, model=None,
                 encoder=None, split: bool = False) -> None:
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.encoder = encoder if split else (encoder or load_encoder())
        if model is not None:
            self.model = model
        elif os.environ.get("AOS_MODEL", "auto") == "offline":
            self.model = OfflineKernel()
        else:
            self.model = DeepSeekKernel() if DeepSeekKernel().ready else OfflineKernel()

        self.table = ProcessTable()
        self.sched = KernelScheduler(max_workers=8)
        self.dispatcher = Dispatcher()
        self.spool = Spool(self.home)
        self._workers: list[asyncio.subprocess.Process] = []
        self._clients: dict[str, UnixClient] = {}
        self._history_cache: list[dict] = []
        self._pager_stats: dict = {"pages": {}, "graph": {}, "memory": {}, "faults": 0}
        self._log_handles: list = []
        self._watch_task: asyncio.Task | None = None
        self._session_tasks: dict[int, asyncio.Task] = {}
        self._turn_sem = asyncio.Semaphore(3)
        if split:
            self.bus = None  # boot 时换成 RemoteBus
            self.pager = None
            self.librarian = None
            self.vfs = None
        else:
            self.bus = MessageBus()
            self.pager = PagerService(self.home, self.encoder, bus=None)
            self.librarian = LibrarianService(
                self.pager.graph, model=self.model, fact_sink=self.pager.commit_memory,
            )
            self.vfs = VFSService(self.home, self.encoder)
        self._last: dict = {}
        self._last_pid: int | None = None
        self.dag = StateDAG(self.home / "dag.db")
        self.journal = Journal(self.home / "tasks.db")
        self.budget = int(os.environ.get("AOS_CTX_BUDGET", "8000"))
        self._state = "booting"
        self.started_at = time.time()

    # ---------- 启动 ----------

    async def boot(self) -> None:
        if self.split:
            await self._boot_split()
            await self.sched.start()
            self._state = "running"
            return
        self.bus.register("kernel", self)
        self.bus.register("pager", self.pager)
        self.bus.register("librarian", self.librarian)
        self.bus.register("vfs", self.vfs)
        await self.bus.start()
        self.pager.bus = self.bus
        await self.pager.boot()
        await self.librarian.start()
        await self.vfs.boot()
        await self.sched.start()
        self._state = "running"

    async def _boot_split(self) -> None:
        ipc = self.home / "ipc"
        ipc.mkdir(parents=True, exist_ok=True)
        root = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        socks = {name: str(ipc / f"{name}.sock") for name in ("vfs", "pager", "librarian")}

        async def start_worker(name: str, extra: list[str]) -> None:
            cmd = [sys.executable, "-m", "services.worker", name,
                   "--home", str(self.home), "--sock", socks[name], *extra]
            log_path = self.home / "logs" / f"worker-{name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("ab")
            self._log_handles.append(handle)
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(root), env=env, stdout=handle, stderr=handle,
            )
            self._workers.append(proc)
            client = UnixClient(socks[name])
            await client.wait_ready(timeout=30)
            self._clients[name] = client

        await start_worker("vfs", [])
        await start_worker("pager", ["--vfs-sock", socks["vfs"]])
        await start_worker("librarian", ["--pager-sock", socks["pager"]])

        self.bus = RemoteBus(self.handle)
        for name, client in self._clients.items():
            self.bus.register(name, client)
        try:
            self._pager_stats = await self.bus.call("pager", "stats", timeout=10)
            self._history_cache = await self.bus.call("pager", "history", limit=200)
        except Exception:
            pass
        self._watch_task = asyncio.create_task(self._watch_stats(), name="kernel:stats")

    async def _watch_stats(self) -> None:
        while self._state in ("booting", "running"):
            try:
                self._pager_stats = await self.bus.call("pager", "stats", timeout=3)
            except Exception:
                pass
            await asyncio.sleep(3)

    async def close(self) -> None:
        self._state = "stopping"
        for session in list(self.dispatcher.sessions.values()):
            await session.queue.put(None)
        for task in list(self._session_tasks.values()):
            task.cancel()
        if self._session_tasks:
            await asyncio.gather(*list(self._session_tasks.values()), return_exceptions=True)
        if self.split:
            try:
                await self.bus.call("librarian", "flush", timeout=10)
            except Exception:
                pass
            if self._watch_task:
                self._watch_task.cancel()
                try:
                    await self._watch_task
                except asyncio.CancelledError:
                    pass
                self._watch_task = None
            await self.sched.stop()
            await self.bus.stop()
            await self._stop_workers()
        else:
            await self.librarian.close()
            await self.sched.stop()
            await self.bus.stop()
            await self.vfs.close()
            await self.pager.close()
        self.dag.close()
        self.journal.close()
        self._state = "stopped"

    async def _stop_workers(self) -> None:
        for proc in self._workers:
            if proc.returncode is None:
                proc.terminate()
        for proc in self._workers:
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
        self._workers.clear()
        for handle in self._log_handles:
            try:
                handle.close()
            except Exception:
                pass
        self._log_handles.clear()

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

        for old in self.dispatcher.expired():
            if old.pid is not None:
                await old.queue.put(None)
            self.dispatcher.drop(old)

        session, is_new = self.dispatcher.route(text, channel=channel)
        proc = self.table.get(session.pid) if session.pid is not None else None
        if is_new or proc is None or proc.state in (ProcState.DONE, ProcState.KILLED):
            proc = self.table.new("session", payload={"session": session})
            session.pid = proc.pid
            session.created = time.time()
            self._session_tasks[proc.pid] = asyncio.create_task(
                self._session_loop(proc), name=f"session:{proc.pid}"
            )

        started = time.time()
        self._history_cache.append({"role": "user", "text": text, "ts": time.time()})
        fut = asyncio.get_running_loop().create_future()
        await session.queue.put({"text": text, "future": fut})
        answer = await asyncio.wait_for(fut, timeout=timeout)
        self.dispatcher.touch(session, text, answer)
        self._history_cache.append({"role": "assistant", "text": answer, "ts": time.time()})
        del self._history_cache[:-200]
        self.spool.output(pid=proc.pid, channel=channel, text=answer)
        if self.split:
            try:
                self._pager_stats = await self.bus.call("pager", "stats", timeout=5)
            except Exception:
                self._pager_stats = {}
        self._last_pid = proc.pid
        self._last = {
            "elapsed": round(time.time() - started, 2),
            "process": proc.pid,
            "answer": answer[:120],
        }
        return answer

    async def _session_loop(self, proc: Process) -> None:
        from kernel.agent_proc import AgentProcess

        session = proc.payload["session"]
        agent = AgentProcess(self)
        try:
            while self._state in ("booting", "running"):
                self.table.set_state(proc.pid, ProcState.WAITING)
                item = await session.queue.get()
                if item is None:
                    break
                self.table.set_state(proc.pid, ProcState.RUNNING)
                future = item.get("future")
                try:
                    async with self._turn_sem:
                        answer = await agent.run_turn(proc, str(item.get("text", "")))
                    if future is not None and not future.done():
                        future.set_result(answer)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    if future is not None and not future.done():
                        future.set_exception(exc)
        finally:
            self.table.set_state(proc.pid, ProcState.DONE)
            self._session_tasks.pop(proc.pid, None)

    async def reply(self, text: str) -> str:
        return await self.ask(text, channel="web")

    @property
    def memory(self):
        if self.split:
            return _CachedMemory(self)
        return self.pager.memory

    def history(self, limit: int = 100) -> list[dict]:
        if self.split:
            return list(self._history_cache[-limit:])
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
        if self.split:
            pages = self._pager_stats.get("pages", {})
            graph = self._pager_stats.get("graph", {})
            memory = self._pager_stats.get("memory", {})
            return {
                "state": self._state,
                "uptime": round(time.time() - self.started_at, 1),
                "processes": self.table.list(),
                "process_stats": self.table.stats(),
                "scheduler": self.sched.stats(),
                "bus": self.bus.stats() if self.bus else {},
                "services": self.bus.endpoints() if self.bus else [],
                "dispatcher": self.dispatcher.stats(),
                "pager": graph,
                "pages": pages,
                "memory": memory,
                "dag": self.dag.stats(),
                "journal": self.journal.stats(),
                "librarian": {},
                "vfs": {},
                "spool": self.spool.stats(),
                "memories": memory,
                "turns": len(self._history_cache) // 2,
                "model": "online" if getattr(self.model, "ready", False) else "offline",
                "last": self._last,
                "tools": [],
                "events": [],
                "kernel": {
                    "nodes": self.dag.stats(),
                    "pages": pages,
                    "scheduler": self.sched.stats(),
                    "journal": self.journal.stats(),
                    "faults": self._pager_stats.get("faults", 0),
                    "steps": 0,
                    "path": self.dag.render(limit=6, pid=self._last_pid)
                    if self._last_pid is not None else "",
                },
            }
        return {
            "state": self._state,
            "uptime": round(time.time() - self.started_at, 1),
            "processes": self.table.list(),
            "process_stats": self.table.stats(),
            "scheduler": self.sched.stats(),
            "bus": self.bus.stats(),
            "services": self.bus.endpoints(),
            "dispatcher": self.dispatcher.stats(),
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


class _CachedMemory:
    def __init__(self, kernel) -> None:
        self.kernel = kernel

    def stats(self) -> dict:
        return self.kernel._pager_stats.get("memory", {"total": 0, "by_kind": {}})
