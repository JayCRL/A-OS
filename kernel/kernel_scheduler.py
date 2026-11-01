"""A-OS 调度内核。

优先级队列 + 协程取消 + 文件互斥锁。
没有时间片；抢占只是「取消当前协程 + 落 checkpoint」，不能回滚副作用。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable


@dataclass(order=True)
class Job:
    priority: int
    seq: int
    job_id: str = field(compare=False, default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = field(compare=False, default="job")
    run: Callable[[], Awaitable[None]] = field(compare=False, default=None)
    budget_tokens: int = field(compare=False, default=8000)
    deadline_s: float = field(compare=False, default=600.0)
    on_cancel: Callable[[str], Awaitable[None]] | None = field(compare=False, default=None)


class FileLockTable:
    """路径级互斥。同一路径同时只能有一个写者。"""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._holders: dict[str, str] = {}

    @contextlib.asynccontextmanager
    async def acquire(self, path: str, owner: str, timeout: float = 30.0):
        lock = self._locks.setdefault(path, asyncio.Lock())
        acquired = False
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
            acquired = True
            self._holders[path] = owner
            yield
        except asyncio.TimeoutError:
            raise TimeoutError(f"{path} 被 {self._holders.get(path)} 占用") from None
        finally:
            if acquired:
                self._holders.pop(path, None)
                lock.release()

    def holder(self, path: str) -> str | None:
        return self._holders.get(path)


class KernelScheduler:
    def __init__(self, *, max_workers: int = 4) -> None:
        self.max_workers = max_workers
        self._ready: asyncio.PriorityQueue[Job] = asyncio.PriorityQueue()
        self._seq = 0
        self._workers: dict[str, asyncio.Task] = {}
        self._running: dict[str, Job] = {}
        self.locks = FileLockTable()
        self._stopped = False
        self._loop_task: asyncio.Task | None = None

    # ---------- 生命周期 ----------

    async def start(self) -> None:
        self._loop_task = asyncio.create_task(self._loop(), name="aos-scheduler")

    async def stop(self) -> None:
        self._stopped = True
        for task in list(self._workers.values()):
            task.cancel()
        await asyncio.gather(*self._workers.values(), return_exceptions=True)
        if self._loop_task:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task

    # ---------- 入队 ----------

    def submit(self, job: Job) -> str:
        job.seq = self._seq
        self._seq += 1
        self._ready.put_nowait(job)
        return job.job_id

    def submit_human(self, summary: str, run: Callable[[], Awaitable[None]]) -> str:
        return self.submit(Job(priority=0, seq=0, name=f"human:{summary}", run=run,
                               budget_tokens=4000, deadline_s=120.0))

    # ---------- 抢占 ----------

    async def preempt(self, reason: str = "human_interrupt") -> list[str]:
        """取消所有 > 0 优先级的运行中协程，等它们落完 checkpoint。"""

        victims = [jid for jid, job in self._running.items() if job.priority > 0]
        tasks = [self._workers[jid] for jid in victims if jid in self._workers]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return victims

    # ---------- 主循环 ----------

    async def _loop(self) -> None:
        while not self._stopped:
            if len(self._workers) >= self.max_workers:
                await asyncio.sleep(0.01)
                continue
            try:
                job = await asyncio.wait_for(self._ready.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            task = asyncio.create_task(self._run(job), name=f"worker:{job.name}")
            self._workers[job.job_id] = task
            self._running[job.job_id] = job
            task.add_done_callback(lambda t, jid=job.job_id: self._reap(jid))

    def _reap(self, job_id: str) -> None:
        self._workers.pop(job_id, None)
        self._running.pop(job_id, None)

    async def _run(self, job: Job) -> None:
        started = time.time()
        try:
            await job.run()
        except asyncio.CancelledError:
            if job.on_cancel:
                with contextlib.suppress(Exception):
                    await job.on_cancel(job.job_id)
            raise
        finally:
            self._last_elapsed = time.time() - started

    # ---------- 观察 ----------

    def stats(self) -> dict[str, object]:
        return {
            "queued": self._ready.qsize(),
            "running": list(self._running),
            "workers": len(self._workers),
            "locks": {p: h for p, h in self.locks._holders.items()},
        }
