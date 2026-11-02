"""独立 Watchdog：先做基础设施兜底，模型级进度判断后续再接。"""

from __future__ import annotations

import asyncio
import time


class Watchdog:
    def __init__(self, kernel, *, interval: float = 10.0,
                 trigger_timeout: float = 300.0) -> None:
        self.k = kernel
        self.interval = max(1.0, float(interval))
        self.trigger_timeout = max(10.0, float(trigger_timeout))
        self.checks = 0
        self.failed_triggers = 0
        self.stalled_workers = 0
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="kernel:watchdog")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        while self.k._state in ("booting", "running"):
            await asyncio.sleep(self.interval)
            await self.tick()

    async def tick(self) -> dict:
        self.checks += 1
        now = time.time()
        for trig in self.k.triggers.list(status="running", limit=100):
            if now - trig.updated_at > self.trigger_timeout:
                self.k.triggers.finish(
                    trig.id, status="failed",
                    result=f"watchdog: trigger 超过 {self.trigger_timeout:.0f}s 未完成",
                )
                self.failed_triggers += 1
        for proc in self.k.table._procs.values():
            if proc.state.value not in ("running", "ready"):
                continue
            if proc.payload.get("role") != "worker" or proc.payload.get("stalled"):
                continue
            if proc.elapsed > self.trigger_timeout:
                self.stalled_workers += 1
                proc.payload["stalled"] = True
        return self.stats()

    def stats(self) -> dict:
        return {
            "checks": self.checks,
            "failed_triggers": self.failed_triggers,
            "stalled_workers": self.stalled_workers,
            "interval": self.interval,
            "trigger_timeout": self.trigger_timeout,
        }
