"""进程表：OS 味道的 ready/running/waiting/done/killed。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum


class ProcState(str, Enum):
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    DONE = "done"
    KILLED = "killed"


@dataclass
class Process:
    pid: int
    name: str
    state: ProcState = ProcState.READY
    parent: int | None = None
    priority: int = 0
    budget: int = 8000
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    payload: dict = field(default_factory=dict)
    result: object = None
    error: str | None = None
    future: asyncio.Future = field(default=None)  # type: ignore[assignment]

    @property
    def elapsed(self) -> float:
        end = self.finished or time.time()
        return round(end - (self.started or self.created), 3)


class ProcessTable:
    def __init__(self) -> None:
        self._procs: dict[int, Process] = {}
        self._next = 1

    def new(self, name: str, *, parent: int | None = None, priority: int = 0,
            budget: int = 8000, payload: dict | None = None) -> Process:
        loop = asyncio.get_running_loop()
        proc = Process(
            pid=self._next, name=name, parent=parent, priority=priority,
            budget=budget, payload=payload or {}, future=loop.create_future(),
        )
        self._next += 1
        self._procs[proc.pid] = proc
        return proc

    def get(self, pid: int) -> Process | None:
        return self._procs.get(pid)

    def set_state(self, pid: int, state: ProcState, *,
                  result: object = None, error: str | None = None) -> None:
        proc = self._procs.get(pid)
        if proc is None:
            return
        now = time.time()
        proc.state = state
        if state is ProcState.RUNNING and proc.started is None:
            proc.started = now
        if state in (ProcState.DONE, ProcState.KILLED):
            proc.finished = now
            proc.result = result
            proc.error = error
            if not proc.future.done():
                if error:
                    proc.future.set_exception(RuntimeError(error))
                else:
                    proc.future.set_result(result)

    def list(self, *, limit: int = 30) -> list[dict]:
        procs = sorted(self._procs.values(), key=lambda p: p.pid, reverse=True)[:limit]
        return [
            {"pid": p.pid, "name": p.name, "state": p.state.value,
             "parent": p.parent, "priority": p.priority,
             "role": p.payload.get("role", ""),
             "elapsed": p.elapsed, "error": p.error}
            for p in procs
        ]

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for proc in self._procs.values():
            counts[proc.state.value] = counts.get(proc.state.value, 0) + 1
        counts["total"] = len(self._procs)
        return counts
