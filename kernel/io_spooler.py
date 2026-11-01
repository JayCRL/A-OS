"""A-OS 输入与 spooling 层。

人类的高频操作先攒着，2 秒内的动作合成一个中断事件，再交给调度器。
意图（要做什么）与内容（改了什么）分开处理：意图去重合并，内容留快照。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class ActionKind(str, Enum):
    INTENT = "intent"      # 我要做什么（合并成一句话）
    CONTENT = "content"    # 我改了什么（保留快照）
    SIGNAL = "signal"      # Ctrl-C / 强制提交


@dataclass
class HumanAction:
    kind: ActionKind
    text: str
    path: str = ""
    ts: float = field(default_factory=time.time)


@dataclass
class InterruptEvent:
    event_id: str
    actions: list[HumanAction]
    summary: str
    snapshot: list[dict]
    priority: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def is_force(self) -> bool:
        return any(a.kind is ActionKind.SIGNAL for a in self.actions)


class IOSpooler:
    """防抖合并器。空闲阈值内没有新动作就 flush。"""

    def __init__(self, *, idle_seconds: float = 2.0, max_wait_seconds: float = 8.0) -> None:
        self.idle_seconds = idle_seconds
        self.max_wait_seconds = max_wait_seconds
        self._buffer: list[HumanAction] = []
        self._first_at: float | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._queue: asyncio.Queue[InterruptEvent] = asyncio.Queue()
        self._flushes = 0

    async def feed(self, action: HumanAction) -> None:
        self._buffer.append(action)
        self._first_at = self._first_at or time.time()
        if self._timer:
            self._timer.cancel()

        waited = time.time() - (self._first_at or time.time())
        if action.kind is ActionKind.SIGNAL or waited >= self.max_wait_seconds:
            await self.flush()
            return

        loop = asyncio.get_running_loop()
        self._timer = loop.call_later(self.idle_seconds, lambda: asyncio.ensure_future(self._flush()))

    async def flush(self) -> None:
        await self._flush()

    async def _flush(self) -> None:
        if not self._buffer:
            return
        actions, self._buffer = self._buffer, []
        self._first_at = None
        if self._timer:
            self._timer.cancel()
            self._timer = None

        intents = [a.text for a in actions if a.kind is ActionKind.INTENT]
        snapshot = [
            {"path": a.path, "text": a.text, "ts": a.ts}
            for a in actions
            if a.kind is ActionKind.CONTENT
        ]
        force = any(a.kind is ActionKind.SIGNAL for a in actions)

        # 意图去重：连续相同的话只留一句
        merged: list[str] = []
        for text in intents:
            if not merged or merged[-1] != text:
                merged.append(text)
        summary = "；".join(merged) if merged else ("强制中断" if force else "内容更新")

        event = InterruptEvent(
            event_id=uuid.uuid4().hex[:12],
            actions=actions,
            summary=summary,
            snapshot=snapshot,
        )
        self._flushes += 1
        await self._queue.put(event)

    async def next_event(self) -> InterruptEvent:
        return await self._queue.get()

    @property
    def stats(self) -> dict[str, int]:
        return {"buffered": len(self._buffer), "flushes": self._flushes, "queued": self._queue.qsize()}
