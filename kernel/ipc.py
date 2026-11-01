"""微内核消息总线。

服务之间不直接互相 import 调用，只通过命名端点和请求/响应消息。
同进程内是 asyncio 队列；以后拆成独立进程时，只需要换传输层。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

Handler = Callable[[str, dict], Awaitable[Any]]


@dataclass
class Message:
    id: str
    src: str
    dst: str
    method: str
    params: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class _Endpoint:
    name: str
    handler: Handler | Any
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: asyncio.Task | None = None
    processed: int = 0
    errors: int = 0

    async def invoke(self, method: str, params: dict) -> Any:
        fn = getattr(self.handler, "handle", None)
        if fn is not None:
            return await fn(method, params)
        return await self.handler(method, params)


class MessageBus:
    def __init__(self) -> None:
        self._endpoints: dict[str, _Endpoint] = {}
        self._started = False
        self._sent = 0
        self._failed = 0

    def register(self, name: str, handler: Handler | Any) -> None:
        if name in self._endpoints:
            raise ValueError(f"端点已存在：{name}")
        self._endpoints[name] = _Endpoint(name=name, handler=handler)

    def endpoints(self) -> list[str]:
        return sorted(self._endpoints)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for endpoint in self._endpoints.values():
            endpoint.task = asyncio.create_task(self._serve(endpoint), name=f"svc:{endpoint.name}")

    async def call(self, dst: str, method: str, *, src: str = "client",
                   timeout: float = 30.0, **params: Any) -> Any:
        endpoint = self._endpoints.get(dst)
        if endpoint is None:
            raise LookupError(f"没有这个服务：{dst}")
        if not self._started:
            raise RuntimeError("消息总线还没启动")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        msg = Message(id=uuid.uuid4().hex[:12], src=src, dst=dst, method=method, params=params)
        self._sent += 1
        await endpoint.queue.put((msg, fut))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._failed += 1
            raise TimeoutError(f"{dst}.{method} 超时") from None

    async def _serve(self, endpoint: _Endpoint) -> None:
        while True:
            msg, fut = await endpoint.queue.get()
            try:
                result = await endpoint.invoke(msg.method, msg.params)
                if not fut.done():
                    fut.set_result(result)
                endpoint.processed += 1
            except asyncio.CancelledError:
                if not fut.done():
                    fut.set_exception(asyncio.CancelledError())
                raise
            except Exception as exc:  # noqa: BLE001
                endpoint.errors += 1
                if not fut.done():
                    fut.set_exception(exc)
            finally:
                endpoint.queue.task_done()

    def stats(self) -> dict:
        return {
            "sent": self._sent,
            "failed": self._failed,
            "services": {
                name: {
                    "queued": ep.queue.qsize(),
                    "processed": ep.processed,
                    "errors": ep.errors,
                    "running": bool(ep.task and not ep.task.done()),
                }
                for name, ep in self._endpoints.items()
            },
        }

    async def stop(self) -> None:
        for endpoint in self._endpoints.values():
            while not endpoint.queue.empty():
                _msg, fut = endpoint.queue.get_nowait()
                if not fut.done():
                    fut.set_exception(RuntimeError("消息总线已停止"))
                endpoint.queue.task_done()
            if endpoint.task:
                endpoint.task.cancel()
        tasks = [ep.task for ep in self._endpoints.values() if ep.task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._started = False
