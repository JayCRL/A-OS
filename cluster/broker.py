"""任务总线：Redis 队列 + 结果通道。

调度器只往队列里丢任务，Worker 抢到就干，结果走 Pub/Sub 回传。
没有 Redis 时退化成本地内存队列（仅单机调试用）。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

QUEUE = "aos:tasks"
RESULTS = "aos:results"
HEARTBEAT = "aos:workers"


@dataclass
class Task:
    goal: str
    workspace: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created_at: float = field(default_factory=time.time)
    budget: int = 8000

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "Task":
        data = json.loads(raw)
        return cls(**data)


class Broker:
    def __init__(self, url: str = "redis://127.0.0.1:6379/0") -> None:
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(url, decode_responses=True)
        self._fallback: list[str] = []

    @property
    def redis(self):
        return self._redis

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:  # noqa: BLE001
            return False

    # ---------- 任务 ----------

    async def submit(self, task: Task) -> str:
        try:
            await self._redis.lpush(QUEUE, task.to_json())
        except Exception:  # noqa: BLE001
            self._fallback.append(task.to_json())
        return task.task_id

    async def take(self, timeout: int = 5) -> Task | None:
        try:
            item = await self._redis.brpop(QUEUE, timeout=timeout)
        except Exception:  # noqa: BLE001
            item = ("", self._fallback.pop()) if self._fallback else None
        if not item:
            return None
        return Task.from_json(item[1])

    async def depth(self) -> int:
        try:
            return int(await self._redis.llen(QUEUE))
        except Exception:  # noqa: BLE001
            return len(self._fallback)

    # ---------- 结果 ----------

    async def report(self, payload: dict) -> None:
        try:
            await self._redis.publish(RESULTS, json.dumps(payload, ensure_ascii=False))
            await self._redis.set(f"aos:result:{payload.get('task_id')}", json.dumps(payload, ensure_ascii=False), ex=3600)
        except Exception:  # noqa: BLE001
            pass

    async def wait_result(self, task_id: str, timeout: float = 120.0) -> dict | None:
        """先查已存结果，再订阅等待。"""

        key = f"aos:result:{task_id}"
        raw = await self._redis.get(key)
        if raw:
            return json.loads(raw)

        pubsub = self._redis.pubsub()
        await pubsub.subscribe(RESULTS)
        deadline = time.time() + timeout
        try:
            while time.time() < deadline:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if not msg:
                    continue
                data = json.loads(msg["data"])
                if data.get("task_id") == task_id:
                    return data
        finally:
            await pubsub.unsubscribe(RESULTS)
            await pubsub.close()
        return None

    # ---------- 心跳 ----------

    async def beat(self, worker_id: str) -> None:
        try:
            await self._redis.hset(HEARTBEAT, worker_id, time.time())
            await self._redis.expire(HEARTBEAT, 300)
        except Exception:  # noqa: BLE001
            pass

    async def workers(self) -> dict[str, float]:
        try:
            raw = await self._redis.hgetall(HEARTBEAT)
        except Exception:  # noqa: BLE001
            return {}
        return {k: float(v) for k, v in raw.items()}

    async def close(self) -> None:
        try:
            await self._redis.aclose()
        except Exception:  # noqa: BLE001
            pass
