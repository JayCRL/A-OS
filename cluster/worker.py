"""独立 Worker 进程：从 Redis 抢任务，跑完把结果写回。

一个 Worker 一个工作目录、一个内核实例；进程挂了不影响别人。
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/aos")

from cluster.broker import Broker, Task


class Worker:
    def __init__(self, name: str, root: Path, *, broker_url: str = "redis://127.0.0.1:6379/0") -> None:
        self.name = name
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.broker = Broker(broker_url)
        self._stop = False

    async def run(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stop", True))
        print(f"[{self.name}] 启动，等待任务…", flush=True)

        heartbeat = asyncio.create_task(self._beat_loop())
        try:
            while not self._stop:
                task = await self.broker.take(timeout=3)
                if task is None:
                    continue
                await self._handle(task)
        finally:
            heartbeat.cancel()
            await self.broker.close()

    async def _beat_loop(self) -> None:
        while True:
            await self.broker.beat(self.name)
            await asyncio.sleep(10)

    async def _handle(self, task: Task) -> None:
        from main import AOS

        started = time.time()
        print(f"[{self.name}] 接单 {task.task_id}: {task.goal[:40]}", flush=True)
        home = self.root / task.task_id
        home.mkdir(parents=True, exist_ok=True)
        (home / "workspace").mkdir(exist_ok=True)

        os.environ["AOS_HOME"] = str(home)
        aos = AOS(home=home)
        try:
            await aos.boot()
            node = aos.dag.add(task.goal, kind="goal")
            await aos._step(node, None, task_id=None)
            result = {
                "task_id": task.task_id,
                "worker": self.name,
                "ok": True,
                "elapsed": round(time.time() - started, 1),
                "tokens": aos.kernel.usage.total,
                "summary": aos.dag.render(limit=3)[-300:],
            }
        except Exception as exc:  # noqa: BLE001
            result = {
                "task_id": task.task_id,
                "worker": self.name,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed": round(time.time() - started, 1),
            }
        finally:
            await aos.shutdown()

        await self.broker.report(result)
        print(f"[{self.name}] 完成 {task.task_id} ok={result['ok']} {result.get('elapsed')}s", flush=True)


async def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else f"worker-{socket.gethostname()}"
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"/opt/aos/data/workers/{name}")
    await Worker(name, root).run()


if __name__ == "__main__":
    asyncio.run(main())
