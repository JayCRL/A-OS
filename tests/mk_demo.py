"""微内核骨架演示：启动 → 服务 → 进程 → pager → 回答。"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drivers.model import ModelReply, Usage  # noqa: E402
from kernel.microkernel import Microkernel  # noqa: E402
from tests.fake_encoder import TokenEncoder  # noqa: E402


class DemoModel:
    ready = True

    def __init__(self) -> None:
        self.usage = Usage()

    async def chat(self, messages, *, max_tokens=512, temperature=0.3):
        return ModelReply("我看到登录页和它的任务了。", Usage(20, 10))


async def main() -> None:
    home = Path(tempfile.mkdtemp())
    mk = Microkernel(home, model=DemoModel(), encoder=TokenEncoder())
    await mk.boot()
    try:
        for module in ("登录页", "支付页"):
            for i in range(10):
                await mk.bus.call("librarian", "observe", module=module,
                                  task="修复越权" if module == "登录页" else "重试幂等",
                                  success=False, lesson=f"方案{i % 3}超时",
                                  text="尝试失败", attempts=10)
            await mk.bus.call("librarian", "observe", module=module,
                              task="修复越权" if module == "登录页" else "重试幂等",
                              success=True, artifact=f"{module}.patch", text="完成")
        await mk.bus.call("librarian", "flush")

        print("kernel:", (await mk.bus.call("kernel", "state"))["state"])
        print("services:", mk.bus.endpoints())
        print("graph:", mk.pager.graph.stats())

        answer = await mk.ask("继续搞登录页")
        print("answer:", answer)
        print("processes:", (await mk.bus.call("kernel", "processes")))
        print("bus:", (await mk.bus.call("kernel", "stats"))["bus"])
    finally:
        await mk.close()


if __name__ == "__main__":
    asyncio.run(main())
