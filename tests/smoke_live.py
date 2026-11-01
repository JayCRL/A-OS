"""真实模型冒烟（会花一点额度）。

用法：AOS_HOME=/tmp/aos-live-smoke python3 tests/smoke_live.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernel.agent import Agent  # noqa: E402


async def main() -> None:
    home = Path("/tmp/aos-live-smoke")
    shutil.rmtree(home, ignore_errors=True)
    (home / "workspace").mkdir(parents=True)
    (home / "workspace" / "plan.md").write_text(
        "# 本周安排\n- 周三：评审接口\n- 周五：交周报\n", encoding="utf-8")
    agent = Agent(home, budget=8000)
    await agent.boot()
    script = [
        "你好，我叫阿澈，做后端的",
        "记住：dya 星期一回来，周一别给我排事",
        "dya 什么时候回来？",
        "我 plan.md 里写了什么？",
        "武汉科技大学官网是哪个？",
        "我叫什么？做什么的？",
    ]
    try:
        for line in script:
            print(f"\n你 > {line}")
            print(f"A > {await agent.reply(line)}")
            await agent.wait_idle()
            k = agent.debug_snapshot()["kernel"]
            print(f"   [pages={k['pages']['active']} swapped={k['pages']['swapped']} "
                  f"faults={k['faults']} steps={k['steps']} nodes={k['nodes']['total']} "
                  f"mem={agent.memory.stats()['total']}]")
        print("\n--- 状态路径 ---")
        print(agent.dag.render(limit=10))
    finally:
        await agent.wait_idle()
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
