"""A-OS 入口。

平时它就是一句话进、一句话出；内核、记忆、工具都藏在后面。
调试信息写到 logs/aos.log，终端默认不显示。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel.agent import Agent  # noqa: E402

DATA = Path(os.environ.get("AOS_HOME", "/opt/aos/data"))

HELP = """可用命令：
/state    看看内核状态（状态树 / 页表 / 缺页）
/memory   看看记了多少东西
/debug    开关内部日志（默认关）
/reset    换一个新话题（记忆保留）
/quit     退出"""


async def run_cli(agent: Agent) -> None:
    loop = asyncio.get_running_loop()
    print("A-OS 在。")
    while True:
        try:
            if sys.stdin.isatty():
                sys.stdout.write("你 > ")
                sys.stdout.flush()
            line = await loop.run_in_executor(None, sys.stdin.readline)
        except (EOFError, RuntimeError):
            break
        if not line:
            break
        text = line.strip()
        if not text:
            continue
        if text in ("/quit", "/exit", "exit", "quit", ":q"):
            break
        if text == "/help":
            print(HELP)
            continue
        if text in ("/state", "/kernel"):
            k = agent.debug_snapshot()["kernel"]
            pages = k["pages"]
            print(f"状态树：{k['nodes']['total']} 节点（成功 {k['nodes'].get('success', 0)}）")
            print(f"页表：活跃 {pages['active']} / 换出 {pages['swapped']}，"
                  f"本轮缺页 {k['faults']}，步数 {k['steps']}")
            print("最近路径：")
            for line in (k["path"] or "（空）").splitlines():
                print("  " + line)
            continue
        if text == "/memory":
            stats = agent.memory.stats()
            kinds = "、".join(f"{k} {v}" for k, v in stats["by_kind"].items()) or "空"
            print(f"记得 {stats['total']} 条（{kinds}），对话 {agent.dialogue.count()} 轮。")
            for mem in agent.memory.recent(limit=6):
                print(f"  · {mem.text}")
            continue
        if text == "/debug":
            agent.debug = not agent.debug
            agent._setup_logging()
            print(f"内部日志：{'开' if agent.debug else '关'}")
            continue
        if text == "/reset":
            agent.dialogue.session = f"main-{int(__import__('time').time())}"
            print("好，重新开始。之前的事我还记得。")
            continue

        answer = await agent.reply(text)
        print(answer)


async def amain() -> None:
    agent = Agent(DATA, debug=bool(os.environ.get("AOS_DEBUG")))
    await agent.boot()
    try:
        if "--web" in sys.argv:
            from web.server import serve
            port = 8799
            for i, arg in enumerate(sys.argv):
                if arg == "--port" and i + 1 < len(sys.argv):
                    port = int(sys.argv[i + 1])
            await serve(agent, "0.0.0.0", port)
        else:
            await run_cli(agent)
    finally:
        await agent.wait_idle()
        await agent.close()


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


# 旧代码（cluster/worker.py）里的名字，指向同一套能力
AOS = Agent

if __name__ == "__main__":
    main()
