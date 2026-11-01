"""A-OS 微内核入口。

和旧 main.py 的区别：启动的是常驻内核；一次输入创建一个进程。
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel.microkernel import Microkernel  # noqa: E402

DATA = Path(os.environ.get("AOS_HOME", "/opt/aos/data"))
HELP = """/state    内核状态（进程/页表/服务/图）
/memory   记忆条数
/quit     退出"""


async def run_cli(mk: Microkernel) -> None:
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
            snap = mk.debug_snapshot()
            print(f"内核：{snap['state']}  进程 {snap['process_stats']}")
            print(f"服务：{', '.join(snap['services'])}")
            pages = snap.get("pages") or {}
            graph = snap.get("pager") or {}
            print(f"页表：{pages.get('active', 0)} 活跃 / {pages.get('swapped', 0)} 换出")
            print(f"图：{graph.get('active', 0)} 节点 / {graph.get('edges', 0)} 边 / "
                  f"{graph.get('trials', 0)} 试错")
            print(f"总线：{snap['bus']['sent']} 条")
            continue
        if text == "/memory":
            stats = mk.pager.memory.stats()
            print(f"记得 {stats['total']} 条：{stats['by_kind']}")
            continue
        try:
            answer = await mk.ask(text)
        except Exception as exc:  # noqa: BLE001
            answer = f"刚才走神了（{type(exc).__name__}）。"
        print(answer)


async def amain() -> None:
    mk = Microkernel(DATA, split="--split" in sys.argv)
    await mk.boot()
    try:
        if "--web" in sys.argv:
            from web.server import serve
            port = 8799
            for i, arg in enumerate(sys.argv):
                if arg == "--port" and i + 1 < len(sys.argv):
                    port = int(sys.argv[i + 1])
            serve_task = asyncio.create_task(serve(mk, "0.0.0.0", port))
            stop = asyncio.Event()
            stop_task = asyncio.create_task(stop.wait())
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, stop.set)
                except NotImplementedError:
                    pass
            done, _ = await asyncio.wait([serve_task, stop_task],
                                         return_when=asyncio.FIRST_COMPLETED)
            if serve_task in done:
                await serve_task
            serve_task.cancel()
            stop_task.cancel()
            await asyncio.gather(serve_task, stop_task, return_exceptions=True)
        else:
            await run_cli(mk)
    finally:
        await mk.close()


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
