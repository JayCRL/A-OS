import asyncio, sys
sys.path.insert(0, "/opt/aos")
from kernel.io_spooler import IOSpooler, HumanAction, ActionKind

async def main():
    sp = IOSpooler(idle_seconds=2.0, max_wait_seconds=8.0)
    # 快速连打：应当合并成一个事件
    await sp.feed(HumanAction(ActionKind.INTENT, "实现登录"))
    await sp.feed(HumanAction(ActionKind.INTENT, "实现登录"))
    await sp.feed(HumanAction(ActionKind.CONTENT, "改了 main.go", path="/a/main.go"))
    await sp.feed(HumanAction(ActionKind.CONTENT, "又改了 handler", path="/a/handler.go"))
    print("喂完，缓存中:", sp.stats)

    ev = await asyncio.wait_for(sp.next_event(), timeout=5)
    print("事件摘要:", ev.summary)
    print("快照条数:", len(ev.snapshot), [s["path"] for s in ev.snapshot])
    print("优先级:", ev.priority, "是否强制:", ev.is_force)

    # 打断信号立即 flush
    await sp.feed(HumanAction(ActionKind.SIGNAL, "Ctrl-C"))
    ev2 = await asyncio.wait_for(sp.next_event(), timeout=2)
    print("信号事件:", ev2.summary, "强制:", ev2.is_force)

asyncio.run(main())
