import asyncio, sys, tempfile, os, time
sys.path.insert(0, "/opt/aos")
from kernel.io_spooler import IOSpooler, HumanAction, ActionKind
from kernel.kernel_scheduler import KernelScheduler, Job

async def main():
    sched = KernelScheduler(max_workers=2)
    await sched.start()
    log = []

    async def worker(name, secs):
        try:
            for i in range(secs):
                await asyncio.sleep(0.2)
                log.append(f"{name} step{i}")
        except asyncio.CancelledError:
            log.append(f"{name} 被中断，写 checkpoint")
            raise

    for i in range(2):
        sched.submit(Job(priority=1, seq=0, name=f"bg{i}", run=lambda i=i: worker(f"bg{i}", 10),
                         on_cancel=lambda jid: asyncio.sleep(0)))
    await asyncio.sleep(0.5)
    print("运行中:", sched.stats()["running"])

    killed = await sched.preempt()
    print("被取消:", killed)
    print("日志:", log[-4:])

    # 人类事件走最高优先级
    async def human():
        log.append("human 处理完成")
    sched.submit_human("改需求", human)
    await asyncio.sleep(0.3)
    print("human 是否执行:", "human 处理完成" in log)

    # 文件锁
    async def hold(path, tag):
        async with sched.locks.acquire(path, tag, timeout=2):
            log.append(f"{tag} 拿到 {path}")
            await asyncio.sleep(0.3)
    await asyncio.gather(hold("/a.go", "A"), hold("/a.go", "B"))
    print("锁顺序:", [x for x in log if "拿到" in x])

    await sched.stop()

asyncio.run(main())
