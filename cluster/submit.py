"""提交任务到集群并等结果：python3 -m cluster.submit "目标" """
import asyncio, sys, time
sys.path.insert(0, "/opt/aos")
from cluster.broker import Broker, Task

async def main():
    goal = " ".join(sys.argv[1:]).strip()
    if not goal:
        print("用法: python3 -m cluster.submit \"任务目标\"")
        return
    b = Broker()
    task = Task(goal=goal, workspace="")
    started = time.time()
    await b.submit(task)
    print(f"已提交 {task.task_id}，队列 {await b.depth()}")
    result = await b.wait_result(task.task_id, timeout=300)
    if not result:
        print("等待超时")
    else:
        print(f"[{result.get('worker')}] ok={result.get('ok')} "
              f"{result.get('elapsed')}s tokens={result.get('tokens')}")
        if result.get("error"):
            print("错误:", result["error"])
        else:
            print(result.get("summary", "")[-300:])
    print(f"总耗时 {time.time()-started:.1f}s")
    await b.close()

if __name__ == "__main__":
    asyncio.run(main())
