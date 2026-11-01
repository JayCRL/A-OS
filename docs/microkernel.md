# 微内核骨架（phase 1）

目标：内核常驻，服务常驻，用户输入创建进程；pager / librarian 在内核外。

当前状态：**骨架跑通，尚未接 PageTable 四段页和缺页陷阱。**

## 结构

```
Microkernel
 ├─ MessageBus        命名端点 + 请求/响应 + 统计
 ├─ ProcessTable      ready/running/waiting/done/killed
 ├─ KernelScheduler   优先级队列，最多 3 个进程协程
 ├─ pager 服务        GraphStore + GraphRetriever
 ├─ librarian 服务    LLMExtractor + Consolidator
 └─ agent_proc        向 pager 要 3~5 个节点，调模型回答
```

内核只做机制：进程、调度、IPC。图检索、模型抽取都是用户态服务。

## 启动

```
mk.boot()
  → 注册 kernel / pager / librarian 端点
  → bus.start()
  → pager.boot()       打开 graph.db，必要时预热谱坐标
  → librarian.start()  启动后台编目 worker
  → sched.start()
  → state = running
```

## 一次请求

```
mk.ask("继续搞登录页")
  → Process(pid=1, name="chat", payload={text})
  → sched 交给 agent_proc
  → bus.call("pager","resolve", query=..., k=5)
  → pager 返回 3~5 个图节点（模块 + 父链 + 任务 + 经验）
  → 调模型
  → bus.call("librarian","observe", user, answer)  // 异步编目
  → 进程 done，future 返回回答
```

流程里 pager 在关键路径，librarian 不在。

## 服务边界

| 服务 | 职责 | 依赖 |
|---|---|---|
| kernel | 进程/调度/IPC | 无业务依赖 |
| pager | 图检索、节点渲染 | graph.db |
| librarian | LLM 编目、Consolidator、写图 | 模型、graph.db |

当前 pager 和 librarian 共享同一个 `GraphStore` 实例（同进程 phase 1）；拆进程时要把写操作收敛到 memory 服务，pager 只读。

## LLMExtractor

真实模型已验证：输入

```
登录页的越权修好了，最后是用方案C完成的，方案A和B都超时
```

抽取结果：

```
module=登录页
task=修复登录页越权问题
status=done
lessons=[方案A和方案B均超时，方案C有效]
```

写入图：1 project + 1 module + 1 task，3 条边，检索"继续搞登录页"能拿到 3 个节点。

## 测试

```bash
python3 -m unittest tests.test_microkernel -v
python3 tests/mk_demo.py
```

8 个用例：启动、IPC 未知服务、进程生命周期、ask 走 pager、librarian 折叠/跳过噪音、重启持久化、总线计数、进程异常。

## 还没做

1. 把 `PageTable` 四段页搬进 pager：`resolve` 改成 `fault` + 页映射。
2. 内核缺页陷阱：进程 `waiting`，发 IPC 给 pager，收回页描述后 `running`。
3. 现有 `AgentProcess._turn`（长期记忆/画像/最近对话）接进来。
4. 输入 spool / DMA 服务。
5. Web/CLI 改成 `mk.ask`。
6. 服务拆独立进程。
