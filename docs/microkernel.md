# 微内核：phase 1 + phase 2

目标：内核常驻，服务常驻，用户输入创建进程；pager / librarian 在内核外。

当前状态：**进程、IPC、缺页陷阱、段页式 pager、长期记忆/画像/最近对话已接通。**

还差：Web/CLI 切换、VFS 工具、DMA/spool、服务拆独立进程。

## 结构

```
Microkernel
 ├─ MessageBus        命名端点 + 请求/响应 + 超时 + 队列统计
 ├─ ProcessTable      ready/running/waiting/done/killed
 ├─ KernelScheduler   优先级队列，最多 3 个进程协程
 ├─ StateDAG          每个进程一条状态路径（pid）
 ├─ Journal           任务租约/断点
 │
 ├─ pager 服务        PageTable + MemoryStore + Dialogue + GraphStore + GraphRetriever
 ├─ librarian 服务    LLMExtractor + Consolidator（后台写图）
 └─ agent_proc        进程 → compose / fault / state / journal → 模型
```

内核只做机制：进程、调度、IPC、缺页陷阱。图检索、模型抽取、页策略都在服务里。

## 启动

```
mk.boot()
  → 注册 kernel / pager / librarian
  → bus.start()
  → pager.boot()
       打开 pages.db / brain.db / graph.db
       挂载 SYSTEM 页（人格、协议、时间）
       图节点 ≥3 时后台算谱坐标
  → librarian.start()   启动后台编目 worker
  → sched.start()
  → state = running
```

## 一次请求：缺页陷阱

```
mk.ask("继续搞登录页")
  → Process(pid=1, state=ready)
  → 调度器 → agent_proc
  → pager.begin_turn(pid, text)
       对话入库
       上一轮 HUMAN 降为 DAG_STATE，旧 DAG 降为 DATA
       当前指令建 HUMAN 页
       身份/偏好从 brain.db 取出
       相关记忆建成 DATA 页
  → kernel.state_add(goal)
  → 循环：
       pager.compose(pid, query, dag, profile, hot)
       → 模型
       ├─ <|PAGE_FAULT: 继续搞登录页|>
       │    → kernel.fault
       │        进程 state=waiting
       │        pager.fault:
       │            GraphRetriever 找 3~5 个节点
       │            每个节点 upsert 成 DATA 页（source=graph:<id>）
       │            hot 页钉住
       │        进程 state=running
       │    → 重组上下文，继续
       ├─ 工具 JSON（暂未接 VFS）→ 记录后继续
       └─ 自然语言 → 回答
  → pager.finish_turn
       回答写成 DAG_STATE 页
       旧 turn 页 compact
       episode 入长期记忆
  → state_add(say) + prune(pid)
  → journal.finish
  → librarian.observe(user, answer)  // 异步写图
  → 进程 done，future 返回回答
```

关键点：

- 缺页时进程真的进入 `waiting`，pager 返回后才回 `running`。
- 缺页是图优先：`GraphRetriever` → 图节点 → DATA 页，不再先读文件。
- `hot` 页本轮钉在 `must_include`，不会被同轮淘汰。
- 状态树按 `pid` 隔离，多个进程不会共用一条主干。

## 段页式

`pager` 拥有 `PageTable`：

| 段 | 内容 | 策略 |
|---|---|---|
| SYSTEM | 人格、协议、时间 | 常驻，每轮刷新时间 |
| HUMAN | 当前用户指令 | 本轮必带 |
| DAG_STATE | 最近对话、回答 | 常驻，超 6 页/2400 token 降级 |
| DATA | 图节点页、长期记忆页 | 按预算换入换出 |

图节点页内容 = `GraphStore.render(node)`（summary + lessons + artifact + 子节点），不是整份文件。

## 服务边界

| 服务 | 职责 | 在缺页路径 |
|---|---|---|
| kernel | 进程/调度/IPC/fault trap/状态树/journal | 是 |
| pager | PageTable + 记忆 + 图检索 | 是 |
| librarian | LLM 编目 + Consolidator + 写图 | 否 |

pager 和 librarian 当前共享同一个 `GraphStore` 实例（同进程）；拆进程时把写操作收敛到 memory 服务，pager 只读。

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

写图：1 project + 1 module + 1 task，检索“继续搞登录页”拿到 3 个节点。

## 测试

```bash
python3 -m unittest tests.test_microkernel -v
python3 tests/mk_demo.py
```

9 个微内核用例：

- 启动/服务注册/内核 state
- 未知服务
- 进程生命周期
- ask 走 pager
- 缺页陷阱：图节点 → DATA 页、SYSTEM/HUMAN/DAG 段、进程 done
- librarian 折叠/跳过噪音
- 重启持久化
- 总线计数
- 进程异常 → killed

全量：

```
Ran 34 tests in 5.775s
OK
```

## 还没做

1. Web/CLI 改成 `mk.ask`。
2. VFS 服务：文件/网页工具，替换 agent 里“工具未接入”。
3. DMA / 输入输出 spool。
4. pager 和 librarian 拆独立进程，消息总线换 Unix socket。
5. 谱坐标定期重算、图上限触发的在线合并。
