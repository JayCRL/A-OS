# 微内核：phase 1 ~ 5

目标：内核常驻，服务常驻，用户输入创建进程；pager / librarian 在内核外。

当前状态：**进程、IPC、缺页陷阱、per-process 段页、VFS/DMA/spool、Web/CLI、服务拆独立进程与 Unix socket 已接通。**

还差：切生产 8799、服务崩溃重启、图在线合并。

## 结构

```
Microkernel
 ├─ MessageBus        命名端点 + 请求/响应 + 超时 + 队列统计
 ├─ ProcessTable      ready/running/waiting/done/killed
 ├─ KernelScheduler   优先级队列，最多 3 个进程协程
 ├─ Dispatcher        输入 → 判断复用哪个会话进程 / 新建进程
 ├─ StateDAG          每个进程一条状态路径（pid）
 ├─ Journal           任务租约/断点
 │
 ├─ pager 服务        PageTable + MemoryStore + Dialogue + GraphStore + GraphRetriever
 ├─ librarian 服务    LLMExtractor（split 时只抽取，写图交给 pager）
 ├─ vfs 服务          文件 / 网页 / DMA 批量搬运
 ├─ spool             inbox.jsonl / outbox.jsonl
 └─ agent_proc        进程 → compose / fault / tool / state / journal → 模型
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

## 会话调度器

每条输入先过 Dispatcher，不再"一条消息一个 pid"：

```
同 channel、最近活跃
  + 出现 要/继续/这个/刚才/？ 等指代
  或 与上一轮文本字面重合 ≥ 0.18
  → 复用已有 session 进程

否则 → 新建 session 进程
```

- 一个会话进程是常驻 loop：收到消息 `running`，空闲 `waiting`；
- 会话 loop 是独立 asyncio task，不占调度 worker；真正跑一轮时才占用 turn 信号量（并发 3），所以 12 个空闲会话也不会把后续进程卡在 ready；
- 同一会话共用一个 pid，HUMAN/DAG/DATA 页天然连续；
- 会话 30 分钟无输入过期，进程收 None 退出；
- 旧会话的工作集按 `_reap_old_pids` 降温，再回来时用最近 8 轮兜底。

这就是"Dsh superagent"那类做法的小号版：Dispatcher 是入口调度，agent 进程是会话执行体；以后要加子 agent，只要让 Dispatcher 把任务路由到 child pid。

## 文件块缺页

图里没有（或只有 project 骨架）时，pager 依次回退：

```
GraphRetriever 图节点
  → vfs.semantic（chunks.db 文件块向量索引）
  → 长期记忆
```

文件块页 `source=file:<path>:<offset>`，tag=file，仍然走同一个 DATA 段和预算。

## 独立服务进程

`mk_main.py --split`：

```
kernel 进程
 ├─ vfs    独立进程  → home/ipc/vfs.sock
 ├─ pager  独立进程  → home/ipc/pager.sock
 └─ librarian 独立进程 → home/ipc/librarian.sock
```

- 传输：Unix socket，一行一个 JSON 请求/响应；
- pager 通过 UnixClient 调 vfs 做文件块缺页；
- librarian 只跑 LLMExtractor，把 Episode 通过 `pager.commit_episode` 写图；
- 服务启动顺序 vfs → pager → librarian；
- 服务日志在 `home/logs/worker-*.log`；
- 关闭时 SIGTERM，等 5 秒再 kill。

## per-process 页表

页表按 pid 隔离：

```
PageTable（全局页帧）
  pages[].pid = 0   → SYSTEM 共享
  pages[].pid = N   → 进程 N 的 HUMAN / DAG_STATE / DATA
```

- `begin_turn(pid)`、`fault(pid)`、`compose(pid)`、`finish_turn(pid)`、`compact(pid)` 全部带 pid；
- `compose(pid)` 只渲染 `SYSTEM + pid 自己的 HUMAN/DAG/DATA`；
- `_enforce_data_cap(pid)` 只淘汰本进程的 DATA；
- 同一图节点被两个进程缺页，会各有一份 DATA 页（帧暂不共享）；
- 并发 `ask()` 不再加锁；两个进程同时对话，HUMAN 页不会串。
- 新进程 `begin_turn` 会从 Dialogue 导入最近 8 轮，映射成自己的 DAG_STATE 页；"要""继续"这类指代能接住。

## 服务边界

| 服务 | 职责 | 在缺页路径 |
|---|---|---|
| kernel | 进程/调度/IPC/fault trap/状态树/journal | 是 |
| pager | PageTable + 记忆 + 图检索 | 是 |
| librarian | LLM 编目 + Consolidator + 写图 | 否 |
| vfs | 文件/网页工具、DMA 搬运 | 否（工具轮次） |

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

20 个微内核用例：

- 启动/服务注册/内核 state
- 未知服务
- 进程生命周期
- ask 走 pager
- 缺页陷阱：图节点 → DATA 页、SYSTEM/HUMAN/DAG 段、进程 done
- librarian 折叠/跳过噪音
- 重启持久化
- 总线计数
- 进程异常 → killed
- 工具调用走 VFS、写文件、spool 计数
- DMA 把文件搬进缓存/索引
- Web 路由走微内核
- 按 pid 状态树隔离
- 并发两个 ask：各自的 HUMAN 页不串、各带一份 SYSTEM
- 文件块缺页：图空时从 VFS 语义索引换入
- split 模式：三个独立进程、远程 pager/librarian/vfs 流程
- 跨进程上下文：第二个 pid 能看到上一轮回答
- Dispatcher 复用：第二轮"要"命中同一个 pid，prompt 带上一轮回答
- 12 个空闲会话不阻塞后续消息

全量：

```
Ran 45 tests in 27.4s
OK
```

## 生产切换

2026-11-01 已把 8799 切到微内核 split：

```
WorkingDirectory=/opt/aos-mk
ExecStart=/usr/bin/python3 /opt/aos-mk/mk_main.py --web --port 8799 --split
```

切换前数据备份在 `/opt/aos/backups/data-<时间戳>/`，旧 unit 也在里面。

当前进程：

```
kernel  /opt/aos-mk/mk_main.py
vfs     services.worker vfs
pager   services.worker pager
librarian services.worker librarian
```

影子对比 6 题通过：记忆（JustProve/dya）、文件缺页（login.md）、网页搜索、闲聊、任务折叠（图 3 节点）。

## 还没做

1. 服务崩溃重启/健康检查；现在只负责启动和 SIGTERM。
2. 谱坐标定期重算、图上限触发的在线合并。
3. 三个服务各加载一份编码器，内存还要再压。
4. 微内核版本还没合回 `main`，目前只在 `microkernel` 分支。
