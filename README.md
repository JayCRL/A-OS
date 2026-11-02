# A-OS

把操作系统的**虚拟内存、进程隔离、按需换页**搬进 LLM Agent 的运行时。

上下文不是一段不断变长的 `messages`，而是一块有预算、能换入换出、按进程隔离的虚拟内存。用户看不到页表，只感觉它“记得住、接得上、不会越聊越傻”。

```
你 > 记住，dya 星期一回来
A-OS > 好，记住了。周一不给你排事。

（十几轮之后，中间聊过别的）

你 > dya 什么时候回来
A-OS > 星期一。
```

想看机制：网页右上角 `⋯`，或 `?debug=1`。

---

## 60 秒认识

现在 A-OS 同时有两层：

| 层 | 做什么 |
|---|---|
| **微内核** | 进程、调度、IPC、段页式上下文、换入换出、服务拆分 |
| **记忆** | 长期事实/偏好/计划、图节点、文件块、空闲巩固、事实失效 |
| **Agent** | 工具调用、计划、子代理、阶段 workflow、自动验收 |

它不是一个“更大的 prompt”，也不是纯 RAG：

- 相关信息在**调模型前自动换入**，不靠模型发缺页、也不靠模型主动调记忆工具；
- 页表按 `pid` 隔离，不同会话/子代理有各自工作集；
- 长期记忆不删除，只降权/失效；
- 工具能真读、真改、真跑，不是只给建议。

---

## 架构

```
用户 / Web / DSH 壳
        │
        ▼
┌─────────────────────────────────────────────┐
│ kernel（微内核）                             │
│  Dispatcher   输入 → 复用会话进程 / 新建     │
│  进程表        ready/running/waiting/done    │
│  调度器        短任务 worker + 会话常驻 loop │
│  IPC           Unix socket JSON-RPC          │
│  状态树        DAG_STATE / journal           │
└───────────────┬─────────────────────────────┘
                │
     ┌──────────┼──────────┐
     ▼          ▼          ▼
   pager     librarian     vfs
   页表       抽取/图      文件/网页
   记忆       巩固         索引/DMA
   图检索
```

split 模式下是四个真实进程：

```
python3 mk_main.py --web --port 8799 --split
  ├─ kernel
  ├─ services.worker pager
  ├─ services.worker librarian
  └─ services.worker vfs
```

同进程模式用于测试和单机调试。

---

## 虚拟记忆

### 分段

`kernel/page_table.py`，落盘在 `pages.db`：

| 段 | 内容 | 策略 |
|---|---|---|
| SYSTEM | 人格、协议、时间 | 常驻，不换出 |
| HUMAN | 当前用户指令 | 本轮必带 |
| DAG_STATE | 最近对话、任务状态 | 常驻，超限降级 |
| DATA | 文件块、图节点、长期记忆、旧对话 | 按分数换入换出 |

- 一页是语义块，不是定长行；
- `compose()` 用 `相似度 + 字面 + 新鲜度 + 重要度` 选 DATA；
- 总预算硬约束（默认 8000 token）；超了换出低分页，不清空整段；
- 上一轮 HUMAN 降为 DAG_STATE，更旧的降为 DATA，需要时还能召回；
- **每个 pid 一张页表**，共享 SYSTEM，不共享 HUMAN/DAG/DATA。

### 自动预取，不是模型缺页

一轮开始：

```
begin_turn(pid, text)
  → 当前指令进 HUMAN
  → 身份/偏好进 DATA
  → 最近对话进 DAG

prefetch(pid, query)
  → GraphRetriever 找相关项目/模块/任务节点
  → VFS 语义索引找文件块
  → 长期记忆多路召回
  → 全部变成当前 pid 的 DATA 页

compose(pid, query, budget)
  → 按段拼装、按预算裁剪
  → 模型只看到已经装好的上下文
```

模型不知道“页”的存在，也不需要输出 `<|PAGE_FAULT|>` 或调用 `read` 才能拿到上下文。
文件工具是**能力**，不是换页入口。

### 会话进程

`kernel/dispatcher.py`：

- 同一个 channel 默认复用最近的会话进程；
- 只有明确“新任务/新话题/重新开始”才新建；
- 会话空闲 10 分钟过期；
- 会话 loop 是独立 asyncio task，不占调度 worker；
- 真正跑一轮时才占用 turn 信号量（并发 3）。

---

## Agent 能力

### 工具

VFS 提供：

```
read    list    glob    grep    search
write   append  edit    shell   verify
web_search      web_fetch
```

- `shell` 走工作区白名单沙箱，30 秒超时，输出截断；
- `edit` 精确替换，改完失效对应页；
- `verify` 自动识别 `pytest / package.json / compileall`；
- 代码文件 `edit` 后自动跑一次 verify，结果回灌；
- 没有实际调用 `write/edit/shell` 却声称“改好了”，会被系统打回一次。

### 计划、子代理、workflow

- `plan/todo`：步骤写进进程状态，每轮注入；
- `agent/task`：派生一个 child pid，父进程等待汇总；
- `agent/tasks`：最多 4 个子任务并行 fan-out，并发 3；
- `workflow/phases`：阶段内并行、阶段间串行，前序结果注入后一阶段；
- 子代理最多两层，避免无限套娃。

工具步数上限 **24**，够走“读 → 改 → 跑 → 修”。

---

## 长期记忆

`kernel/memory.py`，`brain.db`：

```
identity / preference / fact / plan / episode / summary
```

- 每轮 librarian 抽 facts（身份/偏好/事实/计划）；
- 重复记忆按向量相似度合并；
- 冲突用 `superseded / valid_to` 标记失效，不删除；
- 未被引用的记忆缓慢衰减（只降可提取性）；
- 身份/偏好每轮固定带上，其余按需换入；
- 召回：`0.58*向量 + 0.14*字面 + 0.08*新鲜 + 0.20*重要度 + 稀有片段加成`，MMR 去重。

### 空闲巩固

不是固定每 8 轮切一刀，而是**空闲时像休息一样巩固**：

```
队列为空
且不在处理/不在巩固
且未巩固 >= 4 条
且距离最后一次对话 >= 10 分钟
  → consolidation
```

兜底：未巩固 >= 40 条，即使没到 10 分钟也压一次。

巩固输出：

```
summary / facts / supersedes / open_tasks / topics
  → summary 进 MemoryStore
  → facts 进 profile
  → supersedes 让旧事实失效
  → 原文 episode 保留
```

### 记忆事件

`memory_events` 记录每轮：

```
ts, pid, query, page_ids, used_ids, outcome, tokens, latency
```

用于后续 bandit/归因；`used_ids` 和 `outcome` 目前先留空，不假装有奖励。

---

## 图记忆

`kernel/graph.py`：

- 只放 `project / module / task / decision / lesson / artifact / concept`，不放聊天轮次；
- 失败试错进 `trials`，成功才折叠成 task 的 `artifact / lessons`；
- 上层 `parent_id` 骨架树 + 局部图，度上限 12；
- 谱坐标用归一化拉普拉斯的最小非零特征值；
- 检索：别名/向量锚点 → 骨架路径 → 1~2 跳 → 谱重排 → 3~5 个节点。

图现在是 pager 的自动预取来源之一，不再只是独立引擎。

---

## Web 与可观测

```bash
python3 mk_main.py --web --port 8799 --split
```

- 聊天：`http://127.0.0.1:8799/`
- 调试抽屉：右上角 `⋯` 或 `?debug=1`
- 面板每 2 秒刷新：服务灯、进程卡片、会话卡片、记忆环图、页表分段条、最近回答
- 终端模式：`/state`、`/memory`、`/help`、`/quit`

公开部署示例：`https://11408.online/aos/`（nginx 代理 `/aos/` → `127.0.0.1:8799`）。

---

## 接 DSH 产品层

A-OS 可以作为 DSH 的“记忆内核”：

```
DSH：UI / agent loop / tools / skills / 审批 / 子代理
A-OS：长期记忆 / 图 / 文件块 / 事实失效 / 空闲巩固
```

不是 MCP。DSH 侧用原生 Cordis 插件：

- `agent/pre-step`（step 1）→ `POST /api/memory/recall` → 注入 memory block；
- `agent/turn-stopping` → `POST /api/memory/commit` → episode/facts；
- 超时静默降级，不阻塞 DSH。

插件在 [`integrations/dsh-plugin/`](integrations/dsh-plugin/)，独立 profile 安装，不改 DSH 核心。

---

## 快速开始

```bash
cd /opt/aos-mk
pip install -r requirements.txt

# 终端
python3 mk_main.py

# Web + split 服务
AOS_HOME=/opt/aos/data python3 mk_main.py --web --port 8799 --split
```

可选：把 `bge-small-zh-v1.5` 放到 `models/bge-small-zh/`（`model.onnx` + `tokenizer.json`）；没有会自动退化为哈希编码。

模型 Key 读 `DEEPSEEK_API_KEY` 或 `~/.dsh/.env`。

### systemd

```ini
[Unit]
Description=A-OS Microkernel Web
After=network.target

[Service]
WorkingDirectory=/opt/aos-mk
Environment=AOS_HOME=/opt/aos/data
Environment=PYTHONPATH=/opt/aos-mk
ExecStart=/usr/bin/python3 /opt/aos-mk/mk_main.py --web --port 8799 --split
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 配置

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `AOS_HOME` | `/opt/aos/data` | 数据、页表、工作区 |
| `AOS_CTX_BUDGET` | `8000` | 上下文 token 预算 |
| `AOS_MODEL` | `auto` | `offline` 强制离线 |
| `AOS_DEBUG` | 空 | 内部日志同时打终端 |
| `DEEPSEEK_API_KEY` | 读 `~/.dsh/.env` | 模型 Key |

数据目录：

```
data/
  brain.db      记忆 + 对话 + memory_events
  pages.db      页表
  dag.db        状态树
  tasks.db      任务日志/租约
  chunks.db     工作区文件向量索引
  graph.db      图记忆
  workspace/    它能读写的文件
  logs/         worker 日志
```

---

## 测试与基线

```bash
python3 -m unittest tests.test_chat tests.test_graph tests.test_microkernel
# 58 tests
```

记忆基线：

```bash
python3 -m tests.bench_memory
```

```
1000 条：needle recall 20/20，平均 18 ms，上下文 58.6 tok
5000 条：needle recall 20/20，平均 141 ms，上下文 61.0 tok
```

线上任务冒烟：

```bash
python3 -m tests.bench_live
```

```
read / edit / verify / remember / recall / web
success 6/6
avg wall 3.09s
avg tokens 2104
```

完整结果和局限见 [docs/benchmark.md](docs/benchmark.md)。

**注意**：合成 needle 是词面暗号，flat vector 也满分；它证明“5000 条能在预算内命中”，不证明“语义上强于别人”。语义改写、1000 轮长对话、与 DSH/Mem0 的同题 A/B 还没做。

---

## 当前状态与边界

已落地：

- 段页式上下文、per-pid 页表、换入换出；
- 自动预取图/文件/记忆，不走模型缺页；
- Dispatcher 会话复用、常驻会话进程；
- 空闲 consolidation、facts、supersede、decay；
- read/glob/grep/edit/shell/verify；
- plan、subagent、并行 fan-out、workflow；
- DSH 原生插件桥。

边界：

- 子代理最多两层；workflow 是线性阶段，不是 DAG；
- shell 只限工作区/白名单，不是完整终端；
- 没有浏览器自动化、MCP 工具市场；
- 没有流式工具卡片/diff/审批 UI；
- 记忆巩固要等空闲，回答当下可能还没消化；
- 语义召回和千轮对话还缺严格评测。

---

## 路线

1. 语义改写 needle + 1000 轮长对话基准；
2. 与 flat RAG / DSH 原生上下文做同题 A/B；
3. 记忆 bandit（alpha/beta、引用归因、衰减）；
4. 自动修复循环（verify 失败回灌直到通过或超预算）；
5. workflow DAG + 结果去重/投票；
6. 更完整的权限、审计和流式 UI。

---

## 设计取舍

上一版有两个极端：内核日志糊在用户脸上，或者干脆退化成聊天 loop。现在把两件事分开：

- **机制保留**：分页、分段、调度、状态树、租约、记忆、图、巩固都真的在链路上；
- **呈现收起来**：正常对话只给回答，机制进日志和调试面板。

页表也修过几个硬伤：`compose` 不再把没选中的 DATA 全换出；文件改过页会失效；`segment/source/stale` 会落库；预算按实际 token 估，不再严重低估。
