# A-OS

把操作系统的内存管理与进程调度搬进 LLM Agent 的运行时：上下文不是一段不断变长的 `messages` 数组，而是一块可以被换页、被淘汰、按段隔离的虚拟内存。

但用户不该看见这些。对用户来说，它就是一个能接话、记性很好的助手：

```
你 > 记住，dya 星期一回来
A-OS > 好，记住了。周一不给你排事。

（十几轮之后，中间聊过别的）

你 > dya 什么时候回来
A-OS > 星期一。
```

内核在下面跑，日志在文件里，界面只有对话。想看机制，终端 `/state` 或网页右上角 `⋯`。

## 总览

```
用户说话
  ↓
IO 事件 → 调度器（每回合一个任务，并发上限 3）
  ↓
状态树：这一轮做到哪了（抽象状态，不是原始 JSON）
  ↓
段页式上下文（SYSTEM / HUMAN / DAG_STATE / DATA，预算 8000 token）
  ↓
模型
  ├─ 直接说话 ────────────────→ 回复用户
  ├─ <|PAGE_FAULT: 目标|> ────→ 缺页：磁盘 / 语义索引 / 长期记忆
  └─ {"tool": ...} ──────────→ 工具：文件 / 搜索 / 网页
  ↓
看门狗：重复动作、步数上限、空转
  ↓
回答入页（DAG_STATE）→ 下一轮还在
  ↓
后台：抽长期记忆、归档旧对话
```

## 机制

### 分页与分段

`kernel/page_table.py`，落盘在 `pages.db`：

| 段 | 内容 | 策略 |
|---|---|---|
| SYSTEM | 人格、工具协议、时间 | 常驻，永不换出 |
| HUMAN | 当前人类指令 | 本轮必带 |
| DAG_STATE | 最近几轮对话、任务状态 | 常驻，超过 6 页降级 |
| DATA | 文件片段、长期记忆、旧对话 | 按分数换入换出 |

- 一页不是定长行，而是一个语义块（文件片段 / 一条记忆 / 一轮对话）。
- `compose()` 每次重新拼：固定段 + 常驻页 + DATA 里按 `相似度 + 字面重合 + 新鲜度 + 重要度` 选的 Top-K。
- **总预算硬约束**：固定段和常驻页先扣，剩下的才给 DATA；超了就从低分页换出，不整段清空。
- 旧对话不会每轮被清掉：上一轮的 HUMAN 降成 DAG_STATE，更旧的降成 DATA，相关时还能被召回。

### 缺页中断

模型不知道文件内容时，不猜，输出：

```
<|PAGE_FAULT: login.md|>
<|PAGE_FAULT: 上次说的那个登录接口|>
```

内核按顺序找：

1. 工作区文件 → 直接读；
2. 同一 source 的页已在内存 → 只 `swap_in`，不重复建页；
3. 自然语言 → 工作区向量索引 + 长期记忆召回，命中的片段建成 DATA 页。

没有 `read` 工具：读文件内容一律走缺页，这样"什么时候换、换多少"由内核预算决定，而不是模型一次性把整份文件倒进上下文。

换入的页本轮钉在 `must_include` 里，不会再被同轮淘汰。文件改过 → 依赖页 `stale`，下次重新换入。缺页次数会记在状态和调试面板里。

### 状态树

`kernel/state_dag.py`，落盘在 `dag.db`：

- 每个节点是**抽象状态**：`用户：…`、`已换入 login.md`、`web_search "…" → 命中 3 条`、`回答：…`。
- 不存原始 JSON，不存完整文件内容。
- 每步 checkpoint；`prune()` 成功后归档旁支，只留主干。
- 上下文里的「最近状态」就是这个主干，不是一整段聊天记录。

### 看门狗

`kernel/progress.py`：

- 同一个缺页目标 / 同一个工具连续 3 次 → 注入中断，要求换目标或直接回答；
- 步数到 8 → 强制收尾，让模型用一两句话回答；
- 只提醒和叫停，不再"第一次 write 就逼 done"。

### 调度与恢复

- `kernel_scheduler.py`：优先级队列 + 最大并发 3 + 可取消；每个用户回合是一个任务。
- `journal.py`：任务租约、断点、重启恢复。重启时未完成的任务标记失败，不会假装续跑。
- `executor` / `tools.py`：文件路径锁在工作区，写文件原子替换，写完让相关页失效。

### 长期记忆

`kernel/memory.py`，`brain.db`：

- 六类：`identity / preference / fact / plan / episode / summary`。
- 每轮后台静默提取；重复的按向量相似度合并。
- 召回打分：`0.58*向量 + 0.14*字面 + 0.08*新鲜度 + 0.20*重要度 + 稀有片段加成`，MMR 去重。
- 身份/偏好每轮固定带上；其余按需变成 DATA 页换入。
- **库里的记忆不删**。放不进上下文的只是"这次不带"。

## 图记忆（实验）

这一层是"磁盘"的候选实现：把无限长的 episode 流水压成一张轻量图。

- 图里只放 `project / module / task / decision / lesson / artifact / concept`，不放聊天轮次；
- 失败试错进 `trials` 表，成功才折叠成 task 的 `artifact` / `lessons` 属性；
- 上层 `parent_id` 骨架树 + 下层模块内局部图，度上限 12；
- 谱坐标用归一化拉普拉斯的最小非零特征值，不把"最大特征值"当树；
- 检索：别名/向量锚点 → 骨架路径 → 局部 1~2 跳 → 谱重排 → 3~5 个节点。

10k 轮合成压测：**251 个节点、2000 条 trials 不进图、每次换入 5 个节点（316 token）**。

```bash
python3 -m unittest tests.test_graph -v
python3 tests/synthetic_graph.py
```

现在还是独立引擎，没接对话 Agent 和页表；接入方案见 [docs/graph-memory.md](docs/graph-memory.md)。

## 用户看到什么

终端：

```
A-OS 在。
你 > 我 plan.md 里写了什么？
本周就两件事：周三评审接口，周五交周报。周一空着，正好。
```

网页是一个聊天窗口，没有状态树、页表、token 计数。右上角 `⋯` 是维护者抽屉，才看得到：

```
状态树 12 节点 · 成功 8
页表 活跃 5 / 换出 7
缺页 1   本轮步数 3
```

终端命令：

```
/state    内核状态（状态树 / 页表 / 缺页）
/memory   记忆条数与最近几条
/debug    内部日志开关（默认写 logs/aos.log）
/reset    换新话题（记忆保留）
/quit     退出
```

## 跑起来

```bash
pip install -r requirements.txt
cd /opt/aos
python3 main.py                    # 终端
python3 main.py --web --port 8799  # 网页
```

可选：把 `bge-small-zh-v1.5` 放到 `models/bge-small-zh/`（`model.onnx` + `tokenizer.json`），语义召回更准；没有这个目录会自动退化成哈希编码，功能不断。

模型 Key 读 `DEEPSEEK_API_KEY` 或 `~/.dsh/.env`。

## 配置

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `AOS_HOME` | `/opt/aos/data` | 数据、页表、工作区 |
| `AOS_CTX_BUDGET` | `8000` | 上下文 token 预算 |
| `AOS_MODEL` | `auto` | `offline` 强制离线 |
| `AOS_DEBUG` | 空 | 内部日志同时打终端 |
| `DEEPSEEK_API_KEY` | 读 `~/.dsh/.env` | 模型 Key |

```
data/
  brain.db     记忆 + 对话
  pages.db     页表
  dag.db       状态树
  tasks.db     任务日志/租约
  chunks.db    工作区文件向量索引
  workspace/   它能读写的文件
  logs/aos.log 内部日志
```

## 代码结构

```
kernel/
  agent.py            回合：调度 → 组页 → 模型 → 缺页/工具 → 看门狗 → 落页
  page_table.py       分页 / 分段 / 换入换出 / compose
  state_dag.py        状态树
  progress.py         看门狗
  kernel_scheduler.py 调度器
  journal.py          租约与恢复
  memory.py           长期记忆
  dialogue.py         全量对话史
  tools.py            文件 / 搜索 / 网页
  memindex.py         工作区文件切块与语义索引
  persona.py          提示词
  tokens.py           token 估算
web/
  server.py           聊天接口 + 调试接口
  index.html          聊天界面（调试抽屉默认收起）
main.py               入口
tests/
  test_chat.py        17 个用例
  test_graph.py       8 个图记忆用例
  synthetic_graph.py  1 万轮合成压测
  smoke_live.py       真实模型冒烟
```

## 测试

```bash
cd /opt/aos
python3 -m unittest tests.test_chat tests.test_graph -v
python3 tests/synthetic_graph.py
```

覆盖：

- 分页 / 分段 / 预算 / 换出
- `_persist` 真的更新 `segment/source/stale`（旧版这里漏了，重启会复活）
- 缺页换入文件、同一个 source 去重、文件改过后旧页失效
- 状态树节点与最近对话页
- 看门狗拦住重复缺页
- 记忆写入/召回/去重、300 条重启后还在
- 20 轮后上下文仍在预算内
- 工具调用不出现在回答里
- CLI 不打印内核日志
- Web 页面与接口

真实模型冒烟：

```bash
python3 tests/smoke_live.py
```

## 设计取舍

上一版有两个极端：内核日志直接糊在用户脸上；或者干脆不要内核、退化成聊天 loop。这一版把两件事分开：

- **机制保留**：分页、分段、缺页、状态树、看门狗、调度、租约都还在，而且真的在链路上起作用。
- **呈现收起来**：正常对话只输出回答；机制进日志和调试面板。

页表这一层也修了上一版的硬伤：`compose` 不再把没选中的 DATA 全换出；缺页按 `source` 去重；`segment/source/stale` 会落库；token 按中文实际字数估，不再 `len/3` 低估三倍。
