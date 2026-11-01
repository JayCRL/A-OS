# 图记忆：抽象、折叠、两级拓扑

这是 A-OS 的"磁盘"实验层：把无限长的 episode 流水压成一张轻量图，检索时只换入 3~5 个节点。

当前状态：**独立引擎，已跑通合成压测；还没有接进对话 Agent 和页表。**

## 为什么不能按文本建节点

- 每个用户回合建一个节点 → 节点数随聊天轮数线性爆炸。
- 每段代码/每条报错建一个节点 → 图变成日志索引，检索噪声大于信号。
- 规则抽实体 → 遇到"最近有点烦""那个方案"这类笼统记忆直接失效。

所以图里只放**高价值实体**，原始流水放 `episodes`，试错过程放 `trials`。

## 节点

| 类型 | 例子 | 上限 |
|---|---|---|
| project | 校园二手平台 | 1 |
| module | 登录页 / 鉴权 | 每项目几十 |
| task | 修复登录越权 Bug | 每模块 ≤ 200（默认） |
| decision | 选 JWT 不选 session | 每模块少量 |
| lesson | 方案 A/B 超时，C 有效 | 每任务 ≤ 3（属性，不一定是节点） |
| artifact | 最终 patch | 默认是 task 属性；被复用时才升节点 |
| concept | 安全红线 / 幂等规范 | 全局 ≤ 100 |

节点字段：`key, type, title, summary, attrs, vec, parent_id, importance, status, spectral`。

## 三种压缩

### 1. 升维：episode → task

`Consolidator` 在任务完成或缓冲满时归并：

```
10 轮试错 + 39 句闲聊 + 1 次成功
  → 1 个 task 节点
  → 50 条 trials
  → lessons / artifact 挂 task.attrs
```

同 key 的 task 只更新，不新增；同模块内相似度 > 0.93 的节点可合并（默认关闭，避免误合并模块）。

### 2. 成功即折叠

- 失败过程写 `trials(task_key, step, action, result, ok, elapsed)`；
- 成功后才把 `artifact`、`lessons` 写进 task 属性；
- 失败试错不建图节点。

### 3. 两级拓扑

- 骨架：`project → module → task`，用 `parent_id` + `contains` 边。
- 局部：同模块内 `similar` / `depends_on` / `modifies`，度上限 12。
- 跨模块只保留少量强边。

## 谱：特征值到底怎么用

用**归一化拉普拉斯** `L = I - D^{-1/2} A D^{-1/2}`：

- 最小非零特征值 → 社区/慢变模式，Fiedler 向量可以递归二分建树；
- 最大特征值 → 高频噪声/二分性，不用于主干；
- 用 `eigh` 取第 1..k 个特征向量作为节点低频坐标（k=8），用于重排和聚类；
- 大规模用 `eigsh(..., which='SM')`，作为后台任务。

不要把"最大特征值"直接当树；树来自递归二分或图粗化。

## 检索

```
query
  → 别名精确/子串命中（有别名就不再让向量近邻污染锚点）
  → 锚点 + parent_chain（骨架）+ children + 1~2 跳邻居
  → 0.8*base + 0.35*向量 + 0.15*重要度 + 0.05*新鲜度 + 0.10*谱亲和
  → MMR 去重 + token 上限
  → 3~5 个节点
```

## 合成压测

`tests/synthetic_graph.py`：

- 50 模块 × 4 任务 × 50 episode = **10000 轮**
- 每任务 10 次失败 + 39 句闲聊 + 1 次成功

结果：

```
active_nodes=251   edges=712   trials=2000   max_degree=12
by_type: project 1 / module 50 / task 200
nodes/episode=0.0251
检索「继续搞模块17」→ 5 个节点，316 tokens
```

10k 轮流水只留下 251 个节点；试错 2000 条全部在 `trials`，不在图里；每次检索换入 5 个节点。

跑：

```bash
python3 tests/synthetic_graph.py
python3 -m unittest tests.test_graph -v
```

## 接入 A-OS 的下一步

1. `LLMExtractor`：从真实对话填 `module/task/success/lesson/artifact`，替代脚本里的结构化 Episode。
2. 把 `GraphRetriever.context()` 的结果映射成 `PageTable` 的 DATA 页。
3. 缺页从"先读文件"改成"先查图"，文件只作为节点 provenance。
4. Consolidator 挂后台低优先级任务，任务边界触发，不阻塞对话。
