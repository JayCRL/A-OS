# A-OS 基线测试（2026-11-02）

环境：生产 8799，split（kernel/pager/librarian/vfs），DeepSeek flash，BGE-small-zh。
脚本：`tests/bench_memory.py`、`tests/bench_live.py`。

## 1. Needle recall（合成）

20 个暗号随机埋进 1000 / 5000 条事实，查询不带暗号词，只看“第 N 条记录里的暗号”。

```
1000 条：
  flat vector recall@8      20/20
  A-OS hybrid recall@8      20/20
  A-OS context recall@2400  20/20
  平均召回                 17.96 ms
  平均上下文               58.6 tokens

5000 条：
  flat vector recall@8      20/20
  A-OS hybrid recall@8      20/20
  A-OS context recall@2400  20/20
  平均召回                 140.77 ms
  平均上下文               61.0 tokens
```

说明：

- 这是**词面暗号**，不是语义改写；TokenEncoder 下 flat baseline 也满分。
- 它证明的是：5000 条量级仍能在 2400 token 预算内命中，且召回延迟可接受。
- **不证明** A-OS 比普通向量检索更懂语义。

## 2. 线上任务冒烟

6 个任务，真实生产模型 + 工具链：

| 任务 | 结果 | wall | tokens |
|---|---|---|---|
| read `bench_note.txt` | ✅ 原样返回 `BENCH-READ-ALPHA` | 3.28s | 2803 |
| edit `bench_edit.txt` foo→bar | ✅ 文件真实变为 bar | 4.11s | 2764 |
| verify 工作区 | ✅ 编译检查通过 | 3.04s | 2407 |
| 记住暗号 | ✅ 确认 | 2.17s | 1425 |
| 回忆暗号 | ✅ 答出 `青竹AOS` | 1.95s | 1334 |
| 查武汉天气 | ✅ 14°C / 晴 | 3.98s | 1891 |

```
success 6/6
avg wall 3.09s
avg tokens 2104
```

## 3. 已有复杂任务证据

- 并行 fan-out：2 个天气子代理 + 1 个汇总子代理，全部 `done`；
- 阶段 workflow：调研阶段并行、汇总阶段串行，结果正确；
- 自动验收：代码 `edit` 后跑 `verify`，结果回灌；
- 事实失效：北京→上海后旧事实不再召回；
- 空闲巩固：摘要写入 MemoryStore。

## 4. 不能从这组数据得出的结论

- 没有和 DSH / Mem0 / Letta / Zep 做同题对比；
- 没有人工评分，成功标准是脚本判定的；
- 样本只有 6 个任务、20 个 needle；
- 模型/网络/生产状态会影响延迟；
- 语义召回、千轮对话、多用户并发还没测。

## 5. 下一步该补的基准

1. BGE 语义改写 needle（同义句、无实体词）；
2. 1000 轮长对话，隔 500 轮问；
3. 与 flat RAG、DSH 原生上下文做同题 A/B；
4. 任务成功率按类别统计（读/改/跑/查/记）；
5. p50/p95 延迟、token 成本。
