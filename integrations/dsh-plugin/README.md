# @aos/dsh-memory

DSH 原生 Cordis 插件：让 A-OS 管记忆上下文，DSH 管执行上下文。

## 它做什么

- `agent/pre-step`（step 1）：
  - 调 A-OS `POST /api/memory/recall`
  - 把返回的 `context` 作为一条 snapshot 消息插到本轮
  - 超时/失败静默降级
- `agent/turn-stopping`：
  - 从 session 事件取本轮 user/assistant
  - 调 A-OS `POST /api/memory/commit`
  - A-OS 写 episode、抽 facts、后续 consolidation

**不是 MCP**。模型不需要主动调用记忆工具；每轮由插件自动换入。

## 安装（独立 profile，不要动当前 web profile）

```bash
cd ~/.dsh/profiles/<profile>
pnpm add file:/opt/aos-mk/integrations/dsh-plugin
```

在 profile 的 `cordis.patch.yml` 加：

```yaml
- id: aos-memory
  name: '@aos/dsh-memory'
  config:
    baseUrl: http://127.0.0.1:8799
    budget: 2400
    recallTimeoutMs: 800
    commitTimeoutMs: 500
    commit: true
```

重启/重载该 profile。

## 边界

- A-OS 管：长期记忆、图、文件块、事实失效、巩固；
- DSH 管：当前轮、工具结果、plan、子代理、审批、UI；
- 插件只负责注入点和回写，不接管 DSH 的 compaction；
- `budget` 是记忆块上限，插件不把它算进 DSH 的 token 预算；后续需要在 DSH 侧扣减。
