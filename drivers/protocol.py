"""动作协议：模型每次只输出一个 JSON 动作。

容忍 ```json 包裹、前后废话；解析失败就当 done，避免死循环。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

ACTION_NAMES = {"read", "write", "append", "list", "search", "shell", "mcp", "done"}

SYSTEM = """你是 A-OS 的内核进程，只能通过动作与世界交互。

每轮只输出一个 JSON 对象，不要有任何其他文字：
{"action": "read|write|list|search|shell|done", ...}

参数：
- read   {"action":"read","path":"相对路径"}
- write  {"action":"write","path":"相对路径","content":"内容"}
- append {"action":"append","path":"相对路径","content":"追加内容"}
- list   {"action":"list","path":"相对路径，可为 ."}
- search {"action":"search","keyword":"关键词"}
- shell  {"action":"shell","cmd":"命令"}   # 仅限白名单命令
- mcp    {"action":"mcp","server":"服务名","tool":"工具名","arguments":{}}
- done   {"action":"done","summary":"本轮结论"}

规则：
1. 路径一律相对工作根目录，禁止 .. 和绝对路径
2. 一次只做一个动作，拿到观察结果再决定下一步
3. 内容很长时分几次 append，不要一次写完被截断
4. 任务完成必须输出 done
5. 不要解释，不要 markdown，不要输出多个 JSON
"""

_JSON_RE = re.compile(r"\{.*?\}", re.S)


@dataclass
class Action:
    name: str
    args: dict

    @property
    def valid(self) -> bool:
        return self.name in ACTION_NAMES


def parse(text: str) -> Action:
    if not text:
        return Action("done", {"summary": "空回复"})

    block = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", block, re.S)
    if fence:
        block = fence.group(1)

    for candidate in (block, *_JSON_RE.findall(block)):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("action", "")).strip().lower()
        if name in ACTION_NAMES:
            return Action(name, obj)
        return Action("done", {"summary": f"未知动作 {name}"})

    return Action("done", {"summary": f"无法解析：{text[:120]}"})


def salvage(text: str) -> Action | None:
    """输出被截断时，尽量把 path 和 content 抠出来，转成 append。"""

    if not text:
        return None
    m = re.search(r'"action"\s*:\s*"(write|append)"', text)
    if not m:
        return None
    pm = re.search(r'"path"\s*:\s*"([^"]+)"', text)
    cm = re.search(r'"content"\s*:\s*"(.*)$', text, re.S)
    if not pm or not cm:
        return None
    body = cm.group(1)
    body = re.sub(r'"\s*\}\s*$', "", body)
    body = body.replace("\\n", "\n").replace('\\"', '"')
    return Action("append", {"path": pm.group(1), "content": body})
