"""最小的 agent 进程：向 pager 要图节点，调模型，回答。

这一版还没有接 PageTable 的四段页，但进程、IPC、pager、librarian
已经是真实链路；下一步把 pager.resolve 换成 pager.fault + 段页映射。
"""

from __future__ import annotations

import re

from kernel import persona
from kernel.process import Process


class AgentProcess:
    def __init__(self, kernel) -> None:
        self.k = kernel

    async def run(self, proc: Process) -> str:
        text = str(proc.payload.get("text", "")).strip()
        if not text:
            return "在呢，你说。"
        nodes = await self.k.bus.call("pager", "resolve", query=text, k=5)
        context = "\n\n".join(item["text"] for item in nodes) or "（图里没有相关节点）"
        messages = [
            {"role": "system", "content": persona.system_prompt()},
            {"role": "user", "content": f"资料：\n{context}\n\n用户：{text}"},
        ]
        reply = await self.k.model.chat(messages, max_tokens=800)
        answer = _clean(reply.text)
        await self.k.bus.call("librarian", "observe", user=text, answer=answer)
        return answer


def _clean(text: str) -> str:
    answer = re.sub(r"^\s*(A-OS|a-os|助手)\s*[:：]\s*", "", (text or "").strip())
    return answer or "嗯，我在。"
