"""agent 进程：pager 自动把相关记忆/文件/图节点装进上下文。

进程不持有 PageTable，也不需要模型发缺页；只发 compose / state / journal IPC。
"""

from __future__ import annotations

import json
import re

from kernel import persona
from kernel.process import Process

_LEADING = re.compile(r"^\s*(A-OS|a-os|助手)\s*[:：]\s*")
MAX_STEPS = 8


def clean(text: str) -> str:
    answer = _LEADING.sub("", (text or "").strip())
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer or "嗯，我在。"


def parse_tool(text: str) -> tuple[str, dict] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    candidates = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if fence:
        candidates.append(fence.group(1))
    if raw.startswith("{") and raw.endswith("}"):
        candidates.append(raw)
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            candidates.append(line)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (obj.get("tool") or obj.get("action")):
            return str(obj.get("tool") or obj.get("action")), dict(obj)
    return None


class AgentProcess:
    def __init__(self, kernel) -> None:
        self.k = kernel
        self.bus = kernel.bus

    async def run(self, proc: Process) -> str:
        return await self.run_turn(proc, str(proc.payload.get("text", "")))

    async def run_turn(self, proc: Process, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "在呢，你说。"

        begin = await self.bus.call("pager", "begin_turn", pid=proc.pid, text=text)
        profile = begin.get("profile") or []
        await self.bus.call("pager", "prefetch", pid=proc.pid, query=text)
        goal = await self.bus.call("kernel", "state_add", pid=proc.pid,
                                   content=f"用户：{text[:70]}", kind="goal")
        task_id = (await self.bus.call("kernel", "journal_register", text=text,
                                       node_id=goal["id"]))["task_id"]

        trace: list[str] = []
        answer: str | None = None
        last_prompt = ""
        used_web = False
        web_nudged = False

        for step in range(1, MAX_STEPS + 1):
            dag = (await self.bus.call("kernel", "state_render", pid=proc.pid,
                                       limit=5))["text"]
            composed = await self.bus.call(
                "pager", "compose", pid=proc.pid, query=text, dag=dag,
                profile=profile, must_include=[], budget=self.k.budget,
            )
            last_prompt = composed["prompt"]
            if trace:
                last_prompt += "\n\n===【本轮进展】===\n" + "\n".join(trace[-4:])

            reply = await self.k.model.chat(
                [{"role": "user", "content": last_prompt}], max_tokens=1100
            )
            raw = reply.text

            tool = parse_tool(raw)
            if tool is not None:
                name, payload = tool
                args = payload.get("args") if isinstance(payload.get("args"), dict) else {
                    k: v for k, v in payload.items() if k not in ("tool", "action", "args")
                }
                result = await self.bus.call("vfs", "run", name=name, args=args)
                if len(result) > 4000:
                    result = result[:4000].rsplit("\n", 1)[0]
                if name in ("web_search", "web_fetch"):
                    used_web = True
                    if "出错了" in result or "没有返回结果" in result:
                        trace.append("[系统] 联网失败了，换关键词再搜一次；再不行再用 web_fetch 打开具体网页。")
                if name in ("write", "append"):
                    path = str(args.get("path", ""))
                    await self.bus.call("pager", "invalidate", source=path)
                await self.bus.call("kernel", "state_add", pid=proc.pid,
                                    content=f"{name} {args.get('path', '')}"[:70], kind="act")
                await self.bus.call("kernel", "journal_checkpoint", task_id=task_id,
                                    node_id=goal["id"], round=step, steps=step)
                trace.append(f"#{step} 工具 {name} 返回（优先使用）：{result[:1800]}")
                continue

            answer = clean(raw)
            refusal = ("不能联网", "没有联网", "无法联网", "不能搜索",
                       "没有搜索能力", "无法搜索", "没有这个能力", "不具备联网")
            if not used_web and not web_nudged and any(x in answer for x in refusal):
                web_nudged = True
                trace.append("[系统] 你有 web_search / web_fetch 工具，不要说不能联网；先调用工具查。")
                continue
            break

        if answer is None:
            answer = await self._force_answer(last_prompt)

        await self.bus.call("pager", "finish_turn", pid=proc.pid, user=text, answer=answer)
        say = await self.bus.call("kernel", "state_add", pid=proc.pid,
                                  content=f"回答：{answer[:70]}", kind="say")
        await self.bus.call("kernel", "state_prune", node_id=say["id"], pid=proc.pid)
        await self.bus.call("kernel", "journal_finish", task_id=task_id, status="done")
        await self.bus.call("librarian", "observe", user=text, answer=answer)
        return answer

    async def _force_answer(self, prompt: str) -> str:
        try:
            reply = await self.k.model.chat(
                [{"role": "user", "content": prompt + "\n\n[系统] 步数到了，不要再调用工具，直接用一两句话回答。"}],
                max_tokens=300,
            )
            return clean(reply.text)
        except Exception:
            return "我这边绕住了，你换个说法再问一次？"
