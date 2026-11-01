"""librarian 服务：LLMExtractor + Consolidator。

它只做写入侧的"编目"，不在缺页路径上；模型慢、JSON 坏、任务失败，
都不应该影响 pager 和内核。
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from kernel.consolidator import Consolidator, Episode
from kernel.graph import GraphStore

SYSTEM = """你是 A-OS 的编目器。判断这一轮对话是否属于一个可归档的任务。

只输出一个 JSON 对象，不要解释：
{"noise": false, "module": "模块名", "task": "任务名", "success": false,
 "lesson": "一句经验", "artifact": "产物路径", "depends_on": []}

规则：
- 寒暄、问答、闲聊：{"noise": true}
- module 是项目里的模块/页面/文件，不是"用户"
- task 是一件事，不是一句话
- success 只在这件事确实完成时为 true
- lesson 只在有明确经验时写，没有就空字符串
- 不确定就 noise=true，宁可不记，不要造节点"""


class LLMExtractor:
    def __init__(self, model) -> None:
        self.model = model

    async def extract(self, user: str, answer: str) -> dict:
        if self.model is None:
            return {"noise": True}
        reply = await self.model.chat(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": f"用户：{user[:1200]}\n助手：{answer[:600]}"}],
            max_tokens=240, temperature=0.0,
        )
        raw = (reply.text or "").strip()
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return {"noise": True}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"noise": True}
        return data if isinstance(data, dict) else {"noise": True}


class LibrarianService:
    name = "librarian"

    def __init__(self, graph: GraphStore, *, model=None, project: str = "main") -> None:
        self.graph = graph
        self.model = model
        self.extractor = LLMExtractor(model)
        self.consolidator = Consolidator(graph, project=project)
        self.queue: asyncio.Queue = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self.processed = 0
        self.skipped = 0
        self.errors = 0

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="svc:librarian-worker")

    async def handle(self, method: str, params: dict) -> Any:
        if method == "observe":
            await self.queue.put(dict(params))
            return {"queued": self.queue.qsize()}
        if method == "flush":
            await self.queue.join()
            return self.stats()
        if method == "stats":
            return self.stats()
        raise ValueError(f"librarian 没有方法：{method}")

    def stats(self) -> dict:
        return {
            "queued": self.queue.qsize(),
            "processed": self.processed,
            "skipped": self.skipped,
            "errors": self.errors,
            "graph": self.graph.stats(),
        }

    async def _run(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                await self._ingest(item)
                self.processed += 1
            except asyncio.CancelledError:
                self.queue.task_done()
                raise
            except Exception:
                self.errors += 1
            finally:
                self.queue.task_done()

    async def _ingest(self, item: dict) -> None:
        if "module" in item or "task" in item:
            data = item
        else:
            data = await self.extractor.extract(
                str(item.get("user", "")), str(item.get("answer", ""))
            )
        if data.get("noise") or not (data.get("module") or data.get("task")):
            self.skipped += 1
            return
        episode = Episode(
            text=str(item.get("text") or item.get("user") or ""),
            module=str(data.get("module", "")),
            task=str(data.get("task", "")),
            success=bool(data.get("success", False)),
            lesson=str(data.get("lesson", "")),
            artifact=str(data.get("artifact", "")),
            attempts=int(data.get("attempts", 0) or 0),
            action=str(data.get("action", "")),
            result=str(data.get("result", "")),
            depends_on=list(data.get("depends_on") or []),
        )
        self.consolidator.ingest(episode)

    async def close(self) -> None:
        await self.queue.join()
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
