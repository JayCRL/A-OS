"""pager 服务：缺页时把 query 解析成 3~5 个图节点。

内核只负责 trap 和 IPC；这里拥有 GraphStore / GraphRetriever，
是"策略"而不是"机制"。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from kernel.graph import GraphStore
from kernel.retriever import GraphRetriever


class PagerService:
    name = "pager"

    def __init__(self, home: str | Path, encoder) -> None:
        self.home = Path(home)
        self.encoder = encoder
        self.graph = GraphStore(self.home / "graph.db", encoder)
        self.retriever = GraphRetriever(self.graph, encoder)

    async def boot(self) -> None:
        # 谱坐标是后台任务；这里只做一次小规模预热，失败不影响启动
        if self.graph.stats()["active"] >= 3:
            try:
                await asyncio.to_thread(self.graph.spectral, k=8)
            except Exception:
                pass

    async def handle(self, method: str, params: dict) -> Any:
        if method == "resolve":
            query = str(params.get("query", "")).strip()
            k = int(params.get("k", 5))
            return self.retriever.context(query, k=k)
        if method == "explain":
            return self.retriever.explain(str(params.get("query", "")),
                                          k=int(params.get("k", 5)))
        if method == "render":
            node = self.graph.get(int(params.get("id", 0)))
            return self.graph.render(node) if node else ""
        if method == "spectral":
            coords = await asyncio.to_thread(self.graph.spectral, k=int(params.get("k", 8)))
            return {"nodes": len(coords)}
        if method == "stats":
            return self.graph.stats()
        if method == "ping":
            return "pong"
        raise ValueError(f"pager 没有方法：{method}")

    async def close(self) -> None:
        self.graph.close()
