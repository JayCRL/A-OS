"""VFS / DMA 服务：工作区文件、网页、批量搬运。

agent 只通过 IPC 调工具；DMA 把整份文件搬进缓存和向量索引，
不把原始字节塞进模型上下文。
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any

from kernel.tools import Tools

TOOL_NAMES = {"read", "list", "glob", "grep", "search", "write", "append",
              "edit", "shell", "web_search", "web_fetch"}


class VFSService:
    name = "vfs"

    def __init__(self, home: str | Path, encoder) -> None:
        self.home = Path(home)
        self.workspace = self.home / "workspace"
        self.tools = Tools(self.workspace, encoder=encoder)
        self.dma_dir = self.home / "dma"
        self.dma_dir.mkdir(parents=True, exist_ok=True)
        self._scan: asyncio.Task | None = None
        self.dma = {"transfers": 0, "bytes": 0, "last": None}
        self.dma_calls: list[dict] = []

    async def boot(self) -> None:
        self._scan = asyncio.create_task(self._scan_workspace(), name="svc:vfs-dma-scan")

    async def _scan_workspace(self) -> None:
        try:
            await asyncio.to_thread(self.tools._reindex)
        except Exception:
            pass

    async def handle(self, method: str, params: dict) -> Any:
        if method == "run":
            name = str(params.get("name", ""))
            if name not in TOOL_NAMES:
                return f"错误：没有这个工具 {name}"
            return await self.tools.run(name, dict(params.get("args") or {}))
        if method == "tree":
            return self.tools.tree(limit=int(params.get("limit", 40)))
        if method == "semantic":
            query = str(params.get("query", ""))
            k = int(params.get("k", 3))
            return self.tools.semantic_hits(query, k=k)
        if method == "dma_load":
            return await self.dma_load(str(params.get("path", "")))
        if method == "dma_scan":
            await asyncio.to_thread(self.tools._reindex)
            return self.stats()
        if method == "stats":
            return self.stats()
        if method == "ping":
            return "pong"
        raise ValueError(f"vfs 没有方法：{method}")

    async def dma_load(self, path: str) -> dict:
        data = await asyncio.to_thread(self.tools.read_page, path)
        if data is None:
            return {"ok": False, "error": f"没有这个文件：{path}"}
        rel, text = data
        digest = hashlib.sha1(f"{rel}:{time.time()}".encode()).hexdigest()[:12]
        target = self.dma_dir / f"{digest}.txt"
        target.write_text(text, encoding="utf-8")
        await asyncio.to_thread(self.tools._reindex)
        n = len(text.encode("utf-8"))
        self.dma["transfers"] += 1
        self.dma["bytes"] += n
        self.dma["last"] = rel
        record = {"handle": f"dma:{digest}", "path": rel, "bytes": n}
        self.dma_calls.append(record)
        del self.dma_calls[:-20]
        return {"ok": True, **record}

    def stats(self) -> dict:
        try:
            index = self.tools.index.stats() if self.tools.index is not None else {}
        except Exception:
            index = {}
        return {
            "dma": dict(self.dma),
            "index": index,
            "last_calls": self.dma_calls[-5:],
            "tools": self.tools.calls[-10:],
        }

    async def close(self) -> None:
        if self._scan:
            self._scan.cancel()
            try:
                await self._scan
            except asyncio.CancelledError:
                pass
        if self.tools.index is not None:
            self.tools.index.close()
