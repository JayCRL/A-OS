"""最小 MCP 客户端（stdio 传输）。

只实现内核需要的四步：initialize → tools/list → tools/call → shutdown。
消息走换行分隔的 JSON-RPC 2.0，这也是 MCP 的 stdio 约定。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class McpTool:
    name: str
    description: str
    schema: dict = field(default_factory=dict)


class McpServer:
    def __init__(self, name: str, command: list[str], *, cwd: str | None = None, timeout: float = 30.0):
        self.name = name
        self.command = command
        self.cwd = cwd
        self.timeout = timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self.tools: list[McpTool] = []

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self.command,
            cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "a-os", "version": "0.1"},
        })
        await self._notify("notifications/initialized", {})
        listed = await self._request("tools/list", {})
        for item in listed.get("tools", []):
            self.tools.append(McpTool(
                name=item.get("name", ""),
                description=item.get("description", ""),
                schema=item.get("inputSchema", {}),
            ))

    async def call(self, tool: str, arguments: dict) -> str:
        result = await self._request("tools/call", {"name": tool, "arguments": arguments})
        parts: list[str] = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(f"[{item.get('type')}]")
        return "\n".join(parts) or "（无返回）"

    async def stop(self) -> None:
        if not self._proc:
            return
        try:
            self._proc.terminate()
            await asyncio.wait_for(self._proc.wait(), timeout=3)
        except Exception:
            with_suppress = getattr(self._proc, "kill", None)
            if with_suppress:
                self._proc.kill()

    # ---------- JSON-RPC ----------

    async def _send(self, payload: dict) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write((json.dumps(payload) + "\n").encode())
        await self._proc.stdin.drain()

    async def _notify(self, method: str, params: dict) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _request(self, method: str, params: dict) -> dict:
        self._next_id += 1
        rid = self._next_id
        await self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        assert self._proc and self._proc.stdout
        while True:
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=self.timeout)
            if not line:
                raise RuntimeError(f"{self.name}: 连接断开")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") != rid:
                continue
            if "error" in msg:
                raise RuntimeError(f"{self.name}: {msg['error'].get('message')}")
            return msg.get("result", {})


class McpRegistry:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.servers: dict[str, McpServer] = {}

    async def load(self) -> list[str]:
        if not self.config_path.exists():
            return []
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        started: list[str] = []
        for name, item in (data.get("servers") or {}).items():
            server = McpServer(name, list(item["command"]), cwd=item.get("cwd"))
            try:
                await server.start()
            except Exception:
                continue
            self.servers[name] = server
            started.append(name)
        return started

    def catalog(self) -> str:
        if not self.servers:
            return "（没有挂载 MCP 服务）"
        lines: list[str] = []
        for name, server in self.servers.items():
            tools = ", ".join(t.name for t in server.tools[:12])
            lines.append(f"- {name}: {tools}")
        return "\n".join(lines)

    async def call(self, server: str, tool: str, args: dict) -> str:
        s = self.servers.get(server)
        if not s:
            return f"没有这个服务：{server}"
        try:
            return await s.call(tool, args)
        except Exception as exc:  # noqa: BLE001
            return f"MCP 调用失败：{exc}"

    async def stop_all(self) -> None:
        for server in self.servers.values():
            await server.stop()
