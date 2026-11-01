"""Unix socket 上的最小 RPC：一行一个 JSON 请求/响应。

服务进程各自监听一个 .sock；内核用 RemoteBus 调它们。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


class UnixClient:
    def __init__(self, path: str | Path, *, timeout: float = 30.0) -> None:
        self.path = str(path)
        self.timeout = timeout
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()
        self.calls = 0
        self.errors = 0

    async def connect(self) -> None:
        if self._writer is not None:
            return
        self._reader, self._writer = await asyncio.open_unix_connection(self.path)

    async def wait_ready(self, timeout: float = 20.0) -> None:
        deadline = time.time() + timeout
        last: Exception | None = None
        while time.time() < deadline:
            try:
                await self.connect()
                return
            except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                last = exc
                await asyncio.sleep(0.05)
        raise TimeoutError(f"服务没起来：{self.path}（{last}）")

    async def call(self, method: str, *args: Any, timeout: float | None = None,
                   **params: Any) -> Any:
        # 兼容两种写法：call("method", ...) 和 call("service","method", ...)
        if args:
            method = str(args[0])
        async with self._lock:
            await self.connect()
            rid = uuid.uuid4().hex[:12]
            assert self._writer and self._reader
            payload = json.dumps({"id": rid, "method": method, "params": params},
                                 ensure_ascii=False) + "\n"
            self._writer.write(payload.encode("utf-8"))
            await self._writer.drain()
            line = await asyncio.wait_for(self._reader.readline(), timeout=timeout or self.timeout)
            if not line:
                raise ConnectionError(f"服务断开：{self.path}")
            data = json.loads(line)
            self.calls += 1
            if not data.get("ok"):
                self.errors += 1
                raise RuntimeError(data.get("error") or "远程服务错误")
            return data.get("result")

    async def close(self) -> None:
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        self._writer = None
        self._reader = None


class UnixServer:
    def __init__(self, path: str | Path, handler) -> None:
        self.path = str(path)
        self.handler = handler
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        self._server = await asyncio.start_unix_server(self._client, path=self.path)
        os.chmod(self.path, 0o600)

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            line = await reader.readline()
            if not line:
                break
            req = None
            try:
                req = json.loads(line)
                fn = getattr(self.handler, "handle", None)
                result = await (fn(req["method"], req.get("params") or {})
                                if fn else self.handler(req["method"], req.get("params") or {}))
                resp = {"id": req.get("id"), "ok": True, "result": result}
            except Exception as exc:  # noqa: BLE001
                resp = {"id": req.get("id") if req else None,
                        "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            writer.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    async def close(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


class RemoteBus:
    def __init__(self, local_handler) -> None:
        self.local = local_handler
        self.clients: dict[str, UnixClient] = {}
        self.calls = 0
        self.errors = 0

    def register(self, name: str, client: UnixClient) -> None:
        self.clients[name] = client

    def endpoints(self) -> list[str]:
        return sorted(["kernel", *self.clients])

    async def call(self, dst: str, method: str, *, timeout: float = 30.0, **params: Any) -> Any:
        if dst == "kernel":
            self.calls += 1
            fn = getattr(self.local, "handle", self.local)
            return await fn(method, params)
        client = self.clients.get(dst)
        if client is None:
            raise LookupError(f"没有这个服务：{dst}")
        try:
            self.calls += 1
            return await client.call(method, timeout=timeout, **params)
        except Exception:
            self.errors += 1
            raise

    def stats(self) -> dict:
        return {
            "sent": self.calls,
            "failed": self.errors,
            "services": {name: {"calls": c.calls, "errors": c.errors}
                         for name, c in self.clients.items()},
        }

    async def stop(self) -> None:
        for client in self.clients.values():
            await client.close()
