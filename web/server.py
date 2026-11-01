"""A-OS Web 控制台：aiohttp，单页 + WebSocket 推送。

左：状态树；中：终端流；右：页表。所有数据来自内核快照，不额外维护状态。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import web, WSMsgType

PAGE = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


def make_app(aos, *, token: str = "") -> web.Application:
    app = web.Application()

    async def index(_req):
        return web.Response(text=PAGE, content_type="text/html")

    async def api_state(_req):
        return web.json_response(aos.snapshot())

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        task = asyncio.create_task(_push(ws, aos))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await _handle_command(aos, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            task.cancel()
        return ws

    async def inject(request):
        data = await request.json()
        text = str(data.get("text", "")).strip()
        if not text:
            return web.json_response({"ok": False, "error": "空输入"}, status=400)
        await aos.inject(text)
        return web.json_response({"ok": True})

    async def kill(request):
        victims = await aos.sched.preempt()
        return web.json_response({"ok": True, "killed": victims})

    app.router.add_get("/", index)
    app.router.add_get("/api/state", api_state)
    app.router.add_post("/api/inject", inject)
    app.router.add_post("/api/kill", kill)
    app.router.add_get("/ws", ws_handler)
    return app


async def _push(ws, aos) -> None:
    last = ""
    while not ws.closed:
        snap = aos.snapshot()
        payload = json.dumps(snap, ensure_ascii=False)
        if payload != last:
            await ws.send_str(payload)
            last = payload
        await asyncio.sleep(0.5)


async def _handle_command(aos, raw: str) -> None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return
    cmd = data.get("cmd")
    if cmd == "inject":
        await aos.inject(str(data.get("text", "")))
    elif cmd == "kill":
        await aos.sched.preempt()


async def serve(aos, host: str = "127.0.0.1", port: int = 8799) -> None:
    runner = web.AppRunner(make_app(aos))
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"A-OS console: http://{host}:{port}", flush=True)
    await asyncio.Event().wait()
