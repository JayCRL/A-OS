"""A-OS Web 端：一个聊天窗口，没有内核面板。

接口：
  GET  /              页面
  GET  /api/history   最近的对话与记忆计数
  POST /api/chat      发一句话，等一句回复
  GET  /api/debug     内部状态（页面默认不展示）
  WS   /ws            打字状态与回复推送
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import web, WSMsgType

PAGE = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


def make_app(agent) -> web.Application:
    app = web.Application()

    async def index(_req):
        return web.Response(text=PAGE, content_type="text/html")

    async def history(_req):
        return web.json_response({
            "turns": agent.history(limit=120),
            "memories": agent.memory.stats(),
        })

    async def chat(request):
        data = await request.json()
        text = str(data.get("text", "")).strip()
        if not text:
            return web.json_response({"ok": False, "error": "说点什么"}, status=400)
        try:
            reply = await agent.reply(text)
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, "reply": reply})

    async def memory_recall(request):
        data = await request.json()
        text = str(data.get("text", data.get("query", ""))).strip()
        if not text:
            return web.json_response({"ok": False, "error": "text 为空"}, status=400)
        try:
            budget = max(400, min(6000, int(data.get("budget", 2400))))
            out = await agent.recall_memory(text, budget=budget)
            return web.json_response({"ok": True, **out})
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def memory_commit(request):
        data = await request.json()
        user = str(data.get("user", "")).strip()
        answer = str(data.get("answer", "")).strip()
        session = str(data.get("session", "dsh")).strip() or "dsh"
        if not user:
            return web.json_response({"ok": False, "error": "user 为空"}, status=400)
        try:
            out = await agent.commit_memory_turn(session=session, user=user, answer=answer)
            return web.json_response({"ok": True, **out})
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def debug(_req):
        return web.json_response(agent.debug_snapshot())

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=25)
        await ws.prepare(request)
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                text = str(data.get("text", "")).strip()
                if not text:
                    continue
                await ws.send_json({"type": "typing", "on": True})
                try:
                    reply = await agent.reply(text)
                except Exception:  # noqa: BLE001
                    reply = "刚才走神了，你再说一遍？"
                await ws.send_json({"type": "typing", "on": False})
                await ws.send_json({"type": "reply", "text": reply})
        finally:
            pass
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/api/history", history)
    app.router.add_post("/api/chat", chat)
    app.router.add_get("/api/debug", debug)
    app.router.add_post("/api/memory/recall", memory_recall)
    app.router.add_post("/api/memory/commit", memory_commit)
    app.router.add_get("/ws", ws_handler)
    return app


async def serve(agent, host: str = "0.0.0.0", port: int = 8799) -> None:
    runner = web.AppRunner(make_app(agent))
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"A-OS: http://{host}:{port}", flush=True)
    await asyncio.Event().wait()
