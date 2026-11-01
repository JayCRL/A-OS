#!/usr/bin/env python3
"""最小 MCP stdio 服务，用来验证客户端协议。"""
import json, sys

TOOLS = [{
    "name": "echo",
    "description": "回显传入的文本",
    "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
}]

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    method = msg.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mock", "version": "1.0"}}})
    elif method == "notifications/initialized":
        continue
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": TOOLS}})
    elif method == "tools/call":
        name = msg["params"]["name"]
        args = msg["params"].get("arguments") or {}
        if name == "echo":
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "content": [{"type": "text", "text": f"echo: {args.get('text','')}"}]}})
        else:
            send({"jsonrpc": "2.0", "id": msg["id"], "error": {"message": "unknown tool"}})
    else:
        if "id" in msg:
            send({"jsonrpc": "2.0", "id": msg["id"], "error": {"message": "unsupported"}})
