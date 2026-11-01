"""上下文净化：把日志式的原始 JSON 洗成自然语言状态。

模型很容易把 {"action":"write",...} 当成"该干什么"，而不是"干过什么"。
所以入库之前统一过一遍这里。
"""

from __future__ import annotations

import re

_CODE_FENCE = re.compile(r"```(?:json)?.*?```", re.S)
_ACTION_JSON = re.compile(r'\{[^{}]*?"action"\s*:\s*"[a-z]+".*?\}', re.S)
_LEADING_ACTION = re.compile(r"^\[?\s*(write|append|read|list|search|shell|mcp|done|fault)\s*[:：]\s*", re.I)
_WS = re.compile(r"[ \t\u3000]+")


def strip_json(text: str) -> str:
    """去掉代码块和带 action 的 JSON 片段。"""

    if not text:
        return ""
    out = _CODE_FENCE.sub(" ", text)
    out = _ACTION_JSON.sub(" ", out)
    out = out.replace('{"action"', " ").replace('"action":', " ")
    out = _LEADING_ACTION.sub("", out.strip())
    out = _WS.sub(" ", out)
    return out.strip()


def summarize(action: str, args: dict, ok: bool, observation: str) -> str:
    """把一次动作压成一句人话状态，写进 DAG 或页。"""

    path = str(args.get("path") or args.get("keyword") or "").strip()
    obs = strip_json(observation or "")[:160]

    if action == "write":
        n = len(str(args.get("content", "")))
        return f"已写入 {path}（{n} 字符）" if ok else f"写入 {path} 失败：{obs}"
    if action == "append":
        n = len(str(args.get("content", "")))
        return f"已追加到 {path}（+{n} 字符）" if ok else f"追加 {path} 失败：{obs}"
    if action == "shell":
        cmd = str(args.get("cmd", ""))[:40]
        return f"已执行命令 `{cmd}`（成功）" if ok else f"命令 `{cmd}` 失败：{obs}"
    if action == "list":
        return f"已列出目录 {path or '.'}：{obs[:80]}"
    if action == "search":
        return f"已检索「{path}」：{obs[:80]}"
    if action == "mcp":
        return f"已调用 {args.get('server')}.{args.get('tool')}：{obs[:80]}"
    if action == "fault":
        return f"已换入 {path}"
    if action == "done":
        return f"完成：{obs[:120]}"
    return f"{action}：{obs[:120]}" if obs else action


CODE_HINTS = (
    "代码", "文件", "函数", "接口", "改", "修", "写", "实现", "重构", "部署",
    "测试", "报错", "bug", "bugfix", "py", "java", "go", "js", "脚本", "sql",
    "目录", "仓库", "git", "编译", "依赖", "日志",
)


def needs_code(intent: str) -> bool:
    """判断这轮意图是不是要碰代码；不碰就该把 DATA 段清空。"""

    low = (intent or "").lower()
    return any(hint in low for hint in CODE_HINTS)
