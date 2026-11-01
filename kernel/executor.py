"""动作执行器。

所有路径限制在工作根目录内；写文件先写临时文件再原子替换，便于取消时不留半成品。
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_READ = 8000
MAX_OUTPUT = 4000


@dataclass
class Observation:
    ok: bool
    text: str

    def __str__(self) -> str:
        return ("OK " if self.ok else "ERR ") + self.text


class Executor:
    def __init__(self, root: Path, *, allow_shell: bool = False, guard=None,
                 sandbox=None, mcp=None) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.allow_shell = allow_shell
        self.guard = guard
        self.sandbox = sandbox
        self.mcp = mcp

    # ---------- 路径安全 ----------

    def _resolve(self, raw: str) -> Path:
        rel = (raw or ".").strip().lstrip("/")
        target = (self.root / rel).resolve()
        if not str(target).startswith(str(self.root)):
            raise PermissionError(f"越界：{raw}")
        return target

    # ---------- 执行 ----------

    async def run(self, name: str, args: dict) -> Observation:
        try:
            if self.guard is not None and not self.guard.allow(name):
                return Observation(False, "读配额已用完，请用 write 落盘")
            if name == "read":
                blocked = self.guard.blocks_read(args.get("path", "")) if self.guard else None
                if blocked:
                    return Observation(False, blocked)
                return self._read(args)
            if name == "write":
                return self._write(args)
            if name == "append":
                return self._append(args)
            if name == "list":
                blocked = self.guard.blocks_list(args.get("path", ".")) if self.guard else None
                if blocked:
                    return Observation(False, blocked)
                return self._list(args)
            if name == "search":
                return self._search(args)
            if name == "shell":
                return await self._shell(args)
            if name == "mcp":
                return await self._mcp(args)
            if name == "done":
                return Observation(True, str(args.get("summary", "")))
            return Observation(False, f"未知动作 {name}")
        except PermissionError as exc:
            return Observation(False, str(exc))
        except Exception as exc:  # noqa: BLE001
            return Observation(False, f"{type(exc).__name__}: {exc}")

    def _read(self, args: dict) -> Observation:
        path = self._resolve(args.get("path", ""))
        if path.is_dir():
            return Observation(True, f"{args.get('path')} 是目录，自动列出：\n{self._tree(path)}")
        if not path.is_file():
            return Observation(False, f"不存在：{args.get('path')}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text) > MAX_READ:
            text = text[:MAX_READ] + f"\n...（截断，共 {len(text)} 字符）"
        return Observation(True, f"{path.relative_to(self.root)}\n{text}")

    def _write(self, args: dict) -> Observation:
        target = self._resolve(args.get("path", ""))
        content = str(args.get("content", ""))
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".aos-tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)
        return Observation(True, f"写入 {target.relative_to(self.root)}（{len(content)} 字符）")

    def _append(self, args: dict) -> Observation:
        target = self._resolve(args.get("path", ""))
        content = str(args.get("content", ""))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(content + "\n")
        return Observation(True, f"追加 {target.relative_to(self.root)}（+{len(content)} 字符）")

    def _tree(self, path: Path, limit: int = 40) -> str:
        rows: list[str] = []
        for child in sorted(path.rglob("*")):
            if len(rows) >= limit:
                rows.append("...")
                break
            rel = child.relative_to(self.root)
            rows.append(("d " if child.is_dir() else "f ") + str(rel))
        return "\n".join(rows) or "（空）"

    def _list(self, args: dict) -> Observation:
        path = self._resolve(args.get("path", "."))
        if not path.exists():
            return Observation(False, f"不存在：{args.get('path')}")
        items = []
        for child in sorted(path.iterdir())[:100]:
            items.append(("d " if child.is_dir() else "f ") + child.name)
        return Observation(True, "\n".join(items) or "（空）")

    def _search(self, args: dict) -> Observation:
        keyword = str(args.get("keyword", "")).strip()
        if not keyword:
            return Observation(False, "缺少 keyword")
        hits: list[str] = []
        for path in self.root.rglob("*"):
            if len(hits) >= 8:
                break
            if not path.is_file() or path.stat().st_size > 1_000_000:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            idx = text.find(keyword)
            if idx < 0:
                continue
            snippet = text[max(0, idx - 120): idx + 300].replace("\n", " ")
            hits.append(f"{path.relative_to(self.root)}: {snippet}")
        return Observation(True, "\n".join(hits) or f"没有命中 {keyword}")

    async def _mcp(self, args: dict) -> Observation:
        if self.mcp is None:
            return Observation(False, "没有挂载 MCP")
        server = str(args.get("server", ""))
        tool = str(args.get("tool", ""))
        call_args = args.get("arguments") or {}
        if not isinstance(call_args, dict):
            return Observation(False, "arguments 必须是对象")
        text = await self.mcp.call(server, tool, call_args)
        return Observation(True, text[:MAX_OUTPUT])

    async def _shell(self, args: dict) -> Observation:
        if not self.allow_shell:
            return Observation(False, "shell 未启用")
        if self.sandbox is not None:
            cmd0 = str(args.get("cmd", "")).strip()
            if not cmd0:
                return Observation(False, "缺少 cmd")
            res = await self.sandbox.run(cmd0)
            return Observation(res.ok, res.output)
        cmd = str(args.get("cmd", "")).strip()
        if not cmd:
            return Observation(False, "缺少 cmd")
        proc = await asyncio.create_subprocess_shell(
            cmd, cwd=str(self.root),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=20)
        except asyncio.TimeoutError:
            proc.kill()
            return Observation(False, "命令超时")
        text = out.decode("utf-8", errors="ignore")[:MAX_OUTPUT]
        return Observation(proc.returncode == 0, text or "（无输出）")
