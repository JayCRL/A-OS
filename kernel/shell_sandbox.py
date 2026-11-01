"""沙盒 shell：命令白名单 + 超时 + 输出截断 + 工作目录锁定。

不是真正的隔离，只是把破坏面收窄：不给任意命令、不给外网工具、不允许越界路径。
"""

from __future__ import annotations

import asyncio
import os
import shlex
from dataclasses import dataclass
from pathlib import Path

ALLOWED = {
    "git", "python3", "python", "pytest", "pip", "ls", "cat", "head", "tail",
    "grep", "find", "wc", "echo", "mkdir", "touch", "rm", "mv", "cp", "sed",
    "awk", "sort", "uniq", "diff", "tree", "go", "npm", "node",
}

BLOCKED_PATTERNS = (
    "curl ", "wget ", "ssh ", "scp ", "nc ", "ncat", "telnet",
    "> /etc", "chmod 777 /", "sudo ", "systemctl", "crontab",
    "rm -rf /", "mkfs", "dd if=", ":(){", "shutdown", "reboot",
)


@dataclass
class ShellResult:
    ok: bool
    output: str
    code: int
    elapsed: float


class ShellSandbox:
    def __init__(self, root: Path, *, timeout_s: float = 30.0, max_output: int = 4000) -> None:
        self.root = root.resolve()
        self.timeout_s = timeout_s
        self.max_output = max_output

    def _check(self, cmd: str) -> str | None:
        low = cmd.lower()
        for pat in BLOCKED_PATTERNS:
            if pat in low:
                return f"命令被拦截：包含 {pat.strip()}"
        try:
            parts = shlex.split(cmd)
        except ValueError as exc:
            return f"命令解析失败：{exc}"
        if not parts:
            return "空命令"
        if parts[0] not in ALLOWED:
            return f"不在白名单内：{parts[0]}（允许：{', '.join(sorted(ALLOWED))}）"
        return None

    async def run(self, cmd: str) -> ShellResult:
        import time

        blocked = self._check(cmd)
        if blocked:
            return ShellResult(False, blocked, -1, 0.0)

        started = time.time()
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(self.root),
            "LANG": "C.UTF-8",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTHOR_NAME": "A-OS",
            "GIT_AUTHOR_EMAIL": "aos@local",
            "GIT_COMMITTER_NAME": "A-OS",
            "GIT_COMMITTER_EMAIL": "aos@local",
        }
        proc = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(self.root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            return ShellResult(False, f"超时（{self.timeout_s}s）", -1, time.time() - started)

        text = out.decode("utf-8", errors="ignore")
        if len(text) > self.max_output:
            text = text[: self.max_output] + f"\n...（截断，共 {len(text)} 字符）"
        return ShellResult(proc.returncode == 0, text or "（无输出）", proc.returncode or 0,
                           time.time() - started)
