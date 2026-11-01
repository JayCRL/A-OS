"""进度看门狗与动作台账。

三件事：
1. 记最近的动作，检测重复
2. 生成极简台账（给模型看它做过什么，不占上下文）
3. 读配额：只读不写超过阈值就催它落盘
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ProgressTracker:
    read_quota: int = 6
    repeat_threshold: int = 2
    steps: list[tuple[str, str]] = field(default_factory=list)
    read_paths: list[str] = field(default_factory=list)
    list_paths: list[str] = field(default_factory=list)
    writes: int = 0
    verified: bool = False
    _last_msg: str | None = None

    # ---------- 记录 ----------

    def observe(self, name: str, args: dict, ok: bool) -> None:
        target = str(args.get("path") or args.get("keyword") or "")
        self.steps.append((name, target))
        if not ok:
            return
        if name == "read":
            if target not in self.read_paths:
                self.read_paths.append(target)
            elif self.writes:
                self.verified = True
        if name == "list" and target not in self.list_paths:
            self.list_paths.append(target)
        if name == "write":
            self.writes += 1
            self.verified = False

    # ---------- 拦截 ----------

    def blocks_read(self, path: str) -> str | None:
        if path in self.read_paths and not self.writes:
            return "这个文件本轮已经读过，内容在盘外。请用 write 把结论落盘，别重复读"
        return None

    def blocks_list(self, path: str) -> str | None:
        return "项目结构已经常驻在上下文里，不用再 list。直接 read 或 write。"

    def read_quota_exceeded(self) -> bool:
        return len(self.read_paths) >= self.read_quota and self.writes == 0

    # ---------- 干预 ----------

    def intervention(self) -> str | None:
        msg = self._intervention()
        if msg == self._last_msg:
            return None
        self._last_msg = msg
        return msg

    def _intervention(self) -> str | None:
        if self.read_quota_exceeded():
            return ("[系统中断] 你已经读了 "
                    f"{len(self.read_paths)} 个文件却一个字没写。read/list 已禁用，"
                    "立刻用 write 输出成果，或输出 done。")

        if self.writes and self.verified:
            return "[系统中断] 文件已写完并自检过，现在必须输出 done，不要再读再列。"

        if self.writes and not self.verified:
            return "[系统中断] 已经开始产出。最多再 read 自检一次，然后必须 done。"

        tail = self.steps[-self.repeat_threshold:]
        if len(tail) == self.repeat_threshold and len(set(tail)) == 1:
            name, target = tail[0]
            return (f"[系统中断] 你连续 {self.repeat_threshold} 次在做同一件事："
                    f"{name} {target}。换一个动作，或直接 write / done。")
        return None

    def allow(self, name: str) -> bool:
        if name in ("read", "list") and self.read_quota_exceeded():
            return False
        return True

    # ---------- 台账 ----------

    def ledger(self) -> str:
        data = {
            "read": self.read_paths[-8:],
            "writes": self.writes,
            "steps": len(self.steps),
        }
        return json.dumps(data, ensure_ascii=False)
