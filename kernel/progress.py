"""看门狗：盯空转、重复和超步数。

它只负责"提醒 + 叫停"，不再强迫模型在第一次 write 后立刻 done。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProgressTracker:
    max_steps: int = 8
    repeat_threshold: int = 3
    steps: list[tuple[str, str]] = field(default_factory=list)
    faults: int = 0
    tools: int = 0
    _last_nudge: str | None = None

    def observe(self, name: str, args: dict, ok: bool = True) -> None:
        target = str(
            args.get("path") or args.get("query") or args.get("url")
            or args.get("keyword") or args.get("target") or ""
        )
        self.steps.append((name, target))
        if name == "fault":
            self.faults += 1
        elif name not in ("say", "answer"):
            self.tools += 1

    def exhausted(self, step: int) -> bool:
        return step >= self.max_steps

    def intervention(self) -> str | None:
        msg = self._intervention()
        if msg == self._last_nudge:
            return None
        self._last_nudge = msg
        return msg

    def _intervention(self) -> str | None:
        tail = self.steps[-self.repeat_threshold:]
        if len(tail) == self.repeat_threshold and len(set(tail)) == 1:
            name, target = tail[0]
            return (f"[系统中断] 「{target or name}」这一步已经重复 "
                    f"{self.repeat_threshold} 次，换目标或直接回答。")
        if len(self.steps) >= self.max_steps - 1:
            return "[系统中断] 步数快到上限了，现在直接用一句话回答用户。"
        return None

    def allow(self, name: str) -> bool:
        return True

    def ledger(self) -> str:
        return f"steps={len(self.steps)} faults={self.faults} tools={self.tools}"
