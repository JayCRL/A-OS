"""会话调度器：判断一条输入属于哪个 agent 进程。

规则优先，避免每条消息都新建 pid：
- 同通道、最近活跃，且出现"要/继续/那个/？"这类指代 → 复用
- 和上一轮文本有足够字面重合 → 复用
- 否则新建会话进程
"""

from __future__ import annotations

import asyncio
import math
import re
import time
import uuid
from dataclasses import dataclass, field

CONTINUE = (
    "要", "好", "行", "可以", "嗯", "对", "是", "不是", "继续", "然后",
    "再", "这个", "那个", "刚才", "上面", "前面", "它", "他", "她",
    "？", "?", "还有", "展开", "详细", "说说",
)


def _grams(text: str) -> set[str]:
    low = (text or "").lower()
    chars = re.sub(r"\s+", "", low)
    grams = {chars[i: i + 2] for i in range(max(0, len(chars) - 1))}
    grams.update(re.findall(r"[a-z0-9_]{2,}", low))
    return grams


def _sim(a: str, b: str) -> float:
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / math.sqrt(len(ga) * len(gb))


@dataclass
class Session:
    id: str
    channel: str
    title: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    pid: int | None = None
    created: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    last_text: str = ""
    turns: int = 0
    idle_seconds: float = 1800.0

    def expired(self, now: float | None = None) -> bool:
        return (now or time.time()) - self.last_active > self.idle_seconds


class Dispatcher:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.routed = 0
        self.reused = 0
        self.created = 0

    def route(self, text: str, *, channel: str = "cli") -> tuple[Session, bool]:
        """返回 (session, is_new)。"""

        now = time.time()
        active = [s for s in self.sessions.values()
                  if s.channel == channel and not s.expired(now)]
        if active:
            last = max(active, key=lambda s: s.last_active)
            cue = any(c in text for c in CONTINUE)
            if cue or _sim(text, last.last_text) >= 0.18:
                self.routed += 1
                self.reused += 1
                return last, False

        session = Session(
            id=uuid.uuid4().hex[:10],
            channel=channel,
            title=text[:40],
        )
        self.sessions[session.id] = session
        self.routed += 1
        self.created += 1
        return session, True

    def touch(self, session: Session, text: str, answer: str = "") -> None:
        session.last_active = time.time()
        session.turns += 1
        session.last_text = f"{text} {answer}"[:1000]
        if session.turns == 1:
            session.title = text[:40]

    def expired(self) -> list[Session]:
        return [s for s in self.sessions.values() if s.expired()]

    def drop(self, session: Session) -> None:
        self.sessions.pop(session.id, None)

    def stats(self) -> dict:
        return {
            "sessions": len(self.sessions),
            "routed": self.routed,
            "reused": self.reused,
            "created": self.created,
            "list": [
                {"id": s.id, "pid": s.pid, "channel": s.channel,
                 "turns": s.turns, "title": s.title,
                 "idle": round(time.time() - s.last_active, 1)}
                for s in sorted(self.sessions.values(),
                                key=lambda x: x.last_active, reverse=True)[:20]
            ],
        }
