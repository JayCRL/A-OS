"""输入输出 spool：先把事件落盘，再由内核消费。

不是必需的缓存，而是把"接收"和"处理"解耦：
Web/CLI 只往 inbox 追加，进程完成后再往 outbox 追加。
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


class Spool:
    def __init__(self, home: str | Path) -> None:
        self.dir = Path(home) / "spool"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.inbox = self.dir / "inbox.jsonl"
        self.outbox = self.dir / "outbox.jsonl"
        self.in_count = 0
        self.out_count = 0
        self.in_bytes = 0
        self.out_bytes = 0
        self._restore_counts()

    def _restore_counts(self) -> None:
        for path, attr in ((self.inbox, "in"), (self.outbox, "out")):
            if not path.exists():
                continue
            count = 0
            for _ in path.open("r", encoding="utf-8"):
                count += 1
            if attr == "in":
                self.in_count = count
            else:
                self.out_count = count

    def _append(self, path: Path, record: dict) -> int:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        return len(line.encode("utf-8"))

    def input(self, text: str, *, channel: str = "cli") -> dict:
        record = {
            "id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "channel": channel,
            "text": text,
        }
        n = self._append(self.inbox, record)
        self.in_count += 1
        self.in_bytes += n
        return record

    def output(self, *, pid: int, channel: str, text: str) -> dict:
        record = {
            "id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "pid": pid,
            "channel": channel,
            "text": text,
        }
        n = self._append(self.outbox, record)
        self.out_count += 1
        self.out_bytes += n
        return record

    def stats(self) -> dict:
        return {
            "in": self.in_count,
            "out": self.out_count,
            "in_bytes": self.in_bytes,
            "out_bytes": self.out_bytes,
            "files": [str(self.inbox), str(self.outbox)],
        }
