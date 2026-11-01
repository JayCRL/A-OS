"""记忆索引：把工作区文件切块、编码、落库，供缺页按语义寻址。

缺页时给一个文件名 → 直接读；
给一句自然语言（"那个算加法的函数"）→ 走这里做向量检索。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHUNK = 800
OVERLAP = 120
MAX_FILE = 200_000
TEXT_EXT = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".js", ".ts", ".html", ".css", ".sh", ".go", ".java"}


@dataclass
class Hit:
    path: str
    start: int
    text: str
    score: float


class MemoryIndex:
    def __init__(self, db_path: str | Path, encoder, root: Path) -> None:
        self.root = Path(root).resolve()
        self.encoder = encoder
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                path TEXT NOT NULL,
                start INTEGER NOT NULL,
                mtime REAL NOT NULL,
                text TEXT NOT NULL,
                vec BLOB NOT NULL,
                PRIMARY KEY (path, start)
            )
            """
        )
        self._conn.commit()
        self._dirty: set[str] = set()

    # ---------- 建立/更新 ----------

    def refresh(self, *, limit_files: int = 400) -> int:
        """扫一遍工作区，只重算变更过的文件。"""

        seen = 0
        for path in sorted(self.root.rglob("*")):
            if seen >= limit_files:
                break
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() not in TEXT_EXT or path.stat().st_size > MAX_FILE:
                continue
            rel = str(path.relative_to(self.root))
            mtime = path.stat().st_mtime
            row = self._conn.execute(
                "SELECT mtime FROM chunks WHERE path=? LIMIT 1", (rel,)
            ).fetchone()
            if row and abs(row[0] - mtime) < 0.5:
                continue
            self._index_file(rel, path, mtime)
            seen += 1
        self._conn.commit()
        return seen

    def _index_file(self, rel: str, path: Path, mtime: float) -> None:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return
        self._conn.execute("DELETE FROM chunks WHERE path=?", (rel,))
        chunks = self._split(text)
        if not chunks:
            return
        vecs = self.encoder.encode([c for _, c in chunks])
        for (start, body), vec in zip(chunks, vecs):
            self._conn.execute(
                "INSERT OR REPLACE INTO chunks (path, start, mtime, text, vec) VALUES (?,?,?,?,?)",
                (rel, start, mtime, body, vec.astype(np.float32).tobytes()),
            )

    @staticmethod
    def _split(text: str) -> list[tuple[int, str]]:
        if len(text) <= CHUNK:
            return [(0, text)] if text.strip() else []
        out: list[tuple[int, str]] = []
        pos = 0
        while pos < len(text):
            out.append((pos, text[pos: pos + CHUNK]))
            if pos + CHUNK >= len(text):
                break
            pos += CHUNK - OVERLAP
        return out

    # ---------- 检索 ----------

    def query(self, text: str, k: int = 3) -> list[Hit]:
        rows = self._conn.execute("SELECT path, start, text, vec FROM chunks").fetchall()
        if not rows:
            return []
        q = np.asarray(self.encoder.encode_one(text), dtype=np.float32)
        mat = np.frombuffer(b"".join(r[3] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        sims = mat @ q
        order = np.argsort(-sims)[:k]
        return [
            Hit(path=rows[i][0], start=rows[i][1], text=rows[i][2], score=float(sims[i]))
            for i in order
        ]

    def stats(self) -> dict[str, int]:
        n = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        files = self._conn.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()[0]
        return {"chunks": n, "files": files}

    def close(self) -> None:
        self._conn.close()
