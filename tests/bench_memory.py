"""A-OS 记忆召回基准：needle recall + 上下文命中 + 延迟。"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernel.memory import MemoryStore, _normalize  # noqa: E402
from services.pager import PagerService  # noqa: E402
from tests.fake_encoder import TokenEncoder  # noqa: E402

WORDS = ["青竹", "白露", "长风", "星野", "南乔", "北冥", "听雨", "观澜",
         "临江", "落霞", "孤鹜", "秋水", "春山", "夜航", "远帆", "归舟",
         "松间", "石上", "清泉", "明月"]


def run(size: int) -> dict:
    encoder = TokenEncoder()
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        store = MemoryStore(home / "brain.db", encoder)
        needle_at = [max(1, size * (i + 1) // (len(WORDS) + 1)) for i in range(len(WORDS))]
        needles = {pos: WORDS[i] for i, pos in enumerate(needle_at)}
        started = time.time()
        for i in range(size):
            if i in needles:
                text = f"第 {i} 条记录：用户的暗号是{needles[i]}。"
            else:
                text = f"第 {i} 条记录：用户在第 {i} 天做了一件普通的事。"
            store.add(text, kind="fact", importance=0.6, dedupe=False)
        add_s = time.time() - started

        pager = PagerService(home, encoder, bus=None)
        queries = []
        for pos, word in needles.items():
            queries.append((f"第 {pos} 条记录里的暗号是什么", word))

        flat_hits = 0
        hybrid_hits = 0
        context_hits = 0
        latencies = []
        context_tokens = []
        for query, word in queries:
            q = _normalize(encoder.encode_one(query))
            mat = store._matrix()
            ids = store._mat_ids
            scores = mat @ q
            top = sorted(zip(scores, ids), reverse=True)[:8]
            flat = [store.get(int(mid)) for _s, mid in top]
            flat_hits += int(any(m is not None and word in m.text for m in flat))

            t0 = time.time()
            hits = store.recall(query, k=8)
            latencies.append(time.time() - t0)
            hybrid_hits += int(any(word in m.text for _s, m in hits))

            out = asyncio.run(pager.recall_context(query, budget=2400))
            context_tokens.append(out["tokens"])
            context_hits += int(word in out["context"])

        pager.memory.close()
        return {
            "memories": size,
            "needles": len(queries),
            "flat_recall@8": f"{flat_hits}/{len(queries)}",
            "hybrid_recall@8": f"{hybrid_hits}/{len(queries)}",
            "context_recall@2400": f"{context_hits}/{len(queries)}",
            "avg_recall_ms": round(1000 * sum(latencies) / len(latencies), 2),
            "avg_context_tokens": round(sum(context_tokens) / len(context_tokens), 1),
            "add_seconds": round(add_s, 2),
        }


if __name__ == "__main__":
    for n in (1000, 5000):
        print(json.dumps(run(n), ensure_ascii=False))
