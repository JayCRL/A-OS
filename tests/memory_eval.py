"""记忆容量/长间隔召回基线：1000 条事实里埋针，隔数百条再问。"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernel.memory import MemoryStore  # noqa: E402
from tests.fake_encoder import TokenEncoder  # noqa: E402


def main() -> None:
    needles = {50: "青竹", 500: "白露", 900: "长风"}
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "brain.db", TokenEncoder())
        start = time.time()
        for i in range(1000):
            text = f"第 {i} 条事实：用户在第 {i} 天做了一件普通的事。"
            if i in needles:
                text = f"第 {i} 条事实：用户的暗号是{needles[i]}。"
            store.add(text, kind="fact", importance=0.6, dedupe=False)
        elapsed = time.time() - start
        hits = 0
        for i, word in needles.items():
            result = store.recall(f"第 {i} 条事实里的暗号是什么", k=8)
            texts = [m.text for _score, m in result]
            hit = any(word in text and f"第 {i} 条" in text for text in texts)
            hits += int(hit)
            print(f"needle {i:>3} {word}: {'hit' if hit else 'MISS'}  top={len(texts)}")
        broad = [m.text for _score, m in store.recall("我的暗号是什么", k=8)]
        broad_hits = sum(1 for word in needles.values() if any(word in t for t in broad))
        print(f"recall={hits}/{len(needles)}  broad_recall={broad_hits}/{len(needles)}"
              f"  memories=1000  add={elapsed:.2f}s")


if __name__ == "__main__":
    main()
