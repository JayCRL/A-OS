"""一个月项目的合成压测：1 万轮 episode，看图和检索是否还轻。

运行：python3 tests/synthetic_graph.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernel.consolidator import Consolidator, Episode  # noqa: E402
from kernel.graph import GraphStore  # noqa: E402
from kernel.retriever import GraphRetriever  # noqa: E402
from tests.fake_encoder import TokenEncoder  # noqa: E402

MODULES = 50
TASKS_PER_MODULE = 4
FAILS_PER_TASK = 10
NOISE_PER_TASK = 39


def generate():
    for m in range(1, MODULES + 1):
        module = f"模块{m:02d}"
        for t in range(1, TASKS_PER_MODULE + 1):
            task = f"任务{m:02d}-{t}"
            for f in range(FAILS_PER_TASK):
                yield Episode(
                    text=f"尝试{task} 的方案{f % 3}，超时",
                    module=module, task=task, success=False,
                    lesson=f"方案{f % 3}不可行",
                    action=f"try-plan-{f % 3}", result="timeout", attempts=FAILS_PER_TASK,
                )
            for n in range(NOISE_PER_TASK):
                yield Episode(text="用户：继续", module="", task="")
            yield Episode(
                text=f"{task} 最终用方案C完成",
                module=module, task=task, success=True,
                artifact=f"patches/{module}/{task}.diff",
                action="write", result="ok", attempts=FAILS_PER_TASK,
            )


def main() -> int:
    tmp = tempfile.TemporaryDirectory()
    home = Path(tmp.name)
    enc = TokenEncoder()
    graph = GraphStore(home / "graph.db", enc)
    cons = Consolidator(graph, batch=16)
    retriever = GraphRetriever(graph, enc)

    total = MODULES * TASKS_PER_MODULE * (FAILS_PER_TASK + NOISE_PER_TASK + 1)
    started = time.time()
    counts: list[tuple[int, int]] = []
    for i, episode in enumerate(generate(), 1):
        cons.ingest(episode)
        if i % 1000 == 0:
            counts.append((i, graph.stats()["active"]))
    cons.flush()
    graph.enforce_caps(max_nodes=3000, max_per_module=200, max_degree=12)
    elapsed = time.time() - started
    stats = graph.stats()

    print(f"episodes={total}  active_nodes={stats['active']}  "
          f"edges={stats['edges']}  trials={stats['trials']}  "
          f"max_degree={stats['max_degree']}  time={elapsed:.1f}s")
    print("by_type:", stats["by_type"])
    print("growth:", "  ".join(f"{n}轮→{c}节点" for n, c in counts))
    print(f"nodes/episode={stats['active'] / total:.4f}")

    coords = graph.spectral(k=8)
    print(f"spectral_nodes={len(coords)} dim={len(next(iter(coords.values()))) if coords else 0}")

    hits = 0
    sizes = []
    for m in range(1, 21):
        module = f"模块{m:02d}"
        nodes = retriever.retrieve(f"继续搞{module}")
        sizes.append(len(nodes))
        titles = [n.title for n in nodes]
        if module in titles:
            hits += 1
        if m == 17:
            print(f"\n检索「继续搞{module}」→ {len(nodes)} 个节点，"
                  f"{sum(n.tokens for n in nodes)} tokens")
            for node in nodes:
                print(f"  [{node.type}] {node.title}  ← "
                      f"{graph.get(node.parent_id).title if node.parent_id else 'root'}")

    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        if not cond:
            ok = False
            print("FAIL:", msg)
        else:
            print("ok:", msg)

    check(stats["active"] <= 500, "图节点 ≤ 500")
    check(stats["by_type"].get("module") == MODULES, "模块节点 = 50")
    check(stats["by_type"].get("task") == MODULES * TASKS_PER_MODULE, "任务节点 = 200")
    check(stats["trials"] == MODULES * TASKS_PER_MODULE * FAILS_PER_TASK, "试错流水 = 2000")
    check(stats["max_degree"] <= 12, "最大度 ≤ 12")
    check(hits == 20, f"20 条「继续搞X」命中模块（实际 {hits}）")
    check(max(sizes) <= 5, f"每次最多换入 5 个节点（实际最大 {max(sizes)}）")
    check(bool(coords), "谱坐标已生成")
    check(stats["active"] / total < 0.05, "节点数不随轮数线性增长")

    graph.close()
    tmp.cleanup()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
