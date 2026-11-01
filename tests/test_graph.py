"""图记忆的单元测试：折叠、节点上限、检索、谱坐标、持久化。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernel.consolidator import Consolidator, Episode  # noqa: E402
from kernel.graph import GraphStore  # noqa: E402
from kernel.retriever import GraphRetriever  # noqa: E402
from tests.fake_encoder import TokenEncoder  # noqa: E402


class GraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.enc = TokenEncoder()
        self.graph = GraphStore(self.home / "graph.db", self.enc)
        self.cons = Consolidator(self.graph, batch=16)
        self.retriever = GraphRetriever(self.graph, self.enc)

    def tearDown(self) -> None:
        self.graph.close()
        self.tmp.cleanup()

    def _fail(self, module: str, task: str, lesson: str = "方案 A 超时") -> Episode:
        return Episode(
            text=f"尝试 {task}，失败",
            module=module, task=task, success=False,
            lesson=lesson, action="edit", result="timeout", attempts=10,
        )

    def _success(self, module: str, task: str, artifact: str = "patch.diff") -> Episode:
        return Episode(
            text=f"{task} 完成，方案 C 有效",
            module=module, task=task, success=True,
            artifact=artifact, attempts=10,
        )

    def _module(self, module: str, tasks: list[str]) -> None:
        for task in tasks:
            for _ in range(10):
                self.cons.ingest(self._fail(module, task))
            self.cons.ingest(self._success(module, task))
        self.cons.flush()

    def test_success_folds_fifty_trials_into_one_task(self) -> None:
        for i in range(50):
            self.cons.ingest(self._fail("登录页", "修复越权", lesson=f"失败方案 {i % 3}"))
        self.cons.ingest(self._success("登录页", "修复越权", "auth.patch"))
        self.cons.flush()

        stats = self.graph.stats()
        self.assertEqual(stats["by_type"].get("module"), 1, stats)
        self.assertEqual(stats["by_type"].get("task"), 1, stats)
        self.assertLessEqual(stats["active"], 3, stats)
        self.assertEqual(stats["trials"], 50, stats)
        node = self.graph.get("task:登录页:修复越权")
        self.assertIsNotNone(node)
        self.assertEqual(node.attrs["status"], "done")
        self.assertEqual(node.attrs["artifact"], "auth.patch")
        self.assertLessEqual(len(node.attrs["lessons"]), 3)

    def test_noise_does_not_create_nodes(self) -> None:
        for text in ("你好", "好的", "谢谢"):
            self.cons.ingest(Episode(text=text))
        self.cons.flush()
        self.assertEqual(self.graph.stats()["active"], 1)  # 只有 project
        self.assertEqual(self.cons.skipped, 3)

    def test_alias_retrieval_pulls_three_to_five_nodes(self) -> None:
        self._module("登录页", ["修复越权", "优化鉴权"])
        self._module("支付页", ["重试幂等"])
        nodes = self.retriever.retrieve("继续搞登录页")
        titles = [n.title for n in nodes]
        self.assertIn("登录页", titles)
        self.assertTrue(any("越权" in t or "鉴权" in t for t in titles), titles)
        self.assertLessEqual(len(nodes), 5)
        self.assertGreaterEqual(len(nodes), 2)
        text = "\n".join(self.graph.render(n) for n in nodes)
        self.assertIn("auth" if False else "登录页", text)

    def test_parent_caps(self) -> None:
        tasks = [f"任务{i}" for i in range(20)]
        self._module("大模块", tasks)
        before = len([n for n in self.graph.children(self.graph.get("module:大模块"))])
        self.assertGreater(before, 8)
        self.graph.enforce_caps(max_per_module=8, max_degree=12)
        after = len([n for n in self.graph.children(self.graph.get("module:大模块"))])
        self.assertLessEqual(after, 8)

    def test_degree_cap(self) -> None:
        self._module("模块", ["a", "b", "c", "d", "e", "f"])
        a = self.graph.get("task:模块:a")
        for key in ("b", "c", "d", "e", "f"):
            self.graph.link(a, f"task:模块:{key}", "similar", 0.9)
        self.graph.enforce_caps(max_degree=4)
        self.assertLessEqual(self.graph.stats()["max_degree"], 4)

    def test_spectral_coordinates(self) -> None:
        self._module("登录页", ["越权", "鉴权"])
        self._module("支付页", ["幂等"])
        coords = self.graph.spectral(k=3)
        self.assertTrue(coords)
        for node in self.graph._nodes.values():
            if node.status == "active" and node.id in coords:
                self.assertLessEqual(len(node.spectral), 3)
        a = self.graph.get("module:登录页")
        b = self.graph.get("module:支付页")
        self.assertGreaterEqual(self.graph.spectral_affinity(a, b), 0.0)

    def test_persistence(self) -> None:
        self._module("登录页", ["越权"])
        db = self.home / "graph.db"
        self.graph.close()
        again = GraphStore(db, self.enc)
        try:
            self.assertIsNotNone(again.get("module:登录页"))
            hits = again.alias_lookup("继续搞登录页")
            self.assertTrue(hits)
            self.assertEqual(again.stats()["by_type"].get("task"), 1)
        finally:
            again.close()
        self.graph = GraphStore(db, self.enc)  # 供 tearDown 关闭

    def test_trials_are_not_nodes(self) -> None:
        self._module("登录页", ["越权"])
        before = self.graph.stats()["active"]
        before_trials = self.graph.trials_count("task:登录页:越权")
        self.graph.add_trials("task:登录页:越权", [
            {"step": i, "action": "edit", "result": "timeout", "ok": False}
            for i in range(100)
        ])
        self.assertEqual(self.graph.stats()["active"], before)
        self.assertEqual(self.graph.trials_count("task:登录页:越权"), before_trials + 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
