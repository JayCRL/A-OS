"""微内核骨架测试：启动、IPC、进程、pager、librarian、持久化。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drivers.model import ModelReply, Usage  # noqa: E402
from kernel.microkernel import Microkernel  # noqa: E402
from kernel.process import ProcState  # noqa: E402
from tests.fake_encoder import TokenEncoder  # noqa: E402


class ScriptModel:
    ready = True

    def __init__(self, answers):
        self.usage = Usage()
        self.answers = list(answers)
        self.prompts = []

    async def chat(self, messages, *, max_tokens=512, temperature=0.3):
        system = str(messages[0].get("content", ""))
        if "编目器" in system:
            return ModelReply('{"noise": true}', Usage(10, 5))
        self.prompts.append(messages)
        return ModelReply(self.answers.pop(0) if self.answers else "好的。", Usage(10, 5))


class EchoModel:
    ready = True

    def __init__(self) -> None:
        self.usage = Usage()
        self.prompts: list[list[dict]] = []

    async def chat(self, messages, *, max_tokens=512, temperature=0.3):
        self.prompts.append(messages)
        return ModelReply("看到图里的内容了。", Usage(10, 5))


class MicrokernelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.mk = Microkernel(self.home, model=None, encoder=TokenEncoder())
        await self.mk.boot()

    async def asyncTearDown(self) -> None:
        await self.mk.close()
        self.tmp.cleanup()

    async def _seed(self, module: str = "登录页", task: str = "修复越权") -> None:
        for i in range(10):
            await self.mk.bus.call(
                "librarian", "observe",
                module=module, task=task, success=False,
                lesson=f"方案{i % 3}超时", text=f"尝试 {task} 失败",
                action="edit", result="timeout", attempts=10,
            )
        await self.mk.bus.call(
            "librarian", "observe",
            module=module, task=task, success=True,
            artifact="auth.patch", text=f"{task} 完成",
            action="write", result="ok", attempts=10,
        )
        await self.mk.bus.call("librarian", "flush")

    async def test_boot_services_and_kernel_state(self) -> None:
        self.assertEqual(self.mk._state, "running")
        self.assertEqual(self.mk.bus.endpoints(), ["kernel", "librarian", "pager"])
        state = await self.mk.bus.call("kernel", "state")
        self.assertEqual(state["state"], "running")
        self.assertEqual(state["services"], ["kernel", "librarian", "pager"])
        self.assertEqual(await self.mk.bus.call("kernel", "ping"), "pong")
        self.assertEqual(await self.mk.bus.call("pager", "ping"), "pong")

    async def test_unknown_service_raises(self) -> None:
        with self.assertRaises(LookupError):
            await self.mk.bus.call("nope", "ping")

    async def test_process_lifecycle_and_pager_call(self) -> None:
        await self._seed()

        async def routine(proc, mk):
            ctx = await mk.bus.call("pager", "resolve", query="继续搞登录页", k=5)
            self.assertGreaterEqual(len(ctx), 2)
            self.assertTrue(any("登录页" in item["title"] for item in ctx), ctx)
            return {"pid": proc.pid, "nodes": len(ctx)}

        proc = await self.mk.spawn("agent", routine)
        result = await self.mk.wait(proc.pid)
        self.assertEqual(result["nodes"], len(await self.mk.bus.call(
            "pager", "resolve", query="继续搞登录页", k=5)))
        self.assertEqual(proc.state, ProcState.DONE, proc.error)
        states = await self.mk.bus.call("kernel", "processes")
        self.assertTrue(any(p["pid"] == proc.pid and p["state"] == "done" for p in states), states)

    async def test_librarian_folds_and_skips_noise(self) -> None:
        await self._seed()
        await self.mk.bus.call("librarian", "observe", user="你好", answer="你好呀")
        await self.mk.bus.call("librarian", "flush")
        stats = await self.mk.bus.call("librarian", "stats")
        self.assertEqual(stats["graph"]["by_type"].get("task"), 1, stats)
        self.assertEqual(stats["graph"]["trials"], 10, stats)
        self.assertGreaterEqual(stats["skipped"], 1, stats)
        node = self.mk.pager.graph.get("task:登录页:修复越权")
        self.assertIsNotNone(node)
        self.assertEqual(node.attrs["artifact"], "auth.patch")

    async def test_persistence_across_reboot(self) -> None:
        await self._seed()
        before = self.mk.pager.graph.stats()
        await self.mk.close()

        self.mk = Microkernel(self.home, model=None, encoder=TokenEncoder())
        await self.mk.boot()
        after = await self.mk.bus.call("pager", "stats")
        self.assertEqual(after["graph"]["by_type"].get("task"), before["by_type"].get("task"))
        ctx = await self.mk.bus.call("pager", "resolve", query="继续搞登录页", k=5)
        self.assertTrue(any("登录页" in item["title"] for item in ctx), ctx)

    async def test_ask_creates_agent_process_and_uses_pager(self) -> None:
        await self._seed()
        echo = EchoModel()
        self.mk.model = echo
        self.mk.librarian.model = echo
        answer = await self.mk.ask("继续搞登录页")
        self.assertEqual(answer, "看到图里的内容了。")
        prompt = echo.prompts[-1][-1]["content"]
        self.assertIn("登录页", prompt)
        self.assertIn("用户：继续搞登录页", prompt)
        stats = await self.mk.bus.call("kernel", "stats")
        self.assertGreaterEqual(stats["processes"].get("done", 0), 1, stats)

    async def test_page_fault_trap_loads_graph_pages(self) -> None:
        await self._seed()
        model = ScriptModel([
            "<|PAGE_FAULT: 继续搞登录页|>",
            "登录页有一个越权任务，已经完成。",
        ])
        self.mk.model = model
        self.mk.librarian.model = model
        self.mk.pager.memory.add("用户叫 JustProve", kind="identity")

        answer = await self.mk.ask("继续搞登录页")
        self.assertIn("越权", answer)
        second_prompt = model.prompts[-1][-1]["content"]
        self.assertIn("登录页", second_prompt)
        self.assertIn("JustProve", second_prompt)
        pages = self.mk.pager.mmu.stats()
        self.assertGreaterEqual(pages["segments"]["system"], 1, pages)
        self.assertGreaterEqual(pages["segments"]["data"], 1, pages)
        sources = [p.source for p in self.mk.pager.mmu._pages.values()]
        self.assertTrue(any(s.startswith("graph:") for s in sources), sources)
        self.assertEqual((await self.mk.bus.call("kernel", "processes"))[0]["state"], "done")

    async def test_state_tree_is_per_process(self) -> None:
        echo = EchoModel()
        self.mk.model = echo
        self.mk.librarian.model = echo
        await self.mk.ask("第一句话")
        await self.mk.ask("第二句话")
        procs = sorted(self.mk.table._procs.values(), key=lambda p: p.pid)
        first = (await self.mk.bus.call("kernel", "state_render", pid=procs[0].pid))["text"]
        second = (await self.mk.bus.call("kernel", "state_render", pid=procs[1].pid))["text"]
        self.assertIn("第一句话", first)
        self.assertNotIn("第二句话", first)
        self.assertIn("第二句话", second)

    async def test_bus_counts(self) -> None:
        await self.mk.bus.call("kernel", "ping")
        await self.mk.bus.call("pager", "ping")
        stats = await self.mk.bus.call("kernel", "stats")
        self.assertGreaterEqual(stats["bus"]["sent"], 2)
        self.assertGreaterEqual(stats["bus"]["services"]["kernel"]["processed"], 1)

    async def test_process_error_is_marked_killed(self) -> None:
        async def boom(proc, mk):
            raise RuntimeError("boom")

        proc = await self.mk.spawn("bad", boom)
        with self.assertRaises(RuntimeError):
            await self.mk.wait(proc.pid)
        self.assertEqual(proc.state, ProcState.KILLED)
        self.assertIn("boom", proc.error or "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
