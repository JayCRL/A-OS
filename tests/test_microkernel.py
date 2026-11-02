"""微内核骨架测试：启动、IPC、进程、pager、librarian、持久化。"""

from __future__ import annotations

import asyncio
import os
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


class FactModel:
    ready = True

    def __init__(self) -> None:
        self.usage = Usage()

    async def chat(self, messages, *, max_tokens=512, temperature=0.3):
        system = str(messages[0].get("content", ""))
        if "编目器" in system:
            return ModelReply(
                '{"noise": true, "facts": [{"text": "用户叫阿测", "kind": "identity", "importance": 0.95}]}',
                Usage(10, 10),
            )
        return ModelReply("好，记住了。", Usage(10, 5))


class ConflictModel:
    ready = True

    def __init__(self) -> None:
        self.usage = Usage()

    async def chat(self, messages, *, max_tokens=512, temperature=0.3):
        system = str(messages[0].get("content", ""))
        if "编目器" in system:
            return ModelReply(
                '{"noise": true, "facts": [{"text": "用户住在上海", "kind": "identity", "importance": 0.8}], '
                '"supersedes": ["用户住在北京"]}', Usage(10, 10),
            )
        return ModelReply("好，上海。", Usage(10, 5))


class SummaryModel:
    ready = True

    def __init__(self) -> None:
        self.usage = Usage()

    async def chat(self, messages, *, max_tokens=512, temperature=0.3):
        system = str(messages[0].get("content", ""))
        if "编目器" in system:
            return ModelReply(
                '{"noise": true, "facts": [{"text": "用户叫阿测", "kind": "identity", "importance": 0.9}]}',
                Usage(10, 10),
            )
        if "巩固器" in system:
            return ModelReply(
                '{"summary": "用户叫阿测，计划下周三去北京。", '
                '"facts": [{"text": "用户下周三去北京", "kind": "plan", "importance": 0.8}], '
                '"supersedes": [], "open_tasks": [], "topics": ["出行"]}',
                Usage(10, 10),
            )
        return ModelReply("好。", Usage(10, 5))


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
        self.assertEqual(self.mk.bus.endpoints(), ["kernel", "librarian", "pager", "vfs"])
        state = await self.mk.bus.call("kernel", "state")
        self.assertEqual(state["state"], "running")
        self.assertEqual(state["services"], ["kernel", "librarian", "pager", "vfs"])
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
        self.assertGreaterEqual(stats["processes"].get("total", 0), 1, stats)
        self.assertGreaterEqual(stats["processes"].get("waiting", 0), 1, stats)

    async def test_engineering_tools_read_edit_grep_shell(self) -> None:
        run = lambda name, **args: self.mk.bus.call("vfs", "run", name=name, args=args)
        out = await run("write", path="calc.py", content="def add(a, b):\n    return a + b\n")
        self.assertIn("已写入", out)
        text = await run("read", path="calc.py")
        self.assertIn("def add", text)
        found = await run("grep", query="return", path=".")
        self.assertIn("calc.py", found)
        edited = await run("edit", path="calc.py", old="a + b", new="a - b")
        self.assertIn("已替换", edited)
        self.assertIn("a - b", await run("read", path="calc.py"))
        matched = await run("glob", pattern="**/*.py")
        self.assertIn("calc.py", matched)
        shell = await run("shell", cmd='python3 -c "print(6*7)"')
        self.assertIn("42", shell)

    async def test_superseded_fact_not_recalled(self) -> None:
        await self.mk.bus.call("pager", "commit_memory", items=[
            {"text": "用户住在北京", "kind": "fact", "importance": 0.8},
        ])
        await self.mk.bus.call("pager", "supersede_facts", text="用户住在上海")
        await self.mk.bus.call("pager", "commit_memory", items=[
            {"text": "用户住在上海", "kind": "fact", "importance": 0.8},
        ])
        stats = await self.mk.bus.call("pager", "stats")
        self.assertGreaterEqual(stats["memory"].get("superseded", 0), 1, stats)
        hits = self.mk.pager.memory.recall("用户住在哪", k=8)
        texts = [m.text for _s, m in hits]
        self.assertFalse(any("北京" in t for t in texts), texts)
        self.assertTrue(any("上海" in t for t in texts), texts)

    async def test_conflict_supersedes_immediately(self) -> None:
        self.mk.pager.memory.add("用户住在北京", kind="identity", importance=0.8)
        model = ConflictModel()
        self.mk.model = model
        self.mk.librarian.model = model
        self.mk.librarian.extractor.model = model
        await self.mk.ask("我搬去上海了")
        await self.mk.bus.call("librarian", "flush")
        profile = [m.text for m in self.mk.pager.memory.profile()]
        self.assertTrue(any("上海" in text for text in profile), profile)
        self.assertFalse(any("北京" in text for text in profile), profile)
        stats = await self.mk.bus.call("pager", "stats")
        self.assertGreaterEqual(stats["memory"].get("superseded", 0), 1, stats)

    async def test_memory_events_and_consolidation(self) -> None:
        model = SummaryModel()
        self.mk.model = model
        self.mk.librarian.model = model
        self.mk.librarian.extractor.model = model
        self.mk.librarian.idle_seconds = 0.2
        self.mk.librarian.min_batch = 2
        await self.mk.ask("我叫阿测")
        await self.mk.ask("我下周三去北京")
        await asyncio.sleep(1.2)
        await self.mk.bus.call("librarian", "flush")
        lstats = await self.mk.bus.call("librarian", "stats")
        self.assertGreaterEqual(lstats["summaries"], 1, lstats)
        stats = await self.mk.bus.call("pager", "stats")
        self.assertGreaterEqual(stats["memory"]["by_kind"].get("summary", 0), 1, stats)
        events = await self.mk.bus.call("pager", "event_stats")
        self.assertGreaterEqual(events["total"], 2, events)

    async def test_facts_from_extractor_enter_profile(self) -> None:
        model = FactModel()
        self.mk.model = model
        self.mk.librarian.model = model
        await self.mk.ask("我叫阿测")
        await self.mk.bus.call("librarian", "flush")
        profile = [m.text for m in self.mk.pager.memory.profile()]
        self.assertTrue(any("阿测" in text for text in profile), profile)

    async def test_auto_prefetch_loads_graph_pages(self) -> None:
        await self._seed()
        model = EchoModel()
        self.mk.model = model
        self.mk.librarian.model = model
        self.mk.pager.memory.add("用户叫 JustProve", kind="identity")

        answer = await self.mk.ask("继续搞登录页")
        self.assertTrue(answer)
        prompt = model.prompts[-1][-1]["content"]
        self.assertIn("登录页", prompt)
        self.assertIn("JustProve", prompt)
        pages = self.mk.pager.mmu.stats()
        self.assertGreaterEqual(pages["segments"]["system"], 1, pages)
        self.assertGreaterEqual(pages["segments"]["data"], 1, pages)
        sources = [p.source for p in self.mk.pager.mmu._pages.values()]
        self.assertTrue(any(s.startswith("graph:") for s in sources), sources)
        self.assertIn((await self.mk.bus.call("kernel", "processes"))[0]["state"],
                      ("waiting", "running"))

    async def test_tool_call_goes_through_vfs(self) -> None:
        model = ScriptModel([
            '{"tool":"write","path":"note.txt","content":"dya 星期一回来"}',
            "写好了。",
        ])
        self.mk.model = model
        self.mk.librarian.model = model
        answer = await self.mk.ask("把 dya 星期一回来写进 note.txt")
        self.assertEqual(answer, "写好了。")
        self.assertTrue((self.home / "workspace" / "note.txt").exists())
        self.assertGreaterEqual(len(model.prompts), 2)
        spool = await self.mk.bus.call("kernel", "state")
        self.assertEqual(spool["spool"]["in"], 1)
        self.assertEqual(spool["spool"]["out"], 1)

    async def test_dma_loads_into_cache_and_index(self) -> None:
        (self.home / "workspace").mkdir(parents=True, exist_ok=True)
        (self.home / "workspace" / "plan.md").write_text("周三评审接口", encoding="utf-8")
        result = await self.mk.bus.call("vfs", "dma_load", path="plan.md")
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["handle"].startswith("dma:"))
        self.assertGreater(result["bytes"], 0)
        self.assertTrue((self.home / "dma").exists())

    async def test_state_tree_is_per_process(self) -> None:
        echo = EchoModel()
        self.mk.model = echo
        self.mk.librarian.model = echo
        await self.mk.ask("第一句话", channel="a")
        await self.mk.ask("第二句话", channel="b")
        procs = sorted(self.mk.table._procs.values(), key=lambda p: p.pid)
        first = (await self.mk.bus.call("kernel", "state_render", pid=procs[0].pid))["text"]
        second = (await self.mk.bus.call("kernel", "state_render", pid=procs[1].pid))["text"]
        self.assertIn("第一句话", first)
        self.assertNotIn("第二句话", first)
        self.assertIn("第二句话", second)

    async def test_auto_prefetch_loads_file_chunk(self) -> None:
        (self.home / "workspace").mkdir(parents=True, exist_ok=True)
        (self.home / "workspace" / "auth.md").write_text(
            "登录接口要保证幂等：重复提交只处理一次，使用请求号去重。\n",
            encoding="utf-8",
        )
        await self.mk.bus.call("vfs", "dma_scan")
        model = EchoModel()
        self.mk.model = model
        self.mk.librarian.model = model
        await self.mk.ask("登录接口怎么保证幂等")
        prompt = model.prompts[-1][-1]["content"]
        self.assertIn("幂等", prompt)
        pages = [p for p in self.mk.pager.mmu._pages.values() if p.tag == "file"]
        self.assertTrue(pages, "没有显式缺页时也应自动换入 VFS 文件块")
        self.assertTrue(any(p.source.startswith("file:") for p in pages))

    async def test_web_routes_use_microkernel(self) -> None:
        from aiohttp.test_utils import TestClient, TestServer
        from web.server import make_app

        model = EchoModel()
        self.mk.model = model
        self.mk.librarian.model = model
        client = TestClient(TestServer(make_app(self.mk)))
        await client.start_server()
        try:
            page = await client.get("/")
            self.assertEqual(page.status, 200)
            self.assertIn("A-OS", await page.text())
            r = await client.post("/api/chat", json={"text": "你好"})
            data = await r.json()
            self.assertTrue(data["ok"], data)
            self.assertEqual(data["reply"], "看到图里的内容了。")
            hist = await (await client.get("/api/history")).json()
            self.assertEqual(len(hist["turns"]), 2)
            dbg = await (await client.get("/api/debug")).json()
            self.assertIn("kernel", dbg)
        finally:
            await client.close()

    async def test_dispatcher_reuses_session_for_continuation(self) -> None:
        model = ScriptModel(["热搜列表：A、B、C", "好的，再刷一次"])
        self.mk.model = model
        self.mk.librarian.model = model
        await self.mk.ask("热搜有哪些")
        await self.mk.ask("要")
        self.assertEqual(len(self.mk.table._procs), 1)
        prompt = model.prompts[-1][-1]["content"]
        self.assertIn("热搜列表：A、B、C", prompt)
        self.assertIn("用户：要", prompt)
        self.assertEqual(self.mk.dispatcher.stats()["reused"], 1)

    async def test_recent_history_crosses_processes(self) -> None:
        model = ScriptModel(["第一轮：百度热搜有几条", "第二轮：好的，再刷一次"])
        self.mk.model = model
        self.mk.librarian.model = model
        await self.mk.ask("刚才热搜有哪些")
        await self.mk.ask("要")
        prompt = model.prompts[-1][-1]["content"]
        self.assertIn("第一轮：百度热搜有几条", prompt)
        self.assertIn("用户：要", prompt)

    async def test_web_refusal_gets_nudged_then_tool_runs(self) -> None:
        self.mk.vfs.tools._web_search = lambda query: f"搜索结果：{query} → 示例网站 https://example.com"
        model = ScriptModel([
            "我不能联网，也没有搜索能力。",
            '{"tool":"web_search","query":"武汉科技大学官网"}',
            "查到了，官网是 www.wust.edu.cn。",
        ])
        self.mk.model = model
        self.mk.librarian.model = model
        answer = await self.mk.ask("帮我搜索武汉科技大学官网")
        self.assertIn("wust.edu.cn", answer)
        joined = "\n".join(str(m.get("content", "")) for m in model.prompts[-1])
        self.assertIn("武汉科技大学官网", joined)
        tools = (await self.mk.bus.call("vfs", "stats"))["tools"]
        self.assertTrue(any(t["tool"] == "web_search" for t in tools), tools)

    async def test_many_idle_sessions_do_not_block(self) -> None:
        echo = EchoModel()
        self.mk.model = echo
        self.mk.librarian.model = echo
        for i in range(12):
            answer = await self.mk.ask(f"独立话题 {i}", channel=f"c{i}")
            self.assertEqual(answer, "看到图里的内容了。")
        self.assertGreaterEqual(len(self.mk.table._procs), 12)
        self.assertEqual(len(self.mk._session_tasks), 12)

    async def test_concurrent_asks_have_isolated_segments(self) -> None:
        echo = EchoModel()
        self.mk.model = echo
        self.mk.librarian.model = echo
        answers = await asyncio.gather(
            self.mk.ask("进程A的消息", channel="a"),
            self.mk.ask("进程B的消息", channel="b"),
        )
        self.assertEqual(answers, ["看到图里的内容了。", "看到图里的内容了。"])
        procs = sorted(self.mk.table._procs.values(), key=lambda p: p.pid)
        self.assertEqual(len(procs), 2)
        a, b = procs
        pages_a = [p for p in self.mk.pager.mmu._pages.values()
                   if p.pid == a.pid and p.segment.value == "human"]
        pages_b = [p for p in self.mk.pager.mmu._pages.values()
                   if p.pid == b.pid and p.segment.value == "human"]
        self.assertEqual(len(pages_a), 1)
        self.assertEqual(len(pages_b), 1)
        self.assertIn("进程A", pages_a[0].content)
        self.assertNotIn("进程B", pages_a[0].content)
        self.assertIn("进程B", pages_b[0].content)
        for proc in (a, b):
            stats = self.mk.pager.mmu.stats(pid=proc.pid)
            self.assertEqual(stats["segments"]["human"], 1, stats)
            self.assertEqual(stats["segments"]["system"], 1, stats)

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


class SplitServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self._old = {k: os.environ.get(k) for k in ("AOS_MODEL", "AOS_ENCODER")}
        os.environ["AOS_MODEL"] = "offline"
        os.environ["AOS_ENCODER"] = "hash"
        self.mk = Microkernel(self.home, split=True)
        await self.mk.boot()

    async def asyncTearDown(self) -> None:
        await self.mk.close()
        for key, value in self._old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    async def test_services_are_separate_processes(self) -> None:
        self.assertEqual(self.mk.bus.endpoints(), ["kernel", "librarian", "pager", "vfs"])
        self.assertEqual(len(self.mk._workers), 3)
        self.assertTrue(all(p.returncode is None for p in self.mk._workers))
        self.assertEqual(await self.mk.bus.call("vfs", "ping"), "pong")
        self.assertEqual(await self.mk.bus.call("pager", "ping"), "pong")
        self.assertEqual((await self.mk.bus.call("librarian", "stats"))["processed"], 0)

    async def test_remote_pager_librarian_vfs_flow(self) -> None:
        await self.mk.bus.call(
            "librarian", "observe",
            module="登录页", task="修复越权", success=True,
            artifact="auth.patch", lesson="方案C有效", text="完成",
        )
        await self.mk.bus.call("librarian", "flush")
        stats = await self.mk.bus.call("pager", "stats")
        self.assertEqual(stats["graph"]["by_type"].get("task"), 1, stats)
        answer = await self.mk.ask("继续搞登录页")
        self.assertIn("模型", answer)
        hist = self.mk.history()
        self.assertEqual(len(hist), 2)
        self.assertGreater(len(self.mk.debug_snapshot()["services"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
