"""A-OS 内核版验收测试。

不依赖 pytest：python3 -m unittest tests.test_chat -v
模型用 FakeModel，不花钱、不联网；另有一个 CLI 子进程测试保证用户看不到内核日志。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drivers.embedding import HashEncoder  # noqa: E402
from drivers.model import ModelReply, Usage  # noqa: E402
from kernel.agent import Agent, clean_answer, parse_tool  # noqa: E402
from kernel.memory import MemoryStore  # noqa: E402
from kernel.page_table import PageTable, Segment  # noqa: E402
from kernel.tokens import estimate_tokens  # noqa: E402


class FakeModel:
    """按规则回话，并记录每次收到的 prompt，方便断言"页真的换入了"。"""

    ready = True

    def __init__(self) -> None:
        self.usage = Usage()
        self.prompts: list[list[dict]] = []
        self.answers: list[str] = []

    async def chat(self, messages, *, max_tokens: int = 512, temperature: float = 0.3):
        system = str(messages[0].get("content", ""))
        body = "\n".join(str(m.get("content", "")) for m in messages)

        if "记忆整理器" in system:
            items = []
            if "dya" in body:
                items.append({"text": "dya 星期一回来", "kind": "plan", "importance": 0.9})
            if "香菜" in body:
                items.append({"text": "用户不吃香菜", "kind": "preference", "importance": 0.9})
            m = re.search(r"我叫([\u4e00-\u9fffA-Za-z]{1,8})", body)
            if m:
                items.append({"text": f"用户叫{m.group(1)}", "kind": "identity", "importance": 0.95})
            self.usage.prompt_tokens += 20
            return ModelReply(json.dumps(items, ensure_ascii=False), Usage(20, 8))

        if "对话归档器" in system:
            return ModelReply("用户和助手聊过一些日常和安排。", Usage(30, 10))

        self.prompts.append(messages)
        self.usage.prompt_tokens += 100
        if self.answers:
            return ModelReply(self.answers.pop(0), Usage(100, 20))

        if "dya" in body:
            return ModelReply("星期一，别忘。", Usage(100, 10))
        if "香菜" in body or "吃什么" in body:
            return ModelReply("那就避开香菜，吃点别的。", Usage(100, 10))
        return ModelReply("你好呀。", Usage(100, 10))

    def last_prompt(self) -> str:
        return "\n".join(str(m.get("content", "")) for m in self.prompts[-1])


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.enc = HashEncoder()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_add_recall_profile_dedupe(self) -> None:
        store = MemoryStore(self.home / "brain.db", self.enc)
        store.add("用户叫小王", kind="identity")
        store.add("用户不吃香菜", kind="preference")
        store.add("dya 星期一回来", kind="plan")
        store.add("用户在做毕业设计", kind="fact")

        profile = store.profile()
        self.assertTrue(any("小王" in m.text for m in profile))
        self.assertTrue(any("香菜" in m.text for m in profile))

        hits = [m for _, m in store.recall("dya 什么时候回来", k=5)]
        self.assertTrue(any("dya" in m.text for m in hits), hits)

        before = store.stats()["total"]
        store.add("dya 星期一回来", kind="plan")
        self.assertEqual(store.stats()["total"], before, "同一件事不应重复记")
        store.close()

    def test_persistence_and_scale(self) -> None:
        store = MemoryStore(self.home / "brain.db", self.enc)
        for i in range(300):
            store.add(f"第{i}条记忆：用户提到过项目编号 {i:04d} 的细节", kind="fact", dedupe=False)
        store.close()

        again = MemoryStore(self.home / "brain.db", self.enc)
        self.assertEqual(again.stats()["total"], 300)
        hits = [m.text for _, m in again.recall("项目编号 0123", k=3)]
        self.assertTrue(any("0123" in t for t in hits), hits)
        again.close()


class PageTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.enc = HashEncoder()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_segments_budget_and_eviction(self) -> None:
        mmu = PageTable(self.home / "pages.db", max_active=6)
        mmu.add("系统协议", segment=Segment.SYSTEM, importance=1.0)
        mmu.add("用户：帮我看下项目", segment=Segment.HUMAN, importance=1.0)
        mmu.add("状态：已换入 login.md", segment=Segment.DAG_STATE, importance=0.8)
        for i in range(30):
            mmu.add(f"文件片段 {i}：" + "上下文内容" * 20, segment=Segment.DATA,
                    importance=0.5, tag="fault", source=f"f{i}")
        prompt, ids = mmu.compose(
            query_vec=self.enc.encode_one("项目 文件片段 3"),
            query_text="项目 文件片段 3",
            budget_tokens=1500,
            segments={"system": "你是 A-OS", "human": "", "dag": "最近状态"},
            max_data=4,
        )
        self.assertLessEqual(estimate_tokens(prompt), 2200)
        stats = mmu.stats()
        self.assertEqual(stats["segments"]["system"], 1)
        self.assertEqual(stats["segments"]["human"], 1)
        self.assertGreaterEqual(stats["swapped"], 1)
        self.assertIn("系统段", prompt)
        self.assertIn("数据段", prompt)
        mmu.close()

    def test_persistence_and_persist_columns(self) -> None:
        mmu = PageTable(self.home / "pages.db")
        page = mmu.add("文件内容", segment=Segment.DATA, source="a.md", tag="fault")
        page.segment = Segment.DAG_STATE
        page.pinned = True
        page.stale = True
        mmu._persist(page)
        mmu.close()

        again = PageTable(self.home / "pages.db")
        loaded = again.get(page.page_id)
        self.assertEqual(loaded.segment, Segment.DAG_STATE)
        self.assertTrue(loaded.stale)
        self.assertTrue(loaded.pinned)
        again.close()


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.model = FakeModel()
        self.agent = Agent(self.home, model=self.model, encoder=HashEncoder(), budget=4000)
        await self.agent.boot()

    async def asyncTearDown(self) -> None:
        await self.agent.wait_idle()
        await self.agent.close()
        self.tmp.cleanup()

    async def test_chat_is_plain_and_memory_is_used(self) -> None:
        answer = await self.agent.reply("你好")
        self.assertEqual(answer, "你好呀。")
        self.assertNotIn("内核", answer)
        self.assertNotIn("[", answer)

        await self.agent.reply("跟你说个事，dya 星期一回来")
        await self.agent.wait_idle()
        kinds = self.agent.memory.stats()["by_kind"]
        self.assertGreaterEqual(kinds.get("plan", 0), 1, self.agent.memory.stats())

        reply = await self.agent.reply("dya 什么时候回来")
        self.assertEqual(reply, "星期一，别忘。")
        self.assertIn("dya 星期一回来", self.model.last_prompt())

    async def test_preference_survives_many_turns(self) -> None:
        await self.agent.reply("记住，我不吃香菜")
        await self.agent.wait_idle()
        for i in range(12):
            await self.agent.reply(f"随便聊聊第 {i} 句，今天有点忙")
        await self.agent.wait_idle()
        await self.agent.reply("中午吃什么好")
        self.assertIn("不吃香菜", self.model.last_prompt())

    async def test_restart_keeps_memory_and_pages(self) -> None:
        await self.agent.reply("我叫阿澈")
        await self.agent.reply("记住，dya 星期一回来")
        await self.agent.wait_idle()
        pages_before = self.agent.mmu.stats()["pages"]
        await self.agent.close()

        model2 = FakeModel()
        agent2 = Agent(self.home, model=model2, encoder=HashEncoder(), budget=4000)
        await agent2.boot()
        try:
            reply = await agent2.reply("dya 什么时候回来")
            self.assertEqual(reply, "星期一，别忘。")
            self.assertIn("阿澈", model2.last_prompt())
            self.assertGreaterEqual(agent2.mmu.stats()["pages"], pages_before)
        finally:
            await agent2.wait_idle()
            await agent2.close()

    async def test_long_conversation_stays_in_budget(self) -> None:
        for i in range(20):
            await self.agent.reply(f"第 {i} 轮：" + "我们聊一点很长的内容。" * 6)
        await self.agent.wait_idle()
        self.assertEqual(self.agent.dialogue.count(), 40)
        prompt_tokens = estimate_tokens(self.model.last_prompt())
        self.assertLess(prompt_tokens, 7000, f"上下文超预算：{prompt_tokens}")
        stats = self.agent.mmu.stats()
        self.assertGreaterEqual(stats["swapped"], 1, stats)


class KernelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.model = FakeModel()
        self.agent = Agent(self.home, model=self.model, encoder=HashEncoder(), budget=4000)
        await self.agent.boot()

    async def asyncTearDown(self) -> None:
        await self.agent.wait_idle()
        await self.agent.close()
        self.tmp.cleanup()

    async def test_page_fault_swaps_file_in(self) -> None:
        (self.home / "workspace").mkdir(parents=True, exist_ok=True)
        (self.home / "workspace" / "note.txt").write_text("dya 星期一回来", encoding="utf-8")
        self.model.answers = [
            "<|PAGE_FAULT: note.txt|>",
            "note.txt 里写着 dya 星期一回来。",
        ]
        answer = await self.agent.reply("note.txt 里写了什么")
        self.assertIn("星期一", answer)
        page = self.agent.mmu.find_by_source("note.txt")
        self.assertIsNotNone(page)
        self.assertEqual(page.tag, "fault")
        self.assertGreaterEqual(self.agent.mmu.stats()["faults"], 1)
        prompt = self.model.last_prompt()
        self.assertIn("note.txt", prompt)
        self.assertIn("dya 星期一回来", prompt)

    async def test_fault_dedup_and_file_invalidation(self) -> None:
        (self.home / "workspace").mkdir(parents=True, exist_ok=True)
        (self.home / "workspace" / "note.txt").write_text("第一版", encoding="utf-8")
        for expected in ("第一版", "第一版"):
            self.model.answers = ["<|PAGE_FAULT: note.txt|>", "看到了。"]
            await self.agent.reply("note.txt")
        pages = [p for p in self.agent.mmu._pages.values() if p.source == "note.txt"]
        self.assertEqual(len(pages), 1, "同一个 source 不应重复建页")

        self.agent.tools._write("note.txt", "第二版")
        page = self.agent.mmu.find_by_source("note.txt", include_stale=True)
        self.assertTrue(page.stale, "文件改过后旧页必须失效")

    async def test_state_tree_and_segments(self) -> None:
        await self.agent.reply("第一句话")
        await self.agent.reply("第二句话")
        stats = self.agent.dag.stats()
        self.assertGreaterEqual(stats["total"], 3, stats)
        seg = self.agent.mmu.stats()["segments"]
        self.assertGreaterEqual(seg["dag"], 1, seg)
        self.assertEqual(seg["human"], 1, seg)
        self.assertTrue(self.agent.dag.render(limit=10))

    async def test_watchdog_stops_repetition(self) -> None:
        self.model.answers = ["<|PAGE_FAULT: 不存在的东西|>"] * 8
        answer = await self.agent.reply("查一个不存在的东西")
        self.assertTrue(answer)
        guard = self.agent._last_guard
        self.assertIsNotNone(guard)
        self.assertLessEqual(len(guard.steps), 8)

    async def test_debug_snapshot_has_kernel(self) -> None:
        await self.agent.reply("你好")
        snap = self.agent.debug_snapshot()
        self.assertIn("kernel", snap)
        self.assertIn("nodes", snap["kernel"])
        self.assertIn("pages", snap["kernel"])
        self.assertIn("scheduler", snap["kernel"])


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.model = FakeModel()
        self.agent = Agent(self.home, model=self.model, encoder=HashEncoder(), budget=4000)
        await self.agent.boot()

    async def asyncTearDown(self) -> None:
        await self.agent.wait_idle()
        await self.agent.close()
        self.tmp.cleanup()

    async def test_write_then_fault_without_exposing_json(self) -> None:
        self.model.answers = [
            json.dumps({"tool": "write", "path": "note.txt", "content": "dya 星期一回来"}),
            "记好了。",
        ]
        answer = await self.agent.reply("把 dya 星期一回来写进 note.txt")
        self.assertEqual(answer, "记好了。")
        self.assertTrue((self.home / "workspace" / "note.txt").exists())

        self.model.answers = [
            "<|PAGE_FAULT: note.txt|>",
            "note.txt 里写着 dya 星期一回来。",
        ]
        answer = await self.agent.reply("note.txt 里是什么")
        self.assertIn("星期一", answer)
        self.assertNotIn('"tool"', answer)

    async def test_search_tool(self) -> None:
        (self.home / "workspace").mkdir(parents=True, exist_ok=True)
        (self.home / "workspace" / "a.md").write_text("毕业设计的题目是校园二手交易平台", encoding="utf-8")
        self.model.answers = [
            json.dumps({"tool": "search", "query": "毕业设计"}),
            "你写的是校园二手交易平台。",
        ]
        answer = await self.agent.reply("我毕业设计写的啥")
        self.assertIn("二手交易", answer)


class CliTests(unittest.TestCase):
    def test_cli_never_prints_kernel_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "AOS_HOME": tmp, "AOS_MODEL": "offline"}
            proc = subprocess.run(
                [sys.executable, str(ROOT / "main.py")],
                input="/memory\n你好\n/quit\n",
                capture_output=True, text=True, env=env, timeout=120,
            )
            out = proc.stdout + proc.stderr
            self.assertEqual(proc.returncode, 0, out[-2000:])
            self.assertIn("A-OS 在。", out)
            for noise in ("[清段]", "[退场]", "入队", "页表", "内核进程", "任务结束", "done →"):
                self.assertNotIn(noise, out, f"用户看到了内部日志：{noise}")


class WebTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from aiohttp.test_utils import TestClient, TestServer
        from web.server import make_app

        self.tmp = tempfile.TemporaryDirectory()
        self.agent = Agent(Path(self.tmp.name) / "home", model=FakeModel(),
                           encoder=HashEncoder(), budget=4000)
        await self.agent.boot()
        self.client = TestClient(TestServer(make_app(self.agent)))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()
        await self.agent.wait_idle()
        await self.agent.close()
        self.tmp.cleanup()

    async def test_page_and_chat(self) -> None:
        page = await self.client.get("/")
        html = await page.text()
        self.assertEqual(page.status, 200)
        self.assertIn("A-OS", html)
        self.assertNotIn('data-pane="pages"', html)
        self.assertIn("说点什么", html)

        r = await self.client.post("/api/chat", json={"text": "你好"})
        data = await r.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["reply"], "你好呀。")

        hist = await (await self.client.get("/api/history")).json()
        self.assertEqual(len(hist["turns"]), 2)

        dbg = await (await self.client.get("/api/debug")).json()
        self.assertIn("memories", dbg)
        self.assertIn("kernel", dbg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
