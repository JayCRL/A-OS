"""A-OS 对话运行时（内核版）。

外面是一句话进、一句话出；里面仍然是：
  调度器 → 状态树 → 段页式上下文 → 缺页中断 → 看门狗 → 记忆提取
用户看不到这些，调试面板才看得到。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Awaitable, Callable

from drivers.embedding import load_encoder
from drivers.model import DeepSeekKernel, OfflineKernel
from drivers.protocol import faults as parse_faults
from kernel import persona, sanitize
from kernel.context import wants_files
from kernel.dialogue import Dialogue
from kernel.journal import Journal
from kernel.kernel_scheduler import Job, KernelScheduler
from kernel.memory import MemoryStore
from kernel.page_table import PageTable, Segment
from kernel.progress import ProgressTracker
from kernel.state_dag import StateDAG
from kernel.tokens import estimate_tokens
from kernel.tools import Tools

log = logging.getLogger("aos")

TOOL_NAMES = {"list", "search", "write", "append", "web_search", "web_fetch"}
MAX_STEPS = 8
_LEADING_NAME = re.compile(r"^\s*(A-OS|a-os|助手)\s*[:：]\s*")


def parse_tool(text: str) -> tuple[str, dict] | None:
    """从模型输出里找工具调用；找不到就是正常回话。"""

    raw = (text or "").strip()
    if not raw:
        return None
    candidates: list[str] = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if fence:
        candidates.append(fence.group(1))
    if raw.startswith("{") and raw.endswith("}"):
        candidates.append(raw)
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            candidates.append(line)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("tool") or obj.get("action") or "").strip()
        if not name:
            continue
        args = obj.get("args") if isinstance(obj.get("args"), dict) else {
            k: v for k, v in obj.items() if k not in ("tool", "action", "args")
        }
        return name, dict(args)
    return None


def clean_answer(text: str) -> str:
    answer = _LEADING_NAME.sub("", (text or "").strip())
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer or "嗯，我在。"


class Agent:
    def __init__(
        self,
        home: str | Path = "/opt/aos/data",
        *,
        model=None,
        encoder=None,
        session: str = "main",
        budget: int | None = None,
        on_event: Callable[[str, str], Awaitable[None] | None] | None = None,
        debug: bool = False,
    ) -> None:
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "workspace").mkdir(exist_ok=True)
        (self.home / "logs").mkdir(exist_ok=True)

        self.encoder = encoder or load_encoder()
        self.memory = MemoryStore(self.home / "brain.db", self.encoder)
        self.dialogue = Dialogue(self.home / "brain.db", session=session)
        self.tools = Tools(self.home / "workspace", encoder=self.encoder,
                           on_write=self._on_file_written)

        # 内核
        self.mmu = PageTable(self.home / "pages.db", max_active=10)
        self.dag = StateDAG(self.home / "dag.db")
        self.sched = KernelScheduler(max_workers=3)
        self.journal = Journal(self.home / "tasks.db")
        self.guard_factory = ProgressTracker
        self._head = self.dag.head()

        self.budget = int(budget or os.environ.get("AOS_CTX_BUDGET", "8000"))
        if model is not None:
            self.model = model
        elif os.environ.get("AOS_MODEL", "auto") == "offline":
            self.model = OfflineKernel()
        else:
            self.model = DeepSeekKernel() if DeepSeekKernel().ready else OfflineKernel()

        self.on_event = on_event
        self.debug = debug
        self.session = session
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._events: list[dict] = []
        self._running = False
        self._sched_started = False
        self._last_stats: dict = {}
        self._turn_faults = 0
        self._last_guard: ProgressTracker | None = None
        self._setup_logging()

    # ---------- 生命周期 ----------

    def _setup_logging(self) -> None:
        log.setLevel(logging.DEBUG)
        log.propagate = False
        if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
            handler = logging.FileHandler(self.home / "logs" / "aos.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            log.addHandler(handler)
        for old in list(log.handlers):
            if not isinstance(old, logging.FileHandler):
                log.removeHandler(old)
        if self.debug:
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            log.addHandler(console)

    async def boot(self) -> None:
        self._running = True
        await self.sched.start()
        self._sched_started = True
        await asyncio.to_thread(self.tools._reindex)
        self._sanitize_dag()
        for rec in self.journal.recover():
            self.journal.finish(rec.task_id, "failed")
            log.info("恢复：%s 已中断，标记失败", rec.task_id)
        if self._head is None:
            self._head = self.dag.add("系统就绪", kind="boot")
        self._emit("boot", "ready")
        log.info("boot: pages=%s mem=%s turns=%s model=%s",
                 self.mmu.stats()["pages"], self.memory.stats()["total"],
                 self.dialogue.count(), "online" if getattr(self.model, "ready", False) else "offline")

    def _sanitize_dag(self) -> None:
        """老状态节点里可能混进了整段文件内容；只留第一行抽象状态。"""

        fixed = 0
        for node in list(self.dag._nodes.values()):
            first = sanitize.strip_json(node.content).splitlines()[0][:180] if node.content else ""
            if first.startswith(("read：", "read:")):
                first = "已读文件"
            elif node.kind == "act" and re.fullmatch(r"[\w./-]{1,80}", first):
                first = f"已读 {first}"
            if first and first != node.content:
                self.dag.update_content(node.node_id, first)
                fixed += 1
        if fixed:
            log.info("sanitized %s dag nodes", fixed)

    async def close(self) -> None:
        self._running = False
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._sched_started:
            await self.sched.stop()
        self.mmu.close()
        self.dag.close()
        self.journal.close()
        self.memory.close()
        self.dialogue.close()

    async def wait_idle(self) -> None:
        while self._tasks:
            pending = [t for t in self._tasks if not t.done()]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            else:
                await asyncio.sleep(0)
            self._tasks = {t for t in self._tasks if not t.done()}

    # ---------- 对外入口 ----------

    async def reply(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "在呢，你说。"

        async with self._lock:
            try:
                if self._sched_started:
                    answer = await self._dispatch_turn(text)
                else:
                    answer = await self._turn(text)
            except Exception as exc:  # noqa: BLE001
                log.exception("turn failed")
                answer = _friendly_error(exc)
            self._emit("assistant", answer)
            return answer

    async def _dispatch_turn(self, text: str) -> str:
        """每个用户回合 = 一个调度任务。"""

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()

        async def run() -> None:
            try:
                result = await self._turn(text)
                if not fut.done():
                    fut.set_result(result)
            except BaseException as exc:  # noqa: BLE001
                if not fut.done():
                    fut.set_exception(exc)

        async def on_cancel(_jid: str) -> None:
            if not fut.done():
                fut.set_exception(asyncio.CancelledError())

        self.sched.submit(Job(priority=0, seq=0, name="chat-turn", run=run, on_cancel=on_cancel))
        try:
            return await fut
        except asyncio.CancelledError:
            return "刚才被打断了，你再说一遍？"

    async def _turn(self, text: str) -> str:
        started = time.time()
        self._turn_faults = 0
        self.dialogue.add("user", text)

        node = self.dag.add(_abstract("用户", text), parent_id=self._head.node_id if self._head else None,
                            kind="goal")
        self._head = node
        task_id = self.journal.register(text, node.node_id)
        self.journal.claim(task_id)

        self._roll_pages()
        shown = text if len(text) <= 4000 else text[:4000] + "…（已截断）"
        self.mmu.add(f"用户：{shown}", segment=Segment.HUMAN, pinned=True,
                     tag="turn", importance=1.0)

        profile = self.memory.profile()
        # 只有用户明确提到"之前/上次/那个"时才翻闲聊片段；
        # 平时只带身份/偏好/事实/计划，避免把无关旧话塞进上下文。
        wants_past = any(w in text for w in ("上次", "之前", "聊过", "说过", "那天", "刚才", "那个"))
        kinds = None if wants_past else ("identity", "preference", "fact", "plan", "summary")
        recalled = [m for _, m in self.memory.recall(text, k=8, kinds=kinds)
                    if m.id not in {p.id for p in profile}]
        for mem in recalled:
            try:
                self.mmu.upsert(f"（长期记忆）{mem.text}", source=f"mem:{mem.id}",
                                vec=self._vec(mem.text), importance=min(0.9, mem.importance),
                                tag="memory", segment=Segment.DATA)
            except Exception:
                log.exception("memory page failed")

        hot: set[int] = set()
        guard = self.guard_factory(max_steps=MAX_STEPS)
        self._last_guard = guard
        trace: list[str] = []
        answer: str | None = None
        current_node = node

        for step in range(1, MAX_STEPS + 1):
            prompt = self._compose(text, profile, hot)
            if trace:
                prompt += "\n\n===【本轮进展】===\n" + "\n".join(trace[-4:])

            reply = await self.model.chat([{"role": "user", "content": prompt}], max_tokens=1100)
            raw = reply.text

            wanted = parse_faults(raw)
            if wanted:
                obs = await self._page_fault(wanted, hot)
                guard.observe("fault", {"target": wanted[0]}, True)
                current_node = self.dag.add(
                    sanitize.summarize("fault", {"path": wanted[0]}, True, obs),
                    parent_id=current_node.node_id, kind="fault",
                )
                self._head = current_node
                self.journal.checkpoint(task_id, node_id=current_node.node_id, round_no=step, steps=step)
                trace.append(f"#{step} 缺页换入「{wanted[0]}」→ {obs[:180]}")
                self._emit("fault", wanted[0])
                nudge = guard.intervention()
                if nudge:
                    trace.append(nudge)
                continue

            tool = parse_tool(raw)
            if tool is not None:
                name, args = tool
                if name not in TOOL_NAMES:
                    trace.append(f"#{step} 没有工具 {name}，换一种方式或直接回答。")
                    continue
                result = await self.tools.run(name, args)
                if len(result) > 2400:
                    result = result[:2400] + "…（已截断）"
                guard.observe(name, args, not result.startswith(("错误", "出错了")))
                current_node = self.dag.add(
                    sanitize.summarize(name, args, not result.startswith(("错误", "出错了")), result),
                    parent_id=current_node.node_id, kind="act",
                )
                self._head = current_node
                self.journal.checkpoint(task_id, node_id=current_node.node_id, round_no=step, steps=step)
                trace.append(f"#{step} {name} → {result[:220]}")
                self._emit("tool", name)
                nudge = guard.intervention()
                if nudge:
                    trace.append(nudge)
                continue

            answer = clean_answer(raw)
            break

        if answer is None:
            answer = await self._force_answer(text, profile, hot, trace)

        say = self.dag.add(f"回答：{answer[:70]}", parent_id=current_node.node_id, kind="say")
        self._head = say
        self.dag.prune(say.node_id)
        self.mmu.add(f"A-OS：{answer}", segment=Segment.DAG_STATE, pinned=True,
                     tag="answer", importance=0.7, source=f"turn:{task_id}")
        self.mmu.compact(tag="turn", keep=2)

        self.journal.finish(task_id)
        self.dialogue.add("assistant", answer)
        self._last_stats = {
            "elapsed": round(time.time() - started, 2),
            "tokens": getattr(getattr(self.model, "usage", None), "total", 0),
            "pages": self.mmu.stats()["active"],
            "faults": self._turn_faults,
            "steps": len(guard.steps),
            "nodes": self.dag.stats()["total"],
        }
        log.info("turn %.1fs %s", time.time() - started, self._last_stats)

        self._spawn(self._extract(text, answer))
        self._spawn(self._summarize())
        return answer

    async def _force_answer(self, text: str, profile, hot: set[int], trace: list[str]) -> str:
        prompt = self._compose(text, profile, hot)
        if trace:
            prompt += "\n\n===【本轮进展】===\n" + "\n".join(trace[-4:])
        prompt += ("\n\n[系统] 本轮步数到了，不要再调用工具、也不要再要页。"
                   "现在直接用一两句话回答用户；查不到就直说。")
        try:
            reply = await self.model.chat([{"role": "user", "content": prompt}], max_tokens=400)
            answer = clean_answer(reply.text)
            if parse_tool(answer) or "<|PAGE_FAULT" in answer:
                return "我这边绕住了，你换个说法再问一次？"
            return answer
        except Exception:
            return "我查到一半卡住了，你再说一遍？"

    # ---------- 上下文组装 ----------

    def _compose(self, text: str, profile, hot: set[int]) -> str:
        profile_text = ""
        if profile:
            profile_text = "关于用户：\n" + "\n".join(f"- {m.text}" for m in profile)
        dag_text = self.dag.render(limit=5)
        if wants_files(text):
            tree = self.tools.tree(limit=30)
            if tree:
                dag_text += "\n工作区：\n" + tree
        prompt, _ids = self.mmu.compose(
            query_vec=self._vec(text),
            query_text=text,
            budget_tokens=self.budget,
            segments={"system": persona.system_prompt(), "human": "", "dag":
                      (profile_text + ("\n最近状态：\n" + dag_text if dag_text else "")).strip()},
            must_include=hot,
            max_data=6,
        )
        return prompt

    def _roll_pages(self) -> None:
        """上一轮的人类指令降为最近状态；太旧的降为普通记忆。"""

        for page in list(self.mmu._pages.values()):
            if page.segment is Segment.HUMAN and page.tag == "turn":
                page.segment = Segment.DAG_STATE
                page.pinned = True
                self.mmu._persist(page)

        recent = sorted(
            (p for p in self.mmu._pages.values()
             if p.segment is Segment.DAG_STATE and p.state.value == "active"),
            key=lambda p: p.page_id,
        )
        for page in recent[:-6]:
            page.segment = Segment.DATA
            page.pinned = False
            self.mmu._persist(page)

        # 常驻页也要吃预算：最近状态最多留 2400 token，超出就把最旧的降级
        recent = recent[-6:]
        total = sum(p.tokens for p in recent)
        while len(recent) > 2 and total > 2400:
            page = recent.pop(0)
            page.segment = Segment.DATA
            page.pinned = False
            total -= page.tokens
            self.mmu._persist(page)
        self.mmu.compact(tag="turn", keep=2)

    # ---------- 缺页 ----------

    async def _page_fault(self, targets: list[str], hot: set[int]) -> str:
        messages: list[str] = []
        for raw_target in targets[:2]:
            target = raw_target.strip()
            if target.startswith("./"):
                target = target[2:]
            if not target:
                continue

            data = self.tools.read_page(target)
            if data is not None:
                rel, body = data
                page = self.mmu.find_by_source(rel)
                body = body[:2200]
                if page is None:
                    page = self.mmu.add(f"文件 {rel}：\n{body}", vec=self._vec(body[:800]),
                                        importance=0.95, tag="fault", source=rel,
                                        segment=Segment.DATA)
                else:
                    self.mmu.swap_in(page.page_id)
                    self.mmu.touch(page.page_id)
                hot.add(page.page_id)
                self._turn_faults += 1
                messages.append(f"换入 {rel}（页 #{page.page_id}，{estimate_tokens(body)} token）")
                continue

            existing = self.mmu.find_by_source(target)
            if existing is not None:
                self.mmu.swap_in(existing.page_id)
                self.mmu.touch(existing.page_id)
                hot.add(existing.page_id)
                messages.append(f"{target} 已在内存（页 #{existing.page_id}）")
                continue

            if ("." in target or "/" in target) and " " not in target:
                messages.append(f"没有这个文件：{target}")
                continue

            hits: list[tuple[float, str, str]] = []
            for hit in self.tools.semantic_hits(target, k=2):
                hits.append((hit["score"], hit["source"], hit["text"]))
            for score, mem in self.memory.recall(target, k=2):
                hits.append((score, f"mem:{mem.id}", f"（长期记忆）{mem.text}"))
            hits.sort(key=lambda x: x[0], reverse=True)

            added = 0
            for score, source, body in hits[:2]:
                if score < 0.45:
                    continue
                page = self.mmu.upsert(body[:1800], source=source, vec=self._vec(body[:800]),
                                       importance=0.9,
                                       tag="fault" if source.startswith("file:") else "memory",
                                       segment=Segment.DATA)
                hot.add(page.page_id)
                self._turn_faults += 1
                messages.append(f"语义寻址「{target}」→ {source}（{score:.2f}）页 #{page.page_id}")
                added += 1
            if not added:
                messages.append(f"寻址失败：{target}")

        return "；".join(messages) or "没有可换入的内容"

    # ---------- 记忆 ----------

    def _spawn(self, coro) -> None:
        if not self._running:
            return
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _extract(self, user: str, answer: str) -> None:
        try:
            self.memory.add(user, kind="episode", importance=0.4, source="chat")
        except Exception:
            log.exception("episode store failed")
        if len(user) < 3:
            return
        try:
            prompt = persona.EXTRACT.format(user=user[:1200], assistant=answer[:600])
            reply = await self.model.chat(
                [{"role": "system", "content": "你是记忆整理器，只输出 JSON。"},
                 {"role": "user", "content": prompt}],
                max_tokens=420, temperature=0.0,
            )
            items = _json_array(reply.text)
            if not items and "记住" in user:
                cleaned = user.replace("记住", "").replace("记得", "").strip("，。 ：:")
                if cleaned:
                    items = [{"text": cleaned, "kind": "fact", "importance": 0.85}]
            n = self.memory.add_many(items, source="chat")
            if n:
                log.info("memories +%s (%s)", n, self.memory.stats()["total"])
        except Exception:
            log.exception("extract failed")

    async def _summarize(self) -> None:
        try:
            if self.dialogue.count() < 24 or self.dialogue.count() % 8 != 0:
                return
            old = self.dialogue.oldest_unsummarized(keep_recent=16, limit=30)
            if not old:
                return
            text = "\n".join(f"{'用户' if t.role == 'user' else '助手'}：{t.text}" for t in old)[:4000]
            reply = await self.model.chat(
                [{"role": "system", "content": "你是对话归档器，只输出摘要正文。"},
                 {"role": "user", "content": persona.SUMMARY.format(dialogue=text)}],
                max_tokens=240, temperature=0.2,
            )
            summary = (reply.text or "").strip()
            if summary:
                self.memory.add(summary, kind="summary", importance=0.7, source="dialogue")
                self.dialogue.set_meta("summary_upto", str(max(t.id for t in old)))
                log.info("summarized %s turns", len(old))
        except Exception:
            log.exception("summarize failed")

    # ---------- 事件与调试 ----------

    def _emit(self, kind: str, text: str) -> None:
        event = {"kind": kind, "text": text, "ts": time.time()}
        self._events.append(event)
        del self._events[:-200]
        if self.on_event is None:
            return
        try:
            result = self.on_event(kind, text)
            if asyncio.iscoroutine(result):
                self._spawn(result)
        except Exception:
            log.exception("event handler failed")

    def debug_snapshot(self) -> dict:
        pages = sorted(self.mmu._pages.values(), key=lambda p: p.page_id, reverse=True)[:12]
        return {
            "memories": self.memory.stats(),
            "turns": self.dialogue.count(),
            "model": "online" if getattr(self.model, "ready", False) else "offline",
            "last": self._last_stats,
            "tools": self.tools.calls[-8:],
            "events": self._events[-40:],
            "kernel": {
                "nodes": self.dag.stats(),
                "pages": self.mmu.stats(),
                "scheduler": self.sched.stats(),
                "journal": self.journal.stats(),
                "faults": self._turn_faults,
                "steps": len(self._last_guard.steps) if self._last_guard else 0,
                "path": self.dag.render(limit=6),
            },
            "page_list": [
                {"id": p.page_id, "segment": p.segment.value, "state": p.state.value,
                 "tag": p.tag, "source": p.source, "tokens": p.tokens,
                 "preview": p.content[:60].replace("\n", " ")}
                for p in pages
            ],
        }

    def snapshot(self) -> dict:
        return {
            "dag": [], "pages": [], "log": [], "tokens": 0, "running": 0, "workers": 0,
            "locks": {}, "kernel": self.debug_snapshot()["kernel"],
            "turns": self.history(limit=30), "memories": self.memory.stats(),
        }

    def history(self, limit: int = 100) -> list[dict]:
        return [{"role": t.role, "text": t.text, "ts": t.ts} for t in self.dialogue.history(limit=limit)]

    def _on_file_written(self, rel: str) -> None:
        """文件变了 → 相关页失效，下一轮重新换入，避免拿旧内容回答。"""

        n = self.mmu.invalidate_source(rel)
        if n:
            log.info("invalidate %s pages for %s", n, rel)

    def _vec(self, text: str) -> list[float]:
        return self.encoder.encode_one(text)


def _abstract(who: str, text: str) -> str:
    return f"{who}：{text[:70].replace(chr(10), ' ')}"


def _friendly_error(exc: Exception) -> str:
    name = type(exc).__name__
    if "Timeout" in name or "URLError" in name or "Connection" in name:
        return "刚才网络有点慢，我没接上。你再说一遍？"
    if "api_key" in str(exc).lower() or "DEEPSEEK" in str(exc):
        return "我这边模型还没配好，先等等。"
    return "刚才走神了，你再说一遍？"


def _json_array(text: str) -> list[dict]:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.S)
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict) and str(x.get("text", "")).strip()][:5]
