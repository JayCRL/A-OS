"""抽象与聚合：把 episodes 折叠成图节点。

- 原始聊天不建节点
- 失败试错进 trials
- 任务完成生成/更新一个 task 节点，产物和经验挂在属性上
- 同一模块的任务合并，不让节点数随聊天轮数增长
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from kernel.graph import GraphStore, Node, slug


@dataclass
class Episode:
    text: str
    ts: float = field(default_factory=time.time)
    module: str = ""
    task: str = ""
    success: bool = False
    lesson: str = ""
    artifact: str = ""
    attempts: int = 0
    action: str = ""
    result: str = ""
    depends_on: list[str] = field(default_factory=list)
    importance: float = 0.5


class Consolidator:
    """后台归并器。真实运行时由 LLM 抽取 module/task/lesson；这里先做确定性的折叠骨架。"""

    def __init__(
        self,
        graph: GraphStore,
        *,
        project: str = "main",
        project_title: str = "A-OS 项目",
        batch: int = 16,
        max_lessons: int = 3,
        max_nodes: int = 3000,
        max_per_module: int = 200,
        max_degree: int = 12,
    ) -> None:
        self.graph = graph
        self.project_key = f"project:{slug(project)}"
        self.batch = max(1, batch)
        self.max_lessons = max_lessons
        self.max_nodes = max_nodes
        self.max_per_module = max_per_module
        self.max_degree = max_degree
        self._buffer: list[Episode] = []
        self._flushes = 0
        self.skipped = 0
        self.folded = 0
        self.graph.add_node(
            self.project_key, "project", project_title,
            summary="项目主干", importance=1.0, merge_similar=False,
        )

    # ---------- 入口 ----------

    def ingest(self, episode: Episode) -> None:
        if not episode.module and not episode.task:
            self.skipped += 1
            return
        self._buffer.append(episode)
        if episode.success or len(self._buffer) >= self.batch:
            self.flush()

    def ingest_many(self, episodes: Sequence[Episode]) -> None:
        for episode in episodes:
            self.ingest(episode)

    def flush(self) -> dict:
        if not self._buffer:
            return {"tasks": 0}
        groups: dict[tuple[str, str], list[Episode]] = {}
        for episode in self._buffer:
            key = (episode.module.strip() or "misc", episode.task.strip() or episode.text[:40])
            groups.setdefault(key, []).append(episode)
        self._buffer = []
        for (module, task), items in groups.items():
            self._fold(module, task, items)
            self.folded += 1
        self._flushes += 1
        if self._flushes % 8 == 0:
            self.graph.enforce_caps(max_nodes=self.max_nodes,
                                    max_per_module=self.max_per_module,
                                    max_degree=self.max_degree)
        self._refresh_project()
        return {"tasks": len(groups)}

    # ---------- 折叠 ----------

    def _fold(self, module: str, task: str, episodes: list[Episode]) -> Node:
        module_key = f"module:{slug(module)}"
        module_node = self.graph.add_node(
            module_key, "module", module,
            summary=f"模块 {module}",
            parent=self.project_key,
            importance=0.85,
            merge_similar=False,
        )

        failures = [e for e in episodes if not e.success]
        success = next((e for e in episodes if e.success), None)
        lessons: list[str] = []
        for ep in episodes:
            if ep.lesson and ep.lesson not in lessons:
                lessons.append(ep.lesson)
        lessons = lessons[: self.max_lessons]

        artifact = success.artifact if success else ""
        attempts = max([e.attempts for e in episodes] + [len(failures)])
        summary = (success.text if success else episodes[-1].text)[:280]
        task_key = f"task:{slug(module)}:{slug(task)}"
        attrs = {
            "module": module,
            "status": "done" if success else "open",
            "attempts": attempts,
            "lessons": lessons,
        }
        if artifact:
            attrs["artifact"] = artifact
        node = self.graph.add_node(
            task_key, "task", task,
            summary=summary,
            attrs=attrs,
            parent=module_node,
            importance=0.75,
            merge_similar=False,
        )

        rows = [
            {
                "step": i + 1,
                "action": ep.action or ep.text[:80],
                "result": ep.result or ep.text[:160],
                "ok": ep.success,
                "elapsed": 0.0,
                "ts": ep.ts,
            }
            for i, ep in enumerate(failures)
        ]
        if rows:
            self.graph.add_trials(task_key, rows)

        self.graph.link(node, module_node, "modifies", 0.9)
        for dep in {d for ep in episodes for d in ep.depends_on if d}:
            self.graph.link(node, dep, "depends_on", 0.8)
        self.graph.link_similar(node)

        module_node.summary = _module_summary(self.graph, module_node, node)
        self.graph._persist(module_node)
        return node

    def _refresh_project(self) -> None:
        stats = self.graph.stats()
        project = self.graph.get(self.project_key)
        if project is None:
            return
        modules = stats["by_type"].get("module", 0)
        tasks = stats["by_type"].get("task", 0)
        project.summary = f"{modules} 个模块，{tasks} 个任务；试错 {stats['trials']} 次已折叠"
        self.graph._persist(project)


def _module_summary(graph: GraphStore, module: Node, latest: Node) -> str:
    tasks = [n for n in graph.children(module) if n.type == "task"]
    return f"{len(tasks)} 个任务；最近：{latest.title}"[:240]
