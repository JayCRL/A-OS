"""依赖图：从 Python 源码抽 import 关系。

用途有两个：
1. 改了 A.py，引用它的文件对应的记忆页要失效（缓存一致性）
2. 读 A.py 时，顺带知道它依赖谁，少走弯路
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DepGraph:
    root: Path
    edges: dict[str, set[str]] = field(default_factory=dict)      # 文件 -> 依赖
    reverse: dict[str, set[str]] = field(default_factory=dict)    # 文件 -> 被谁依赖
    mtime: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(cls, root: Path) -> "DepGraph":
        graph = cls(root=root.resolve())
        graph.refresh()
        return graph

    def refresh(self) -> None:
        self.edges.clear()
        self.reverse.clear()
        for path in self.root.rglob("*.py"):
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            rel = str(path.relative_to(self.root))
            self.mtime[rel] = path.stat().st_mtime
            self.edges[rel] = self._imports(path)

        for src, deps in self.edges.items():
            for dep in deps:
                self.reverse.setdefault(dep, set()).add(src)

    def _imports(self, path: Path) -> set[str]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, OSError):
            return set()

        mods: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mods.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    mods.add(node.module.split(".")[0])

        resolved: set[str] = set()
        for mod in mods:
            candidate = mod.replace(".", "/") + ".py"
            if (self.root / candidate).exists():
                resolved.add(candidate)
            else:
                pkg = mod.replace(".", "/")
                if (self.root / pkg).is_dir():
                    resolved.add(pkg)
        return resolved

    # ---------- 查询 ----------

    def deps_of(self, rel: str) -> set[str]:
        return self.edges.get(rel, set())

    def dependents_of(self, rel: str) -> set[str]:
        return self.reverse.get(rel, set())

    def affected_by(self, rel: str) -> set[str]:
        """改了 rel，哪些文件受影响（直接 + 一层传递）。"""

        direct = self.dependents_of(rel)
        second = set()
        for item in direct:
            second |= self.dependents_of(item)
        return direct | second

    def changed_since_last_build(self) -> set[str]:
        changed: set[str] = set()
        for rel, ts in list(self.mtime.items()):
            path = self.root / rel
            if not path.exists():
                changed.add(rel)
            elif path.stat().st_mtime > ts:
                changed.add(rel)
        return changed

    def render(self, limit: int = 12) -> str:
        if not self.edges:
            return "（无依赖）"
        rows: list[str] = []
        for src in sorted(self.edges)[:limit]:
            deps = ", ".join(sorted(self.edges[src])) or "-"
            rows.append(f"{src} → {deps}")
        return "\n".join(rows)
