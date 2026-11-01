"""图检索：别名/向量锚点 → 骨架路径 → 局部扩跳 → 谱特征重排。

目标不是"把所有相关节点都拉出来"，而是每次只换入 3~5 个最值得看的节点。
"""

from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np

from kernel.graph import GraphStore, Node


def _cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or len(a) != len(b):
        return 0.0
    return max(0.0, float(np.dot(a, b)))


class GraphRetriever:
    def __init__(self, graph: GraphStore, encoder, *, default_k: int = 5,
                 max_tokens: int = 900) -> None:
        self.graph = graph
        self.encoder = encoder
        self.default_k = default_k
        self.max_tokens = max_tokens

    def retrieve(self, query: str, *, k: int | None = None) -> list[Node]:
        query = (query or "").strip()
        if not query:
            return []
        k = k or self.default_k
        qvec = np.asarray(self.encoder.encode_one(query), dtype=np.float32)

        anchors = [node for score, node in self.graph.alias_lookup(query, limit=2)]
        if not anchors:
            # 没有明确别名时才退化为纯向量种子；有别名就不要让近邻模块污染锚点
            anchors = [node for _score, node in self.graph.vector_search(qvec, k=4)]
        if not anchors:
            return []

        base: dict[int, float] = {}

        def bump(node: Node, score: float) -> None:
            if node.status != "active":
                return
            base[node.id] = max(base.get(node.id, 0.0), score)

        for rank, anchor in enumerate(anchors[:2]):
            bump(anchor, 2.2 - rank * 0.2)
            chain = self.graph.parent_chain(anchor)
            for depth, node in enumerate(reversed(chain[:-1])):
                bump(node, 1.1 / (depth + 1))
            for child in self.graph.children(anchor)[:8]:
                bump(child, 0.95)
            for other, _kind, weight in self.graph.neighbors(anchor, limit=10):
                bump(other, 0.7 * weight)
                for grand, _k2, w2 in self.graph.neighbors(other, limit=4):
                    bump(grand, 0.25 * weight * w2)

        if not base:
            return []

        first = anchors[0]
        now = time.time()
        scored: list[tuple[float, Node]] = []
        for nid, prior in base.items():
            node = self.graph.get(nid)
            if node is None:
                continue
            sim = _cos(qvec, node.vec)
            age_days = max(0.0, (now - node.updated) / 86400.0)
            recency = 1.0 / (1.0 + math.log1p(age_days / 14.0))
            spectral = self.graph.spectral_affinity(first, node)
            score = (prior + 0.35 * sim + 0.15 * node.importance
                     + 0.05 * recency + 0.10 * spectral)
            scored.append((score, node))

        scored.sort(key=lambda x: -x[0])

        # MMR：别一口气给五条同模块同义任务
        picked: list[Node] = []
        picked_vecs: list[np.ndarray] = []
        used = 0
        for score, node in scored:
            if len(picked) >= k:
                break
            if node.tokens + used > self.max_tokens and picked:
                continue
            if node.vec is not None and picked_vecs:
                overlap = max(_cos(node.vec, v) for v in picked_vecs)
                mmr = 0.8 * score - 0.2 * overlap
            else:
                mmr = score
            picked.append(node)
            used += node.tokens
            if node.vec is not None:
                picked_vecs.append(node.vec)
        return picked

    def context(self, query: str, *, k: int | None = None) -> list[dict]:
        nodes = self.retrieve(query, k=k)
        return [
            {
                "id": node.id,
                "key": node.key,
                "type": node.type,
                "title": node.title,
                "text": self.graph.render(node),
                "tokens": node.tokens,
            }
            for node in nodes
        ]

    def explain(self, query: str, *, k: int | None = None) -> dict:
        nodes = self.retrieve(query, k=k)
        return {
            "query": query,
            "nodes": [{"id": n.id, "type": n.type, "title": n.title,
                       "parent": self.graph.get(n.parent_id).title if n.parent_id else None}
                      for n in nodes],
            "tokens": sum(n.tokens for n in nodes),
        }
