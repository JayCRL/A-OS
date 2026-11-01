"""测试用的确定性编码器：按中文二元组/英文词做哈希向量。

不依赖 ONNX，10k 条文本也很快；语义相近的文本会共享 token。
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np


class TokenEncoder:
    name = "token-hash"
    dim = 128

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in self._tokens(text):
                idx = int.from_bytes(hashlib.sha1(token.encode("utf-8")).digest()[:4], "big") % self.dim
                out[i, idx] += 1.0
        return _norm(out)

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0].tolist()

    @staticmethod
    def _tokens(text: str) -> list[str]:
        low = (text or "").lower()
        chars = re.sub(r"\s+", "", low)
        grams = [chars[i: i + 2] for i in range(max(0, len(chars) - 1))]
        grams += re.findall(r"[a-z0-9_]{2,}", low)
        return grams or [low[:8] or "empty"]


def _norm(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n
