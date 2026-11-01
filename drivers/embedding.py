"""本地向量编码器：bge-small-zh-v1.5（ONNX，CPU）。

不联网、不花额度；首次加载约 1 秒，之后每条 5~10ms。
模型缺失时自动降级到哈希编码，保证系统不倒。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_DIR = Path("/opt/aos/models/bge-small-zh")
MAX_LEN = 256
DIM = 512


class HashEncoder:
    """兜底：字符哈希，能跑但没有语义。"""

    name = "hash"
    dim = 64

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for j, ch in enumerate(text[:400]):
                out[i, (ord(ch) + j) % self.dim] += 1.0
        return _normalize(out)

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0].tolist()


class BgeEncoder:
    """bge-small-zh-v1.5 ONNX 推理，CLS 池化 + L2 归一化。"""

    name = "bge-small-zh"
    dim = DIM

    def __init__(self, model_dir: Path | str = DEFAULT_DIR) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        onnx_path = model_dir / "model.onnx"
        tok_path = model_dir / "tokenizer.json"
        if not onnx_path.exists() or not tok_path.exists():
            raise FileNotFoundError(f"模型缺失：{model_dir}")

        self._tok = Tokenizer.from_file(str(tok_path))
        self._tok.enable_truncation(max_length=MAX_LEN)
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        encs = self._tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self._sess.run(None, feed)[0]
        cls = hidden[:, 0, :]
        return _normalize(np.asarray(cls, dtype=np.float32))

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0].tolist()


def _normalize(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return mat / norm


def load_encoder(model_dir: Path | str = DEFAULT_DIR):
    try:
        enc = BgeEncoder(model_dir)
        enc.encode_one("预热")
        return enc
    except Exception:
        return HashEncoder()


def cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    if matrix.size == 0:
        return []
    sims = matrix @ query
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]
