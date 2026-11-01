"""统一的 token 估算。

中文一个字约一个 token，英文按词估；只用于预算，不追求精确。
"""

from __future__ import annotations

import re

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    text = text or ""
    cjk = len(_CJK.findall(text))
    words = len(_WORD.findall(text))
    rest = max(0, len(text) - cjk)
    return max(1, cjk + words + rest // 4)
