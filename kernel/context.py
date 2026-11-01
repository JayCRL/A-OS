"""上下文辅助：判断这轮要不要带工作区目录，以及按 token 截断。

真正的组装在 kernel/page_table.py 里，按段和预算来。
"""

from __future__ import annotations

from kernel.tokens import estimate_tokens

FILE_HINTS = (
    "文件", "目录", "代码", "项目", "仓库", "工程", "脚本", "文档", "笔记",
    "readme", "md", "py", "java", "go", "js", "json", "配置", "日志",
    "路径", "里面", "打开", "看看", "读一下", "改一下", "写个", "生成",
)


def wants_files(query: str) -> bool:
    low = (query or "").lower()
    return any(h in low for h in FILE_HINTS)


def truncate(text: str, max_tokens: int) -> str:
    text = text or ""
    if max_tokens <= 0 or estimate_tokens(text) <= max_tokens:
        return text
    keep = max(20, int(len(text) * max_tokens / max(1, estimate_tokens(text))))
    return text[:keep].rstrip() + "…"


__all__ = ["estimate_tokens", "wants_files", "truncate"]
