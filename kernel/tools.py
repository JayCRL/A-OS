"""给对话用的工具。默认全部藏在后面，用户只看得到结果。

文件读写限制在工作区；联网工具只做 GET，限时限量。
"""

from __future__ import annotations

import asyncio
import html as html_lib
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from kernel.memindex import MemoryIndex

MAX_READ = 12000
MAX_OUTPUT = 5000
TEXT_EXT = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".js", ".ts",
    ".html", ".css", ".sh", ".go", ".java", ".c", ".h", ".rs", ".sql",
}
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

DESCRIPTION = """read {"path":"README.md"}          读工作区文件
list {"path":"."}                     列目录
search {"query":"关键词"}             在工作区里找内容
write {"path":"a.md","content":"..."} 新建/覆盖文件（用户明确要产出时才用）
append {"path":"note.txt","content":"..."} 追加
web_search {"query":"..."}            联网搜索
web_fetch {"url":"https://..."}       打开网页取正文"""


@dataclass
class Tools:
    root: Path
    encoder: object | None = None
    index: MemoryIndex | None = None
    on_write: object | None = None
    calls: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.index is None and self.encoder is not None:
            try:
                self.index = MemoryIndex(self.root.parent / "chunks.db", self.encoder, self.root)
            except Exception:
                self.index = None

    # ---------- 入口 ----------

    async def run(self, name: str, args: dict) -> str:
        args = args or {}
        started = __import__("time").time()
        try:
            if name == "read":
                out = await asyncio.to_thread(self._read, str(args.get("path", "")))
            elif name == "list":
                out = await asyncio.to_thread(self._list, str(args.get("path", ".")))
            elif name == "search":
                out = await asyncio.to_thread(self._search, str(args.get("query", args.get("keyword", ""))))
            elif name == "write":
                out = await asyncio.to_thread(self._write, str(args.get("path", "")), str(args.get("content", "")))
            elif name == "append":
                out = await asyncio.to_thread(self._append, str(args.get("path", "")), str(args.get("content", "")))
            elif name == "web_search":
                out = await asyncio.to_thread(self._web_search, str(args.get("query", "")))
            elif name == "web_fetch":
                out = await asyncio.to_thread(self._web_fetch, str(args.get("url", "")))
            else:
                out = f"错误：没有这个工具 {name}"
        except Exception as exc:  # noqa: BLE001
            out = f"出错了：{type(exc).__name__}: {exc}"
        self.calls.append({
            "tool": name, "args": args, "ok": not out.startswith(("错误", "出错了")),
            "elapsed": round(__import__("time").time() - started, 2),
        })
        del self.calls[:-40]
        return out[:MAX_OUTPUT]

    # ---------- 路径 ----------

    def _resolve(self, raw: str) -> Path:
        rel = (raw or ".").strip().replace("\\", "/").lstrip("/")
        target = (self.root / rel).resolve()
        if target != self.root and self.root not in target.parents:
            raise PermissionError(f"路径超出工作区：{raw}")
        return target

    # ---------- 文件 ----------

    def read_page(self, path: str) -> tuple[str, str] | None:
        """给缺页换入用：返回 (相对路径, 正文)。不在这里拼提示词。"""

        try:
            target = self._resolve(path)
        except PermissionError:
            return None
        if not target.is_file():
            return None
        text = target.read_text(encoding="utf-8", errors="ignore")
        if len(text) > MAX_READ:
            text = text[:MAX_READ] + f"\n…（已截断，全文 {len(text)} 字符）"
        return str(target.relative_to(self.root)), text

    def semantic_hits(self, query: str, *, k: int = 2) -> list[dict]:
        """在工作区文件块里做语义寻址，供缺页使用。"""

        if self.index is None:
            return []
        try:
            hits = self.index.query(query, k=k)
        except Exception:
            return []
        return [
            {"source": f"file:{h.path}:{h.start}", "path": h.path,
             "text": h.text, "score": h.score}
            for h in hits if h.score >= 0.48
        ]

    def _read(self, path: str) -> str:
        target = self._resolve(path)
        if target.is_dir():
            return self._list(path)
        if not target.is_file():
            return f"错误：没有这个文件：{path}"
        text = target.read_text(encoding="utf-8", errors="ignore")
        rel = target.relative_to(self.root)
        if len(text) > MAX_READ:
            text = text[:MAX_READ] + f"\n…（已截断，全文 {len(text)} 字符）"
        return f"{rel}\n{text}"

    def _list(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            return f"错误：没有这个目录：{path}"
        items = []
        for child in sorted(target.iterdir())[:100]:
            items.append(("目录 " if child.is_dir() else "文件 ") + child.name)
        return "\n".join(items) or "（空目录）"

    def _search(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "错误：搜索词为空"
        hits: list[str] = []
        for path in sorted(self.root.rglob("*")):
            if len(hits) >= 8:
                break
            if not path.is_file() or path.suffix.lower() not in TEXT_EXT:
                continue
            if any(p in path.parts for p in (".git", "__pycache__", "node_modules")):
                continue
            try:
                if path.stat().st_size > 800_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            idx = text.lower().find(query.lower())
            if idx < 0:
                continue
            snippet = text[max(0, idx - 100): idx + 260].replace("\n", " ")
            hits.append(f"{path.relative_to(self.root)}: {snippet}")
        if not hits and self.index is not None:
            try:
                for hit in self.index.query(query, k=3):
                    if hit.score < 0.5:
                        continue
                    snippet = hit.text[:260].replace("\n", " ")
                    hits.append(f"{hit.path}:{hit.start}（语义匹配 {hit.score:.2f}）: {snippet}")
            except Exception:
                pass
        return "\n".join(hits) or f"没有找到和「{query}」有关的内容"

    def _write(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".aos-tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)
        rel = str(target.relative_to(self.root))
        if self.on_write is not None:
            try:
                self.on_write(rel)
            except Exception:
                pass
        self._reindex()
        return f"已写入 {rel}（{len(content)} 字符）"

    def _append(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(content + "\n")
        rel = str(target.relative_to(self.root))
        if self.on_write is not None:
            try:
                self.on_write(rel)
            except Exception:
                pass
        self._reindex()
        return f"已追加到 {rel}（+{len(content)} 字符）"

    def _reindex(self) -> None:
        if self.index is not None:
            try:
                self.index.refresh()
            except Exception:
                pass

    def tree(self, *, limit: int = 40) -> str:
        rows: list[str] = []
        for child in sorted(self.root.rglob("*")):
            if len(rows) >= limit:
                rows.append("…")
                break
            if any(p in child.parts for p in (".git", "__pycache__", "node_modules")):
                continue
            rel = child.relative_to(self.root)
            rows.append(("目录 " if child.is_dir() else "文件 ") + "./" + str(rel))
        return "\n".join(rows)

    # ---------- 联网 ----------

    def _web_search(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "错误：搜索词为空"
        url = "https://cn.bing.com/search?q=" + urllib.parse.quote(query)
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            page = resp.read(600_000).decode("utf-8", "ignore")
        blocks = re.findall(r'<li class="b_algo".*?</li>', page, re.S)
        results: list[str] = []
        for block in blocks[:6]:
            m = re.search(r"<h2[^>]*>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", block, re.S)
            if not m:
                continue
            link = html_lib.unescape(m.group(1))
            title = _strip_tags(m.group(2))
            snippet = ""
            sm = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
            if sm:
                snippet = _strip_tags(sm.group(1))
            results.append(f"- {title}\n  {link}\n  {snippet[:220]}")
        if not results:
            return "搜索没有返回结果"
        return f"「{query}」的搜索结果：\n" + "\n".join(results)

    def _web_fetch(self, url: str) -> str:
        url = url.strip()
        if not re.match(r"^https?://", url):
            return "错误：只支持 http/https 链接"
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read(800_000)
            charset = resp.headers.get_content_charset() or "utf-8"
        try:
            page = raw.decode(charset, "ignore")
        except LookupError:
            page = raw.decode("utf-8", "ignore")
        page = re.sub(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", page, flags=re.S | re.I)
        page = re.sub(r"<!--.*?-->", " ", page, flags=re.S)
        text = _strip_tags(page)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > 6000:
            text = text[:6000] + "\n…（已截断）"
        return f"{url}\n{text or '（页面没有可读正文）'}"


def _strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html_lib.unescape(text)
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()
