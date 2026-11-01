"""模型驱动：OpenAI 兼容接口。

Key 从环境变量或 ~/.dsh/.env 读，不写死在代码里。
支持超时、重试、tokens 计账。
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


def _read_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    for path in (Path.home() / ".dsh" / ".env", Path("/root/.dsh/.env")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEEPSEEK_API_KEY="):
                    return line.split("=", 1)[1].strip()
        except OSError:
            continue
    return ""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ModelReply:
    text: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    elapsed: float = 0.0


class DeepSeekKernel:
    def __init__(
        self,
        *,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        api_key: str | None = None,
        timeout_s: float = 60.0,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or _read_key()
        self.timeout_s = timeout_s
        self.retries = retries
        self.usage = Usage()
        self.calls = 0

    @property
    def ready(self) -> bool:
        return bool(self.api_key)

    async def infer(self, prompt: str, *, budget: int = 4000) -> str:
        reply = await self.chat(
            [
                {"role": "system", "content": "你是 A-OS 的内核进程。只输出下一步动作，简洁。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=min(600, max(64, budget // 8)),
        )
        return reply.text

    async def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> ModelReply:
        if not self.ready:
            raise RuntimeError("没有 DEEPSEEK_API_KEY")
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }).encode()

        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return await asyncio.to_thread(self._post, payload)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.retries:
                    await asyncio.sleep(0.8 * (2 ** attempt))
        raise RuntimeError(f"模型调用失败：{last_err}")

    def _post(self, payload: bytes) -> ModelReply:
        import time

        started = time.time()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode())

        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "").strip()
        u = data.get("usage") or {}
        usage = Usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))

        self.calls += 1
        self.usage.prompt_tokens += usage.prompt_tokens
        self.usage.completion_tokens += usage.completion_tokens
        return ModelReply(
            text=text,
            usage=usage,
            model=data.get("model", self.model),
            elapsed=time.time() - started,
        )


class OfflineKernel:
    """没 key 时的兜底，保证系统能跑通流程。"""

    def __init__(self) -> None:
        self.usage = Usage()

    @property
    def ready(self) -> bool:
        return False

    async def infer(self, prompt: str, *, budget: int = 4000) -> str:
        await asyncio.sleep(0.01)
        return "决定：继续当前分支（离线模式）"
