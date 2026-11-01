"""跨进程锁。

两层：
1. Redis SET NX PX —— 跨机器、跨容器都能用
2. fcntl.flock —— Redis 不可用时兜底，同机多进程有效

锁按路径粒度，带 TTL，进程崩了会自动过期，不会永久占着。
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

LOCK_PREFIX = "aos:lock:"
_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""
_RENEW_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('pexpire', KEYS[1], ARGV[2])
end
return 0
"""


class RedisLock:
    def __init__(self, client, *, ttl_ms: int = 15000, wait_s: float = 20.0) -> None:
        self._redis = client
        self.ttl_ms = ttl_ms
        self.wait_s = wait_s

    def _key(self, path: str) -> str:
        digest = hashlib.sha1(path.encode()).hexdigest()[:16]
        return f"{LOCK_PREFIX}{digest}"

    async def acquire(self, path: str, owner: str) -> tuple[str, str] | None:
        key = self._key(path)
        token = f"{owner}:{os.getpid()}:{time.time()}"
        deadline = time.time() + self.wait_s
        while time.time() < deadline:
            got = await self._redis.set(key, token, nx=True, px=self.ttl_ms)
            if got:
                return key, token
            await asyncio.sleep(0.05)
        return None

    async def release(self, key: str, token: str) -> None:
        try:
            await self._redis.eval(_RELEASE_SCRIPT, 1, key, token)
        except Exception:  # noqa: BLE001
            pass

    async def renew(self, key: str, token: str) -> None:
        try:
            await self._redis.eval(_RENEW_SCRIPT, 1, key, token, self.ttl_ms)
        except Exception:  # noqa: BLE001
            pass


class FileLock:
    """同机兜底：对锁文件 flock，阻塞等待。"""

    def __init__(self, lock_dir: Path, *, wait_s: float = 20.0) -> None:
        self.dir = lock_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.wait_s = wait_s

    async def acquire(self, path: str, owner: str) -> tuple[str, str] | None:
        digest = hashlib.sha1(path.encode()).hexdigest()[:16]
        fd_path = self.dir / f"{digest}.lock"

        def _try() -> tuple[str, str] | None:
            fd = os.open(fd_path, os.O_CREAT | os.O_RDWR, 0o644)
            deadline = time.time() + self.wait_s
            while time.time() < deadline:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.write(fd, f"{owner}\n".encode())
                    return str(fd_path), str(fd)
                except BlockingIOError:
                    time.sleep(0.05)
            os.close(fd)
            return None

        return await asyncio.to_thread(_try)

    async def release(self, key: str, token: str) -> None:
        try:
            fd = int(token)
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except Exception:  # noqa: BLE001
            pass


class PathLocks:
    """对外统一入口：优先 Redis，失败退化到 flock。"""

    def __init__(self, *, redis_client=None, lock_dir: Path | None = None, owner: str = "aos") -> None:
        self.owner = owner
        self.redis_lock = RedisLock(redis_client) if redis_client is not None else None
        self.file_lock = FileLock(lock_dir or Path("/tmp/aos-locks"))
        self.hits = {"redis": 0, "file": 0}
        self.waited = 0

    @asynccontextmanager
    async def hold(self, path: str):
        started = time.time()
        backend = None
        handle = None

        if self.redis_lock is not None:
            try:
                handle = await self.redis_lock.acquire(path, self.owner)
                if handle:
                    backend = "redis"
            except Exception:  # noqa: BLE001
                handle = None

        if handle is None:
            handle = await self.file_lock.acquire(path, self.owner)
            if handle:
                backend = "file"

        if handle is None:
            raise TimeoutError(f"等锁超时：{path}")

        self.hits[backend] += 1
        waited = time.time() - started
        if waited > 0.1:
            self.waited += 1

        try:
            yield backend
        finally:
            if backend == "redis":
                await self.redis_lock.release(*handle)
            else:
                await self.file_lock.release(*handle)

    def stats(self) -> dict[str, int]:
        return {**self.hits, "waited": self.waited}
