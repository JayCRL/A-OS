"""独立服务进程入口。

用法：
  python3 -m services.worker vfs     --home ... --sock ...
  python3 -m services.worker pager   --home ... --sock ... --vfs-sock ...
  python3 -m services.worker librarian --home ... --sock ... --pager-sock ...
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drivers.embedding import HashEncoder, load_encoder  # noqa: E402
from drivers.model import DeepSeekKernel, OfflineKernel  # noqa: E402
from kernel.unix_ipc import UnixClient, UnixServer  # noqa: E402
from services.librarian import LibrarianService  # noqa: E402
from services.pager import PagerService  # noqa: E402
from services.vfs import VFSService  # noqa: E402


def make_encoder():
    if os.environ.get("AOS_ENCODER") == "hash":
        return HashEncoder()
    return load_encoder()


def make_model():
    if os.environ.get("AOS_MODEL") == "offline":
        return OfflineKernel()
    return DeepSeekKernel() if DeepSeekKernel().ready else OfflineKernel()


async def run(args) -> None:
    home = Path(args.home)
    encoder = None if args.service == "librarian" else make_encoder()
    client = None
    service = None

    if args.service == "vfs":
        service = VFSService(home, encoder)
        await service.boot()
    elif args.service == "pager":
        client = UnixClient(args.vfs_sock)
        await client.wait_ready()
        service = PagerService(home, encoder, bus=client)
        await service.boot()
    elif args.service == "librarian":
        client = UnixClient(args.pager_sock)
        await client.wait_ready()
        service = LibrarianService(graph=None, model=make_model(), backend=client)
        await service.start()
    else:
        raise SystemExit(f"未知服务：{args.service}")

    server = UnixServer(args.sock, service.handle)
    await server.start()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    try:
        await stop.wait()
    finally:
        await server.close()
        if hasattr(service, "close"):
            await service.close()
        if client is not None:
            await client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=["vfs", "pager", "librarian"])
    parser.add_argument("--home", required=True)
    parser.add_argument("--sock", required=True)
    parser.add_argument("--vfs-sock", default="")
    parser.add_argument("--pager-sock", default="")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
