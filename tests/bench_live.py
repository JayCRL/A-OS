"""线上任务冒烟基准：wall 延迟、事件 token、任务是否真的完成。"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8799/api/chat"
DB = Path("/opt/aos/data/brain.db")
WS = Path("/opt/aos/data/workspace")


def post(text: str) -> tuple[str, float]:
    started = time.time()
    req = urllib.request.Request(
        API, data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        data = json.loads(resp.read().decode())
    return str(data.get("reply") or data.get("error") or ""), time.time() - started


def event_row(after_id: int) -> dict:
    con = sqlite3.connect(DB)
    try:
        row = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(tokens),0), COALESCE(SUM(latency),0)"
            " FROM memory_events WHERE id>?", (after_id,)
        ).fetchone()
        last = con.execute("SELECT COALESCE(MAX(id),0) FROM memory_events").fetchone()[0]
    finally:
        con.close()
    return {"turns": int(row[0]), "tokens": int(row[1]),
            "agent_seconds": round(float(row[2]), 2), "last_id": int(last)}


def run(name: str, prompt: str, check) -> dict:
    con = sqlite3.connect(DB)
    before = con.execute("SELECT COALESCE(MAX(id),0) FROM memory_events").fetchone()[0]
    con.close()
    reply, wall = post(prompt)
    ev = event_row(before)
    ok = bool(check(reply))
    return {"task": name, "ok": ok, "wall_s": round(wall, 2),
            "tokens": ev["tokens"], "agent_s": ev["agent_seconds"],
            "reply": reply[:180]}


def main() -> None:
    WS.mkdir(parents=True, exist_ok=True)
    (WS / "bench_note.txt").write_text("BENCH-READ-ALPHA\n", encoding="utf-8")
    (WS / "bench_edit.txt").write_text("foo\n", encoding="utf-8")
    tasks = [
        ("read", "读一下 workspace/bench_note.txt，把内容原样告诉我",
         lambda r: "BENCH-READ-ALPHA" in r),
        ("edit", "把 workspace/bench_edit.txt 里的 foo 改成 bar",
         lambda r: (WS / "bench_edit.txt").read_text().strip() == "bar"),
        ("verify", "运行 verify 检查工作区",
         lambda r: any(x in r for x in ("通过", "没有报错", "verify"))),
        ("remember", "记住：我的测试暗号是青竹AOS",
         lambda r: bool(r)),
        ("recall", "我的测试暗号是什么",
         lambda r: "青竹AOS" in r),
        ("web", "查一下今天武汉天气",
         lambda r: "武汉" in r and any(x in r for x in ("°C", "℃", "晴", "雨", "温度"))),
    ]
    results = [run(*t) for t in tasks]
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    ok = sum(1 for r in results if r["ok"])
    print(json.dumps({
        "success": f"{ok}/{len(results)}",
        "avg_wall_s": round(sum(r["wall_s"] for r in results) / len(results), 2),
        "avg_tokens": round(sum(r["tokens"] for r in results) / len(results), 0),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
