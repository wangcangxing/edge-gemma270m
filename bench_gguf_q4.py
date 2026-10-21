r"""GGUF Q4_K_M + llama.cpp server 的 20 帧描述基准。

    python bench_gguf_q4.py

服务需先启动：
    llama-server.exe -m gemma-270m-q4_k_m.gguf -t 2 -c 512 --port 18080 --no-webui
"""
from __future__ import annotations

import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

import psutil

from eval_tuned import ALL_OBJECTS, allowed_objects, mentioned, person_ok
from paths import SCENES
from prompting import build_messages, clean_output
from targets import _clean_obj
from train_lora import to_prompt_data

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
URL = "http://127.0.0.1:18080/v1/chat/completions"


def ask(messages: list[dict], max_tokens: int = 32) -> tuple[str, float, dict]:
    body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "cache_prompt": False, "stream": False}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode())
    dt = time.perf_counter() - t0
    return d["choices"][0]["message"]["content"], dt, d.get("usage", {})


def main() -> None:
    recs = [json.loads(l) for l in SCENES.read_text(encoding="utf-8").splitlines()]
    test = [r for r in recs if r["split"] == "test"]
    random.Random(1).shuffle(test)
    sample = test[:20]

    proc = next((p for p in psutil.process_iter(["name", "pid"])
                 if p.info["name"] == "llama-server"), None)
    ask(build_messages(to_prompt_data(sample[0])), 1)  # warmup

    rows = []
    for r in sample:
        out, dt, usage = ask(build_messages(to_prompt_data(r)))
        out = clean_output(out)
        objs = [_clean_obj(o) for o in r["objects"]]
        allowed = allowed_objects(r)
        hall = [w for w in mentioned(out, sorted(ALL_OBJECTS)) if w not in allowed]
        rows.append({"image": Path(r["image"]).name, "n": len(r["persons"]),
                     "desc": out, "latency_s": round(dt, 3),
                     "person_ok": person_ok(out, len(r["persons"])),
                     "recall": (len(mentioned(out, objs)) / len(objs)) if objs else 1.0,
                     "hallucinated": hall, "usage": usage})

    lat = [x["latency_s"] for x in rows]
    s = sorted(lat)
    out = {
        "n": len(rows),
        "model": "gemma-270m-q4_k_m.gguf",
        "threads": 2,
        "server_rss_mb": round(proc.memory_info().rss / 1024 ** 2, 1) if proc else None,
        "latency_s": {"median": round(statistics.median(lat), 3),
                      "mean": round(statistics.mean(lat), 3),
                      "std": round(statistics.stdev(lat), 3),
                      "p95": round(s[int(round(0.95 * (len(s) - 1)))], 3)},
        "person_ok": round(sum(x["person_ok"] for x in rows) / len(rows), 3),
        "recall": round(statistics.mean(x["recall"] for x in rows), 3),
        "halluc_rate": round(sum(1 for x in rows if x["hallucinated"]) / len(rows), 3),
        "rows": rows,
    }
    (RESULTS / "edge_gguf_q4_20.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    for x in rows:
        print(f'{x["latency_s"]:>6}s ok={x["person_ok"]} | {x["desc"]}')


if __name__ == "__main__":
    main()
