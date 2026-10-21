r"""完整 pipeline（姿态开启）下的线程 × 亲和性对照：20 帧。

    python bench_thread_sweep_posture.py --runs 20

控制变量：同一进程、同一批 20 帧、同 batch=1、max_new_tokens=48、姿态开启（默认 crop 模式）、
greedy；只改 torch.set_num_threads 与 CPU 亲和性。每配置预热 3 次后计时 20 帧。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import psutil
import torch

from gemma_describe import DEFAULT_MODEL, GemmaDescriber
from paths import ADAPTER, POSTURE_WEIGHTS, SCENES
from scene_yolo import DEFAULT_WEIGHTS, SceneExtractor, load_image

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

CONFIGS = [
    ("1t", 1, [0]),
    ("2t", 2, [0, 2]),
    ("4t", 4, [0, 2, 4, 6]),
    ("6t", 6, [0, 2, 4, 6, 8, 10]),
    ("8t", 8, [0, 2, 4, 6, 8, 10, 12, 14]),
    ("14t", 14, [0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19]),
    ("1e", 1, [12]),
    ("1e2t", 2, [12]),
    ("2e", 2, [12, 13]),
    ("4e", 4, [12, 13, 14, 15]),
    ("8e", 8, [12, 13, 14, 15, 16, 17, 18, 19]),
]


def stats(xs: list[float]) -> dict:
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return {"median": round(statistics.median(xs), 1),
            "mean": round(statistics.mean(xs), 1),
            "std": round(statistics.stdev(xs), 1) if len(xs) > 1 else 0.0,
            "p95": round(s[k], 1),
            "min": round(min(xs), 1), "max": round(max(xs), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--tag", default="thread_sweep_posture")
    ap.add_argument("--order", default="",
                    help="逗号分隔的配置名，可重复，例如 2t,6t,4t,6t,4t")
    ap.add_argument("--global-warmup", type=int, default=0,
                    help="正式测量前用 4t 预热若干次（不计入）")
    args = ap.parse_args()

    recs = [json.loads(l) for l in SCENES.read_text(encoding="utf-8").splitlines()]
    test = [r for r in recs if r["split"] == "test"]
    random.Random(1).shuffle(test)
    with_person = [r for r in test if r["persons"]][: args.runs]
    paths = [Path(r["image"]) for r in with_person]
    print(f"frames {len(paths)} (all with person)")

    ext = SceneExtractor(DEFAULT_WEIGHTS, str(POSTURE_WEIGHTS), device="cpu")
    ext.warmup()
    desc = GemmaDescriber(DEFAULT_MODEL, "cpu", int8_cpu=True, threads=1,
                          adapter=str(ADAPTER))
    desc.warmup()
    imgs = [load_image(p) for p in paths]

    out = {"batch": 1, "runs": len(paths), "posture": True,
           "max_new_tokens": args.max_new_tokens, "warmup": args.warmup,
           "global_warmup": args.global_warmup, "configs": {}}
    if args.global_warmup:
        psutil.Process().cpu_affinity([0, 2, 4, 6])
        torch.set_num_threads(4)
        for _ in range(args.global_warmup):
            s = ext.extract(imgs[0], with_posture=True)
            desc.describe(s.to_prompt_data(), "zh",
                          max_new_tokens=args.max_new_tokens, greedy=True)
        print(f"global warmup {args.global_warmup}x done", flush=True)
    by_name = {name: (threads, cpus) for name, threads, cpus in CONFIGS}
    order = [n.strip() for n in args.order.split(",") if n.strip()] or [c[0] for c in CONFIGS]
    for idx, name in enumerate(order):
        threads, cpus = by_name[name]
        key = name if order.count(name) == 1 else f"{name}#{idx}"
        psutil.Process().cpu_affinity(cpus)
        torch.set_num_threads(threads)
        for _ in range(args.warmup):
            s = ext.extract(imgs[0], with_posture=True)
            desc.describe(s.to_prompt_data(), "zh",
                          max_new_tokens=args.max_new_tokens, greedy=True)

        total, scene_ms, gen_ms, tok_s = [], [], [], []
        for img in imgs:
            t0 = time.perf_counter()
            s = ext.extract(img, with_posture=True)
            t1 = time.perf_counter()
            r = desc.describe(s.to_prompt_data(), "zh",
                              max_new_tokens=args.max_new_tokens, greedy=True)
            t2 = time.perf_counter()
            scene_ms.append((t1 - t0) * 1000)
            gen_ms.append((t2 - t1) * 1000)
            total.append((t2 - t0) * 1000)
            if r["tok_s"]:
                tok_s.append(r["tok_s"])
        out["configs"][key] = {
            "threads": threads, "cpus": cpus,
            "total_ms": stats(total), "scene_ms": stats(scene_ms),
            "gemma_ms": stats(gen_ms),
            "tok_s_mean": round(statistics.mean(tok_s), 1) if tok_s else None,
            "total_ms_list": [round(x, 1) for x in total],
        }
        print(f"{key:>6} total med/mean/std/p95 = "
              f"{out['configs'][key]['total_ms']['median']}/"
              f"{out['configs'][key]['total_ms']['mean']}/"
              f"{out['configs'][key]['total_ms']['std']}/"
              f"{out['configs'][key]['total_ms']['p95']} ms", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"edge_{args.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v["total_ms"] for k, v in out["configs"].items()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
