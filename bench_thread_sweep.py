r"""线程数 × 亲和性扫描：同一进程、同一 batch、同一 max_new_tokens、姿态关闭。

    python bench_thread_sweep.py --runs 20 --warmup 3

每个配置只改 torch.set_num_threads 与 CPU 亲和性；其余全部固定：
- batch = 1 帧（同一张 1280x720 室内帧）
- max_new_tokens = 48，greedy
- 姿态开关 = 关闭（with_posture=False）
- 同一进程、同一 int8 模型、同一 YOLO 权重
预热 3 次后计时 20 次，报 median / mean / std / p95。
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import psutil
import torch

from gemma_describe import DEFAULT_MODEL, GemmaDescriber
from paths import ADAPTER, SAMPLE_IMAGE
from scene_yolo import DEFAULT_WEIGHTS, SceneExtractor, load_image

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# 1/2/4/6 用 P 核（0,2,4,6,8,10），8/14 再补 E 核（12~19）
CONFIGS = [
    ("1t", 1, [0]),
    ("2t", 2, [0, 2]),
    ("4t", 4, [0, 2, 4, 6]),
    ("6t", 6, [0, 2, 4, 6, 8, 10]),
    ("8t", 8, [0, 2, 4, 6, 8, 10, 12, 14]),
    ("14t", 14, [0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19]),
]


def stats(xs: list[float]) -> dict:
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    k = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
    return {
        "median": round(statistics.median(xs), 1),
        "mean": round(statistics.mean(xs), 1),
        "std": round(statistics.stdev(xs), 1) if n > 1 else 0.0,
        "p95": round(xs_sorted[k], 1),
        "min": round(min(xs), 1),
        "max": round(max(xs), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(SAMPLE_IMAGE))
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--tag", default="thread_sweep")
    args = ap.parse_args()

    img = load_image(args.image)
    ext = SceneExtractor(DEFAULT_WEIGHTS, None, device="cpu")
    ext.warmup()
    desc = GemmaDescriber(DEFAULT_MODEL, "cpu", int8_cpu=True, threads=1,
                          adapter=str(ADAPTER))
    desc.warmup()
    scene = ext.extract(img, with_posture=False)

    out = {"batch": 1, "max_new_tokens": args.max_new_tokens, "posture": False,
           "runs": args.runs, "warmup": args.warmup, "configs": {}}
    for name, threads, cpus in CONFIGS:
        psutil.Process().cpu_affinity(cpus)
        torch.set_num_threads(threads)
        for _ in range(args.warmup):
            d = desc.describe(scene.to_prompt_data(), "zh",
                              max_new_tokens=args.max_new_tokens, greedy=True)
            ext.extract(img, with_posture=False)

        total_ms, gen_ms, tok_s = [], [], []
        tokens = 0
        for _ in range(args.runs):
            t0 = time.perf_counter()
            s = ext.extract(img, with_posture=False)
            r = desc.describe(s.to_prompt_data(), "zh",
                              max_new_tokens=args.max_new_tokens, greedy=True)
            total_ms.append((time.perf_counter() - t0) * 1000)
            gen_ms.append(r["gen_s"] * 1000)
            tokens = r["gen_tokens"]
            if r["tok_s"]:
                tok_s.append(r["tok_s"])
        out["configs"][name] = {
            "threads": threads, "cpus": cpus,
            "tokens": tokens,
            "total_ms": stats(total_ms),
            "gemma_ms": stats(gen_ms),
            "tok_s_mean": round(statistics.mean(tok_s), 1) if tok_s else None,
        }
        print(f"{name:>4}  total med/mean/std/p95 = "
              f"{out['configs'][name]['total_ms']['median']}/"
              f"{out['configs'][name]['total_ms']['mean']}/"
              f"{out['configs'][name]['total_ms']['std']}/"
              f"{out['configs'][name]['total_ms']['p95']} ms", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"edge_{args.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
