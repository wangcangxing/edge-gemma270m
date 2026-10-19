r"""指定 CPU 亲和性与线程数的延迟/内存实测（默认 2 核 2 线程）。

    python bench_2c.py                         # 2 核 2 线程（绑逻辑 CPU 0,2）
    python bench_2c.py --cpus 0 --threads 1    # 单核单线程
    EDGE_CPUS=0,1 python bench_2c.py           # 也可用环境变量给默认值

亲和性必须在 import torch 之前设置；内存用采样线程记录峰値（RSS 与 private）。
"""
from __future__ import annotations

import argparse
import os
import statistics
import threading
import time

import psutil

_ap = argparse.ArgumentParser(description="指定核数/线程数的端侧延迟与内存实测")
_ap.add_argument("--cpus", default=os.environ.get("EDGE_CPUS", "0,2"),
                 help="逗号分隔的逻辑 CPU 列表，默认 0,2")
_ap.add_argument("--threads", type=int, default=0, help="torch 线程数；默认等于核数")
_ap.add_argument("--image", default="", help="测试图路径；默认读 EDGE_SAMPLE_IMAGE")
_ap.add_argument("--adapter", default="", help="LoRA adapter；默认读 EDGE_ADAPTER")
_ap.add_argument("--runs", type=int, default=3)
ARGS = _ap.parse_args()

CPUS = [int(x) for x in ARGS.cpus.split(",") if x.strip() != ""]
THREADS = ARGS.threads or len(CPUS)
psutil.Process().cpu_affinity(CPUS)
os.environ["OMP_NUM_THREADS"] = str(THREADS)
os.environ["MKL_NUM_THREADS"] = str(THREADS)

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(THREADS)
torch.set_num_interop_threads(1)

from gemma_describe import DEFAULT_MODEL, GemmaDescriber  # noqa: E402
from paths import ADAPTER, SAMPLE_IMAGE  # noqa: E402
from scene_yolo import DEFAULT_WEIGHTS, SceneExtractor, load_image  # noqa: E402

SCENE = {"画面": "1280x720，正常",
         "人物": [{"位置": "左侧下", "身高占比": 0.43, "脚部接近地面": True, "姿态": "躺卧"}],
         "物体": ["沙发x2", "茶几"]}


class RssSampler:
    """后台采样进程内存峰値（RSS + private）。"""

    def __init__(self, interval: float = 0.02) -> None:
        self.peak_rss = 0
        self.peak_private = 0
        self._stop = threading.Event()
        self._interval = interval
        self._proc = psutil.Process()

    def _run(self) -> None:
        while not self._stop.is_set():
            mi = self._proc.memory_info()
            self.peak_rss = max(self.peak_rss, mi.rss)
            self.peak_private = max(self.peak_private, getattr(mi, "private", 0))
            time.sleep(self._interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        return False


def main() -> None:
    image = Path(ARGS.image) if ARGS.image else SAMPLE_IMAGE
    adapter = ARGS.adapter or str(ADAPTER)
    if not Path(image).exists():
        raise SystemExit(f"找不到测试图 {image}；用 --image 指定一张 1280x720 图片")

    proc = psutil.Process()
    rss_base = proc.memory_info().rss

    d = GemmaDescriber(DEFAULT_MODEL, "cpu", int8_cpu=True, threads=THREADS,
                       adapter=adapter)
    d.warmup()
    rss_gemma = proc.memory_info().rss

    ex = SceneExtractor(DEFAULT_WEIGHTS, None, device="cpu")
    ex.warmup()
    img = load_image(image)
    rss_both = proc.memory_info().rss

    gen = []
    with RssSampler() as smp_gen:
        for _ in range(ARGS.runs):
            r = d.describe(SCENE, "zh", max_new_tokens=48, greedy=True)
            gen.append((r["gen_s"], r["gen_tokens"]))
    gen_s = statistics.median(t for t, _ in gen)
    tok_s = statistics.mean(n for _, n in gen) / gen_s

    y_times = []
    with RssSampler() as smp_yolo:
        for _ in range(ARGS.runs):
            t0 = time.perf_counter()
            ex.extract(img, with_posture=False)
            y_times.append((time.perf_counter() - t0) * 1000)
    y_ms = statistics.median(y_times)

    peak_rss = max(rss_both, smp_gen.peak_rss, smp_yolo.peak_rss)
    peak_private = max(smp_gen.peak_private, smp_yolo.peak_private)
    out = {
        "cpus": CPUS,
        "torch_threads": THREADS,
        "gemma_int8": {"gen_s_median": round(gen_s, 3),
                       "tok_s": round(tok_s, 1),
                       "gen_tokens": gen[0][1]},
        "yolo_cpu_ms": round(y_ms, 1),
        "e2e_s_per_frame": round(y_ms / 1000 + gen_s, 2),
        "memory_gb": {
            "rss_base": round(rss_base / 1024 ** 3, 2),
            "rss_after_gemma": round(rss_gemma / 1024 ** 3, 2),
            "rss_after_both": round(rss_both / 1024 ** 3, 2),
            "peak_rss": round(peak_rss / 1024 ** 3, 2),
            "peak_private": round(peak_private / 1024 ** 3, 2),
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    res = Path(__file__).resolve().parent / "results"
    res.mkdir(exist_ok=True)
    name = "edge_bench_2c.json" if (len(CPUS) == 2 and THREADS == 2) \
        else f"edge_bench_{len(CPUS)}c{THREADS}t.json"
    (res / name).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", res / name)


if __name__ == "__main__":
    main()
