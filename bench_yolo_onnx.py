r"""YOLOv10n ONNX Runtime 2 线程（绑 2 个 E 核）单帧延迟。

    python bench_yolo_onnx.py
"""
from __future__ import annotations

import json
import os
import random
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil

from paths import SCENES

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODEL = HERE / "yolov10n.onnx"


def letterbox(img: np.ndarray, size: int = 640) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def main() -> None:
    cpus = [int(x) for x in os.environ.get("ONNX_CPUS", "12,13").split(",")]
    psutil.Process().cpu_affinity(cpus)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(MODEL), sess_options=so,
                                providers=["CPUExecutionProvider"])

    recs = [json.loads(l) for l in SCENES.read_text(encoding="utf-8").splitlines()]
    test = [r for r in recs if r["split"] == "test"]
    random.Random(1).shuffle(test)
    paths = [Path(r["image"]) for r in test[:20]]
    imgs = []
    for p in paths:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        imgs.append(letterbox(img))

    for x in imgs[:3]:
        sess.run(None, {"images": x})
    lat = []
    for x in imgs:
        t0 = time.perf_counter()
        sess.run(None, {"images": x})
        lat.append((time.perf_counter() - t0) * 1000)
    s = sorted(lat)
    out = {"n": len(lat), "model": "yolov10n.onnx", "threads": 2, "cpus": cpus,
           "median_ms": round(statistics.median(lat), 1),
           "mean_ms": round(statistics.mean(lat), 1),
           "std_ms": round(statistics.stdev(lat), 1),
           "p95_ms": round(s[int(round(0.95 * (len(s) - 1)))], 1),
           "rss_mb": round(psutil.Process().memory_info().rss / 1024 ** 2, 1)}
    (RESULTS / "edge_yolo_onnx_2e.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
