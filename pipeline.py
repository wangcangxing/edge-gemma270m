"""端到端：YOLOv10n 看 → Gemma-3-270M 说。

    python pipeline.py --source image.jpg
    python pipeline.py --source clip.mp4 --fps 1 --max-frames 4
    python pipeline.py --source image.jpg --gemma-device cpu --int8

输出写到 results/<tag>.json：每帧的场景、提示词 token 数、生成 tokens、耗时、峰值显存。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from gemma_describe import DEFAULT_MODEL, GemmaDescriber
from paths import ADAPTER
from scene_yolo import DEFAULT_WEIGHTS, SceneExtractor

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def iter_frames(source: str, fps: float, max_frames: int):
    p = Path(source)
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        # 中文路径安全读取
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"读不到图片: {p}")
        yield 0, img
        return

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise FileNotFoundError(f"读不到视频: {p}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / fps)))
    idx = 0
    emitted = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield idx, frame
            emitted += 1
            if max_frames and emitted >= max_frames:
                break
        idx += 1
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLOv10n + Gemma-3-270M 语义描述")
    ap.add_argument("--source", required=True, help="图片或视频路径")
    ap.add_argument("--yolo", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--posture-weights", default="")
    ap.add_argument("--gemma", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=str(ADAPTER),
                    help="LoRA adapter 目录（默认 out/lora_gemma270m_zh，可用 EDGE_ADAPTER 覆盖）")
    ap.add_argument("--yolo-device", default="cuda:0")
    ap.add_argument("--gemma-device", default="cuda:0")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--fps", type=float, default=1.0, help="视频抽帧率")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--only-with-person", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--int8", action="store_true", help="CPU 动态 int8 量化")
    ap.add_argument("--tag", default="")
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    ext = SceneExtractor(args.yolo, args.posture_weights or None,
                         device=args.yolo_device)
    ext.warmup()
    describer = GemmaDescriber(args.gemma, args.gemma_device, args.int8,
                               adapter=args.adapter)
    describer.warmup()
    print(f"[load] gemma {describer.load_s:.2f}s on {args.gemma_device}"
          f"{' (int8)' if describer.int8 else ''}")

    records: list[dict] = []
    tag = args.tag or Path(args.source).stem
    for idx, img in iter_frames(args.source, args.fps, args.max_frames):
        t0 = time.perf_counter()
        scene = ext.extract(img)
        yolo_ms = (time.perf_counter() - t0) * 1000
        scene_data = scene.to_prompt_data()
        rec = {"frame": idx, "yolo_ms": round(yolo_ms, 2),
               "scene": scene_data}
        if args.only_with_person and not scene.persons:
            rec["description"] = ""
            rec["skipped"] = "no_person"
        else:
            t1 = time.perf_counter()
            r = describer.describe(scene_data, args.lang, args.max_new_tokens,
                                   greedy=args.greedy)
            r["e2e_s"] = round(time.perf_counter() - t1 + yolo_ms / 1000, 3)
            rec.update(r)
        records.append(rec)
        print(f"[{idx:>5}] yolo {yolo_ms:5.1f}ms | {rec.get('description', '')}")

    bench = {
        "source": str(Path(args.source).resolve()),
        "lang": args.lang,
        "yolo": str(args.yolo),
        "gemma": args.gemma,
        "gemma_device": args.gemma_device,
        "gemma_load_s": round(describer.load_s, 2),
        "gemma_mem_after_load_gb": round(describer.mem_after_load_gb, 2),
        "records": records,
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"edge_{tag}.json"
    out.write_text(json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out}")

    if records:
        avg_y = sum(r["yolo_ms"] for r in records) / len(records)
        gens = [r for r in records if r.get("gen_s")]
        print(f"[summary] frames={len(records)} yolo_avg={avg_y:.1f}ms", end="")
        if gens:
            print(f" gen_avg={sum(r['gen_s'] for r in gens)/len(gens):.2f}s"
                  f" tok/s_avg={sum(r['tok_s'] for r in gens)/len(gens):.1f}")
        else:
            print()


if __name__ == "__main__":
    main()
