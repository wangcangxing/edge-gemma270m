r"""与 Mage 项目原协议对齐的跌倒判定实验。

用例来自 results/probe_vlm_crossdomain.json（Mage 跨域测试的 221 行：66 Lie / 80 Stand / 75 Sit，
Mage 答案已记录）。这里映射回同一批帧，跑当前 edge 语义层：
    YOLOv10n + 姿态检测器 → 场景 JSON → Gemma-3-270M 一句描述
判定：描述里出现“躺/卧”即回答“有人躺在地上”。

    python compare_mage_fall.py --probe <probe_vlm_crossdomain.json>
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from gemma_describe import DEFAULT_MODEL, GemmaDescriber
from paths import ADAPTER, DOMAIN_IMAGES, POSTURE_WEIGHTS
from scene_yolo import DEFAULT_WEIGHTS, SceneExtractor, load_image

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def find_frame(video: str, t: int) -> Path | None:
    stem = f'{video.replace(".mp4", "")}_t{t:04d}'
    hits = list(DOMAIN_IMAGES.rglob(f"{stem}_*.jpg"))
    return hits[0] if hits else None


def metrics(rows: list[dict], key: str) -> dict:
    tp = sum(1 for r in rows if r["gt"] == "Lie" and r[key])
    fn = sum(1 for r in rows if r["gt"] == "Lie" and not r[key])
    fp = sum(1 for r in rows if r["gt"] != "Lie" and r[key])
    tn = sum(1 for r in rows if r["gt"] != "Lie" and not r[key])
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 0,
            "precision": round(tp / (tp + fp), 3) if tp + fp else 0,
            "accuracy": round((tp + tn) / len(rows), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--tag", default="edge_vs_mage_fall")
    args = ap.parse_args()

    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    src = probe["rows"]

    ext = SceneExtractor(DEFAULT_WEIGHTS, str(POSTURE_WEIGHTS), device="cuda:0")
    ext.warmup()
    desc = GemmaDescriber(DEFAULT_MODEL, "cuda:0", adapter=str(ADAPTER))
    desc.warmup()

    rows, missing = [], 0
    for r in src:
        img = find_frame(r["video"], r["t"])
        if img is None:
            missing += 1
            continue
        scene = ext.extract(load_image(img), posture_whole=True, posture_min_conf=0.25)
        text = desc.describe(scene.to_prompt_data(), "zh", max_new_tokens=48,
                             greedy=True)["description"]
        posture = scene.persons[0].get("姿态") if scene.persons else None
        rows.append({
            "video": r["video"], "t": r["t"], "gt": r["gt"],
            "mage_lie": bool(r["vlm_lie"]),
            "old_detector": r["detector"],
            "edge_posture": posture,
            "edge_lie_detector": posture == "躺卧",
            "edge_desc": text,
            "edge_lie_desc": ("躺" in text) or ("卧" in text),
            "edge_person": bool(scene.persons),
        })
        if len(rows) % 40 == 0:
            print(f"{len(rows)}  gt={r['gt']}  posture={posture}  {text[:30]}", flush=True)

    out = {
        "n": len(rows), "missing_frames": missing,
        "counts": {k: sum(1 for r in rows if r["gt"] == k) for k in ("Lie", "Stand", "Sit")},
        "mage": metrics(rows, "mage_lie"),
        "edge_detector": metrics(rows, "edge_lie_detector"),
        "edge_desc": metrics(rows, "edge_lie_desc"),
        "edge_person_rate": round(sum(r["edge_person"] for r in rows) / len(rows), 3),
        "rows": rows,
    }
    (RESULTS / f"edge_{args.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
