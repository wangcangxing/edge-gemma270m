"""把域内视频帧批量转成场景 JSONL（训练 / 验证 / 测试按视频切分）。

    python build_dataset.py --limit 50          # 先小样试跑
    python build_dataset.py                     # 全量 3116 张

输出 dataset/scenes.jsonl，每行一条：
    {"image": "...", "video": "fall_detection_4", "split": "train", **场景字段}
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from paths import DOMAIN_IMAGES, POSTURE_WEIGHTS
from scene_yolo import DEFAULT_WEIGHTS, SceneExtractor, load_image

HERE = Path(__file__).resolve().parent
DOMAIN = DOMAIN_IMAGES
POSTURE = POSTURE_WEIGHTS
SPLIT = {
    "train": {"fall_detection_4", "fall_detection_5", "fall_detection_6",
              "fall_detection_7", "fall_detection_8", "fall_detection_9",
              "fall_detection_10"},
    "val": {"hr_fall_detection_2"},
    "test": {"hr_fall_detection_1", "hr_fall_detection_3"},
}


def video_of(path: Path) -> str:
    return re.sub(r"_t\d+_\d+$", "", path.stem)


def split_of(video: str) -> str:
    for name, vids in SPLIT.items():
        if video in vids:
            return name
    return "train"


GT_POSTURE = {0: "站立", 1: "坐", 2: "躺卧"}


def gt_postures(img_path: Path, w: int, h: int) -> list[tuple[list[float], str]]:
    """读 YOLO 标注（cls cx cy bw bh），返回 [(像素框, 姿态)]。"""
    lab = Path(str(img_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")
    if not lab.exists():
        return []
    out = []
    for line in lab.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        if cls in GT_POSTURE:
            out.append(([x1, y1, x2, y2], GT_POSTURE[cls]))
    return out


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_posture(scene, img_path: Path) -> None:
    """用数据集标注的姿态覆盖/补全检测器的姿态（标注更可信）。"""
    gts = gt_postures(img_path, scene.width, scene.height)
    if not gts:
        return
    for person, box in zip(scene.persons, scene.person_boxes):
        best = max(gts, key=lambda g: iou(g[0], box))
        if iou(best[0], box) >= 0.3:
            person["姿态"] = best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(DOMAIN))
    ap.add_argument("--out", default=str(HERE / "dataset" / "scenes.jsonl"))
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--posture-weights", default=str(POSTURE) if POSTURE.exists() else "")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(Path(args.images).rglob("*.jpg"))
    if args.limit:
        files = files[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ext = SceneExtractor(args.weights, args.posture_weights or None, device=args.device)
    ext.warmup()
    t0 = time.perf_counter()
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for p in files:
            try:
                scene = ext.extract(load_image(p))
            except Exception as e:
                print(f"[skip] {p.name}: {e}")
                continue
            merge_posture(scene, p)
            rec = {"image": str(p), "video": video_of(p), "split": split_of(video_of(p)),
                   "width": scene.width, "height": scene.height, "light": scene.light,
                   "persons": scene.persons, "objects": scene.objects}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0:
                print(f"{n}/{len(files)}  {time.perf_counter()-t0:.0f}s")
    print(f"done {n} -> {out}  ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
