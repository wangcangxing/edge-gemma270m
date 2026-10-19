r"""YOLOv10n 在 test 划分上的检测指标：人数、物体自一致性、姿态召回。

    python eval_yolo_test.py

口径：
- 人数：YOLO 检测人数 vs 数据集标注框数（class 0/1/2 的行数）
- 物体：场景全局物体名 vs 建库时 scenes.jsonl 记录的物体名（**无 COCO 真值，只能算自一致性**）
- 姿态：按 IoU≥0.5 把检测框配到标注框；整图姿态模型 conf≥0.25 的预测才算命中
先跑全部 592 帧 test，再单独报 seed=1 抽的 100 帧与全体 lie 帧。
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from paths import POSTURE_WEIGHTS, SCENES
from scene_yolo import DEFAULT_WEIGHTS, SceneExtractor, load_image, _iou
from targets import _clean_obj

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
GT_POSTURE = {0: "站立", 1: "坐", 2: "躺卧"}


def label_path(img: Path) -> Path:
    return Path(str(img).replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def gt_boxes(img: Path, w: int, h: int) -> list[tuple[list[float], str]]:
    p = label_path(img)
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls not in GT_POSTURE:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        out.append(([(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h], GT_POSTURE[cls]))
    return out


def main() -> None:
    recs = [json.loads(l) for l in SCENES.read_text(encoding="utf-8").splitlines()]
    test = [r for r in recs if r["split"] == "test"]
    random.Random(1).shuffle(test)
    sample100 = {r["image"] for r in test[:100]}

    ext = SceneExtractor(DEFAULT_WEIGHTS, str(POSTURE_WEIGHTS), device="cuda:0")
    ext.warmup()

    frames = []
    for r in test:
        img_path = Path(r["image"])
        img = load_image(img_path)
        scene = ext.extract(img, posture_whole=True, posture_min_conf=0.25)
        gts = gt_boxes(img_path, scene.width, scene.height)
        frames.append({"image": r["image"], "scene": scene, "gts": gts,
                       "stored_objects": [_clean_obj(o) for o in r["objects"]]})

    def aggregate(rows: list[dict]) -> dict:
        # 人数
        count_ok = 0
        abs_err = []
        conf = Counter()
        # 物体自一致性（按“帧内名称集合”）
        inter = union = 0
        # 姿态
        stat = {k: Counter() for k in ("站立", "坐", "躺卧")}
        person_recall = [0, 0]
        for f in rows:
            pred_n = len(f["scene"].persons)
            gt_n = len(f["gts"])
            count_ok += int(pred_n == gt_n)
            abs_err.append(abs(pred_n - gt_n))
            conf[(gt_n, pred_n)] += 1

            stored = set(f["stored_objects"])
            detected = {_clean_obj(o) for o in f["scene"].objects}
            inter += len(stored & detected)
            union += len(stored)

            for box, cls in f["gts"]:
                person_recall[1] += 1
                idx = None
                best_iou = 0.0
                for i, pbox in enumerate(f["scene"].person_boxes):
                    v = _iou(pbox, box)
                    if v > best_iou:
                        best_iou, idx = v, i
                if idx is not None and best_iou >= 0.5:
                    person_recall[0] += 1
                    pred = f["scene"].persons[idx].get("姿态")
                else:
                    pred = None
                if pred == cls:
                    stat[cls]["正确"] += 1
                elif pred is None:
                    stat[cls]["未检出/未判"] += 1
                else:
                    stat[cls]["错判"] += 1

        posture = {c: {"n": sum(v.values()), **dict(v),
                       "recall": round(v["正确"] / sum(v.values()), 3) if sum(v.values()) else None}
                   for c, v in stat.items()}
        return {
            "n": len(rows),
            "person_count_accuracy": round(count_ok / len(rows), 3),
            "person_count_mae": round(sum(abs_err) / len(rows), 3),
            "person_count_confusion": {f"{g}->{p}": n for (g, p), n in sorted(conf.items())},
            "object_self_consistency_recall": round(inter / union, 3) if union else None,
            "object_frames_with_ref": sum(1 for f in rows if f["stored_objects"]),
            "posture": posture,
            "person_box_recall": round(person_recall[0] / person_recall[1], 3)
                                 if person_recall[1] else None,
        }

    out = {
        "test_total": len(frames),
        "sample100": aggregate([f for f in frames if f["image"] in sample100]),
        "all_test": aggregate(frames),
    }
    (RESULTS / "edge_yolo_test_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
