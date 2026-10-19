r"""在 Mage 协议的 207 帧上评估不同姿态权重的“躺地”判定（不跑 Gemma，只看检测器）。

    python eval_posture_folds.py --probe <probe.json> --weights A.pt B.pt

用于控制变量：foldA 训练时未见过 hr_fall_detection_1/3（这两个视频在 foldA 的 val 里），
foldB 见过；v5_mixed 见过全部域数据。同一批帧、同一阈值，比较三种权重的召回。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_mage_fall import find_frame, metrics
from paths import POSTURE_WEIGHTS
from scene_yolo import DEFAULT_WEIGHTS, SceneExtractor, load_image

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    args = ap.parse_args()
    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))["rows"]

    out = {}
    for w in args.weights:
        ext = SceneExtractor(DEFAULT_WEIGHTS, w, device="cuda:0")
        ext.warmup()
        rows = []
        for r in probe:
            img = find_frame(r["video"], r["t"])
            if img is None:
                continue
            scene = ext.extract(load_image(img), posture_whole=True, posture_min_conf=0.25)
            posture = scene.persons[0].get("姿态") if scene.persons else None
            rows.append({"gt": r["gt"], "lie": posture == "躺卧", "mage_lie": bool(r["vlm_lie"])})
        out[Path(w).parent.parent.name or Path(w).name] = metrics(rows, "lie")
        print(w, out[Path(w).parent.parent.name or Path(w).name], flush=True)

    (RESULTS / "edge_posture_folds_fall.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
