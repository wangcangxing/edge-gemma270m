r"""核对两帧 TP 的人体框 IoU 与姿态置信度。

    python check_two_tp.py
"""
from __future__ import annotations

import json
from pathlib import Path

from compare_mage_fall import find_frame
from paths import POSTURE_WEIGHTS
from scene_yolo import DEFAULT_WEIGHTS, POSTURE_ZH, SceneExtractor, load_image, _iou

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CASES = [("hr_fall_detection_2.mp4", 165), ("fall_detection_10.mp4", 66)]


def gt_lie_boxes(img_path: Path, w: int, h: int):
    lp = Path(str(img_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")
    out = []
    if not lp.exists():
        return out
    for line in lp.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 5 and int(float(p[0])) == 2:
            cx, cy, bw, bh = map(float, p[1:5])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def main() -> None:
    ext = SceneExtractor(DEFAULT_WEIGHTS, str(POSTURE_WEIGHTS), device="cuda:0")
    ext.warmup()
    rows = []
    for video, t in CASES:
        img_path = find_frame(video, t)
        img = load_image(img_path)
        h, w = img.shape[:2]
        gts = gt_lie_boxes(img_path, w, h)
        # 人体检测
        det = ext.model.predict(img, imgsz=640, conf=0.01, device=ext.device,
                                verbose=False)[0]
        persons = [(b, float(cf)) for b, c, cf in
                   zip(det.boxes.xyxy.cpu().numpy(), det.boxes.cls.cpu().numpy(),
                       det.boxes.conf.cpu().numpy()) if int(c) == 0]
        # 姿态检测
        post = ext.posture.predict(img, imgsz=640, conf=0.01, device=ext.device,
                                   verbose=False)[0]
        posts = [(b, POSTURE_ZH.get(str(ext.posture.names[int(c)]), ""), float(cf))
                 for b, c, cf in zip(post.boxes.xyxy.cpu().numpy(),
                                     post.boxes.cls.cpu().numpy(),
                                     post.boxes.conf.cpu().numpy())]

        def best_person(box):
            best = (0.0, 0.0)
            for b, cf in persons:
                v = float(_iou(b, box))
                if v > best[0]:
                    best = (v, cf)
            return best

        def best_posture(box):
            best = (0.0, "", 0.0)
            for b, name, cf in posts:
                v = float(_iou(b, box))
                if v > best[0]:
                    best = (v, name, cf)
            return best

        row = {"video": video, "t": t, "image": img_path.name,
               "gt_lie_boxes": len(gts), "persons": len(persons)}
        if gts:
            iou, cf = best_person(gts[0])
            piou, pname, pcf = best_posture(gts[0])
            row.update({"person_iou_001": round(iou, 3), "person_conf_001": round(cf, 3),
                        "posture_iou": round(piou, 3), "posture": pname,
                        "posture_conf": round(pcf, 3)})
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    (RESULTS / "edge_two_tp_check.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
