r"""诊断 test 592 帧里的 19 个躺卧帧：是没检到人体框，还是框对但姿态判错。

    python diag_lie19.py

姿态用整图模式、conf≥0.25（与 test 指标一致）。对每个 GT 躺卧框：
- 与 COCO 人体框最大 IoU < 0.5  -> 未检出人体框
- 否则看整图姿态模型对应框：无框/低置信 -> 框对、姿态未判
- 判成站立/坐 -> 框对、姿态判错
- 判成躺卧 -> 命中
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from paths import POSTURE_WEIGHTS, SCENES
from scene_yolo import (DEFAULT_WEIGHTS, SceneExtractor, load_image, _iou,
                        POSTURE_ZH)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
GT = {0: "站立", 1: "坐", 2: "躺卧"}


def label_path(img: Path) -> Path:
    return Path(str(img).replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def main() -> None:
    ext = SceneExtractor(DEFAULT_WEIGHTS, str(POSTURE_WEIGHTS), device="cuda:0")
    ext.warmup()

    recs = [json.loads(l) for l in SCENES.read_text(encoding="utf-8").splitlines()]
    test = [r for r in recs if r["split"] == "test"]

    rows = []
    for r in test:
        img_path = Path(r["image"])
        lp = label_path(img_path)
        if not lp.exists():
            continue
        lie_boxes = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 5 and int(float(p[0])) == 2:
                img = load_image(img_path)
                h, w = img.shape[:2]
                cx, cy, bw, bh = map(float, p[1:5])
                lie_boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                  (cx + bw / 2) * w, (cy + bh / 2) * h])
        if not lie_boxes:
            continue

        img = load_image(img_path)
        coco = ext._predict(img)
        det_boxes = [box.tolist() for box, cls in zip(coco.boxes.xyxy.cpu().numpy(),
                                                      coco.boxes.cls.cpu().numpy())
                     if int(cls) == 0]
        post = ext._postures_whole(img)
        low = ext.model.predict(img, imgsz=640, conf=0.01, device=ext.device,
                                verbose=False)[0]
        post_low = ext.posture.predict(img, imgsz=640, conf=0.01,
                                       device=ext.device, verbose=False)[0]
        low_person = [(b, float(cf)) for b, c, cf in
                      zip(low.boxes.xyxy.cpu().numpy(), low.boxes.cls.cpu().numpy(),
                          low.boxes.conf.cpu().numpy()) if int(c) == 0]
        for gb in lie_boxes:
            plow = (None, 0.0, 0.0)
            for b, c, cf in zip(post_low.boxes.xyxy.cpu().numpy(),
                                post_low.boxes.cls.cpu().numpy(),
                                post_low.boxes.conf.cpu().numpy()):
                v = float(_iou(b, gb))
                if v >= 0.3 and float(cf) > plow[1]:
                    plow = (POSTURE_ZH.get(str(ext.posture.names[int(c)]), ""),
                            float(cf), v)
        for gb in lie_boxes:
            bi, biou = None, 0.0
            for i, b in enumerate(det_boxes):
                v = _iou(b, gb)
                if v > biou:
                    biou, bi = v, i
            low_iou, low_conf = 0.0, 0.0
            for b, cf in low_person:
                v = float(_iou(b, gb))
                if v > low_iou:
                    low_iou, low_conf = v, float(cf)
            if biou < 0.5:
                if low_iou >= 0.5:
                    diag = f"0.25 阈值漏检（低分 {low_conf:.2f}）"
                else:
                    diag = "未检出人体框（0.01 也无）"
                pz, pc = None, 0.0
            else:
                pz, pc = None, 0.0
                if post:
                    best = max(post, key=lambda g: _iou(g[0], gb))
                    if _iou(best[0], gb) >= 0.3:
                        pz, pc = best[1], best[2]
                if pz == "躺卧":
                    diag = "命中"
                elif pz is None:
                    diag = "框对、姿态未判"
                else:
                    diag = f"框对、判成{pz}"
            rows.append({"image": img_path.name, "video": r["video"],
                         "iou": round(biou, 3), "person_detected": biou >= 0.5,
                         "low_iou": round(low_iou, 3), "low_conf": round(low_conf, 3),
                         "posture_low": plow[0], "posture_low_conf": round(plow[1], 3),
                         "posture_low_iou": round(plow[2], 3),
                         "posture": pz, "posture_conf": round(pc, 3),
                         "diag": diag})

    summary = Counter(x["diag"] for x in rows)
    out = {"n": len(rows), "summary": dict(summary), "rows": rows}
    (RESULTS / "edge_lie19_diag.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": out["n"], "summary": out["summary"]},
                     ensure_ascii=False, indent=2))
    for x in rows:
        print(f'{x["image"]:<32} iou={x["iou"]:<5} pz={x["posture"]} '
              f'conf={x["posture_conf"]:<5} -> {x["diag"]}')


if __name__ == "__main__":
    main()
