r"""YOLOv10n → 结构化场景 JSON。

语义层的第一半：把像素压成"谁、在哪、旁边有什么"的符号事实。
Gemma-3-270M 是纯文本模型（无视觉塔），所以它读到的是这里的 JSON，不是图片。

用法：
    python scene_yolo.py --image path/to.jpg --json
    python scene_yolo.py --image path/to.jpg --posture-weights path/to/posture_best.pt
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO / "yolov10n.pt"

# COCO 80 类 → 中文。检测器只认这 80 类，语义层不越界描述。
COCO_ZH = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车",
    "airplane": "飞机", "bus": "公交车", "train": "火车", "truck": "卡车",
    "boat": "船", "traffic light": "红绿灯", "fire hydrant": "消防栓",
    "stop sign": "停车标志", "parking meter": "停车计时器", "bench": "长椅",
    "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马", "sheep": "羊",
    "cow": "牛", "elephant": "大象", "bear": "熊", "zebra": "斑马",
    "giraffe": "长颈鹿", "backpack": "背包", "umbrella": "雨伞",
    "handbag": "手提包", "tie": "领带", "suitcase": "行李箱", "frisbee": "飞盘",
    "skis": "滑雪板", "snowboard": "单板滑雪板", "sports ball": "球",
    "kite": "风筝", "baseball bat": "棒球棒", "baseball glove": "棒球手套",
    "skateboard": "滑板", "surfboard": "冲浪板", "tennis racket": "网球拍",
    "bottle": "瓶子", "wine glass": "酒杯", "cup": "杯子", "fork": "叉子",
    "knife": "刀", "spoon": "勺子", "bowl": "碗", "banana": "香蕉",
    "apple": "苹果", "sandwich": "三明治", "orange": "橙子", "broccoli": "西兰花",
    "carrot": "胡萝卜", "hot dog": "热狗", "pizza": "披萨", "donut": "甜甜圈",
    "cake": "蛋糕", "chair": "椅子", "couch": "沙发", "potted plant": "盆栽",
    "bed": "床", "dining table": "餐桌", "toilet": "马桶", "tv": "电视",
    "laptop": "笔记本电脑", "mouse": "鼠标", "remote": "遥控器",
    "keyboard": "键盘", "cell phone": "手机", "microwave": "微波炉",
    "oven": "烤箱", "toaster": "烤面包机", "sink": "水槽", "refrigerator": "冰箱",
    "book": "书", "clock": "时钟", "vase": "花瓶", "scissors": "剪刀",
    "teddy bear": "泰迪熊", "hair drier": "吹风机", "toothbrush": "牙刷",
}

POSTURE_ZH = {"stand": "站立", "sit": "坐", "lie": "躺卧"}

# 人脚位置低于画面 82% 视为"接近地面"（躺/坐在地上与坐在沙发上的粗线索）
GROUND_RATIO = 0.82


def _zone(cx: float, cy: float, w: int, h: int) -> str:
    hx = "左" if cx < w / 3 else ("右" if cx > 2 * w / 3 else "中")
    vy = "上" if cy < h / 3 else ("下" if cy > 2 * h / 3 else "中")
    return f"{hx}{vy}"


def _dist_bucket(d: float) -> str:
    if d < 0.15:
        return "旁边"
    if d < 0.40:
        return "附近"
    return "远处"


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _brightness(img: np.ndarray) -> str:
    y = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    if y < 60:
        return "偏暗"
    if y > 180:
        return "偏亮"
    return "正常"


@dataclass
class Scene:
    width: int
    height: int
    light: str
    persons: list[dict[str, Any]] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    person_boxes: list[list[float]] = field(default_factory=list)

    def to_prompt_data(self) -> dict[str, Any]:
        """给语言模型的精简版：只留能落成句子的字段。"""
        return {
            "画面": f"{self.width}x{self.height}，{self.light}",
            "人物": self.persons,
            "物体": self.objects,
        }

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class SceneExtractor:
    """YOLOv10n 检测器 + 可选的姿态权重；两次推理都在同一张图上。"""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS,
                 posture_weights: str | Path | None = None,
                 device: str | int = "cuda:0", imgsz: int = 640,
                 conf: float = 0.25, iou: float = 0.5) -> None:
        from ultralytics import YOLO

        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.model = YOLO(str(weights))
        self.posture = YOLO(str(posture_weights)) if posture_weights else None
        self.names = self.model.names

    def warmup(self) -> None:
        """首帧会触发 CUDA 初始化，必须先跑一次空图，否则延迟数据全是启动开销。"""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self._predict(dummy)

    # -- 检测 ---------------------------------------------------------
    def _predict(self, img: np.ndarray) -> Any:
        return self.model.predict(img, imgsz=self.imgsz, conf=self.conf,
                                  iou=self.iou, device=self.device,
                                  verbose=False)[0]

    def _posture_of(self, img: np.ndarray, box: np.ndarray) -> tuple[str, float]:
        """在扩充 15% 的人体框上跑姿态模型，取最高分。"""
        if self.posture is None:
            return "", 0.0
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ex, ey = bw * 0.65, bh * 0.15
        X1, Y1 = int(max(0, cx - ex)), int(max(0, cy - ey))
        X2, Y2 = int(min(w, cx + ex)), int(min(h, cy + ey))
        crop = img[Y1:Y2, X1:X2]
        if crop.size == 0:
            return "", 0.0
        r = self.posture.predict(crop, imgsz=320, conf=0.25, device=self.device,
                                 verbose=False)[0]
        if not len(r.boxes):
            return "", 0.0
        best = int(r.boxes.cls[0].item())
        return POSTURE_ZH.get(str(self.posture.names[best]), ""), float(r.boxes.conf[0])

    # -- 场景化 -------------------------------------------------------
    def _postures_whole(self, img: np.ndarray) -> list[tuple[np.ndarray, str, float]]:
        """整图跑姿态模型（与训练分布一致），再按 IoU 分配给人体框。"""
        r = self.posture.predict(img, imgsz=640, conf=0.25, device=self.device,
                                 verbose=False)[0]
        out = []
        for box, cls, cf in zip(r.boxes.xyxy.cpu().numpy(),
                                r.boxes.cls.cpu().numpy(),
                                r.boxes.conf.cpu().numpy()):
            name = POSTURE_ZH.get(str(self.posture.names[int(cls)]), "")
            out.append((box, name, float(cf)))
        return out

    def extract(self, img: np.ndarray, with_posture: bool = True,
                posture_whole: bool = False,
                posture_min_conf: float = 0.55) -> Scene:
        h, w = img.shape[:2]
        r = self._predict(img)
        whole = self._postures_whole(img) if (with_posture and posture_whole) else []
        persons: list[dict[str, Any]] = []
        objs: list[tuple[str, np.ndarray, float]] = []

        for box, cls, cf in zip(r.boxes.xyxy.cpu().numpy(),
                                r.boxes.cls.cpu().numpy(),
                                r.boxes.conf.cpu().numpy()):
            name = str(self.names[int(cls)])
            zh = COCO_ZH.get(name, name)
            x1, y1, x2, y2 = [float(v) for v in box]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if name == "person":
                p = {
                    "位置": _zone(cx, cy, w, h),
                    "身高占比": round((y2 - y1) / h, 2),
                    "脚部接近地面": bool(y2 >= GROUND_RATIO * h),
                }
                if with_posture:
                    if whole:
                        best = max(whole, key=lambda g: _iou(g[0], box))
                        pz, conf = (best[1], best[2]) if _iou(best[0], box) >= 0.3 else ("", 0.0)
                    else:
                        pz, conf = self._posture_of(img, box)
                    if pz and conf >= posture_min_conf:
                        p["姿态"] = pz
                p["_box"] = (x1, y1, x2, y2)
                persons.append(p)
            else:
                objs.append((zh, np.array([x1, y1, x2, y2]), float(cf)))

        # 物体按"离最近的人"折算距离；无人则全部记为画面内。
        boxes_out: list[list[float]] = []
        for p in persons:
            px1, py1, px2, py2 = p.pop("_box")
            boxes_out.append([round(px1), round(py1), round(px2), round(py2)])
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            scored = []
            for zh, b, cf in objs:
                ocx, ocy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                d = float(np.hypot((ocx - pcx) / w, (ocy - pcy) / h))
                scored.append((d, zh))
            scored.sort()
            seen: set[str] = set()
            near: list[str] = []
            for d, zh in scored:
                if zh in seen:
                    continue
                seen.add(zh)
                near.append(f"{zh}({_dist_bucket(d)})")
                if len(near) == 3:
                    break
            p["旁边物体"] = near

        counts: dict[str, list[float]] = {}
        for zh, b, cf in objs:
            counts.setdefault(zh, []).append(cf)
        # 同类取最高分一次，最多 6 类，避免提示词里堆重复名词
        objects = [f"{zh}x{len(v)}" if len(v) > 1 else zh
                   for zh, v in sorted(counts.items(), key=lambda kv: -max(kv[1]))[:6]]

        return Scene(width=w, height=h, light=_brightness(img),
                     persons=persons, objects=objects, person_boxes=boxes_out)


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读不到图片: {path}")
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLOv10n → 场景 JSON")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--posture-weights", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    args = ap.parse_args()

    ex = SceneExtractor(args.weights, args.posture_weights or None,
                        device=args.device, imgsz=args.imgsz)
    ex.warmup()
    img = load_image(args.image)
    t0 = time.perf_counter()
    scene = ex.extract(img)
    dt = (time.perf_counter() - t0) * 1000
    if args.json:
        print(scene.to_json())
    else:
        print(f"{dt:.1f} ms")
        print(json.dumps(scene.to_prompt_data(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
