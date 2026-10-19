"""场景 JSON → 目标句（模板）。

270M 的基座几乎不会按指令改写 JSON（见 probe 结果），所以先用确定性模板造监督数据，
再让 LoRA 学会这条映射；推理时模型输出的是它自己组织的句子，不是模板拼接。
目标句只允许包含场景里已有的信息：人数、位置、姿态、旁边物体、亮度。
"""
from __future__ import annotations

import random
from typing import Any

# 位置词不带"画面"前缀，由模板决定怎么接，避免"画面中央有人在画面中央"这类重复
POS_ZH = {
    "左上": "左上方", "中上": "上方", "右上": "右上方",
    "左中": "左侧", "中中": "中央", "右中": "右侧",
    "左下": "左下方", "中下": "下方", "右下": "右下方",
}


def _clean_obj(name: str) -> str:
    name = name.split("(")[0]
    if "x" in name:
        base, _, n = name.partition("x")
        if n.isdigit():
            return base
    return name


def _objects(scene: dict[str, Any]) -> list[str]:
    persons = scene.get("人物") or []
    if persons and persons[0].get("旁边物体"):
        return [_clean_obj(x) for x in persons[0]["旁边物体"]]
    return [_clean_obj(x) for x in (scene.get("物体") or [])]


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]}和{items[1]}"
    return "、".join(items[:-1]) + "和" + items[-1]


def make_target(scene: dict[str, Any], rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    persons = scene.get("人物") or []
    objs = _objects(scene)
    obj_s = _join(objs)

    if not persons:
        if not objs:
            return rng.choice([
                "画面中没有看到人，室内空无一人。",
                "画面里没有人。",
            ])
        return rng.choice([
            f"画面中没有看到人，只有{obj_s}。",
            f"画面里没有看到人，{obj_s}都很清楚。",
            f"画面上看不到人，{obj_s}清晰可见。",
        ])

    if len(persons) == 1:
        p = persons[0]
        pos = POS_ZH.get(p.get("位置", ""), "中央")
        pz = p.get("姿态", "")
        cands = []
        if pz:
            if obj_s:
                cands += [
                    f"画面{pos}有一人{pz}，旁边是{obj_s}。",
                    f"一个人{pz}在画面{pos}，附近有{obj_s}。",
                    f"画面{pos}有人{pz}，{obj_s}就在旁边。",
                ]
            else:
                cands += [f"画面{pos}有一人{pz}。", f"有一个人{pz}，位于画面{pos}。"]
        else:
            if obj_s:
                cands += [
                    f"画面{pos}有一人，旁边是{obj_s}。",
                    f"画面里有人在画面{pos}，附近有{obj_s}。",
                ]
            else:
                cands += [f"画面{pos}有一人。"]
        if scene.get("画面", "").endswith("偏暗"):
            cands.append(f"光线偏暗，画面{pos}有一人{pz}。" if pz
                         else f"光线偏暗，画面{pos}有一人。")
        return rng.choice(cands)

    # 多人：最多描述两个人，避免句子过长
    parts = []
    for p in persons[:2]:
        pos = POS_ZH.get(p.get("位置", ""), "中央")
        pz = p.get("姿态", "")
        parts.append(f"画面{pos}一人{pz}".strip())
    return f"画面中有{len(persons)}人，{'，'.join(parts)}。"
