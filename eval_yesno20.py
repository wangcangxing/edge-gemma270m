r"""Edge 的 yes/no 版本：同 20 帧、只输出 lie / not-lie，用于和 Mage 的 0.64 s 对齐。

    python eval_yesno20.py

帧来源与 Mage 重跑一致（JPG 协议）：从 207 帧里取 10 Lie + 10 非 Lie。
提示词只改输出：只回答 yes / no，不要描述。
"""
from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

import torch
from transformers import LogitsProcessor

from compare_mage_fall import find_frame
from gemma_describe import DEFAULT_MODEL, GemmaDescriber
from paths import ADAPTER, POSTURE_WEIGHTS
from scene_yolo import DEFAULT_WEIGHTS, SceneExtractor, load_image

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROMPT_SYS = ("根据场景数据回答是否有人躺在地上，只输出 yes 或 no。示例：\n"
              '数据：{"人物":[{"姿态":"站立"}]} -> no\n'
              '数据：{"人物":[{"姿态":"坐"}]} -> no\n'
              '数据：{"人物":[{"姿态":"躺卧"}]} -> yes')
PROMPT_USR = "数据：{data}\nAnswer yes or no:"


class FirstTokenYesNo(LogitsProcessor):
    """把第一个 token 限制为 yes / no，避免模型继续写描述。"""

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.ids] = 0
        return scores + mask


def yes(ans: str) -> bool | None:
    a = ans.strip().lower()
    if re.match(r"^(yes|y)\b", a):
        return True
    if re.match(r"^(no|n)\b", a):
        return False
    if a.startswith("是"):
        return True
    if a.startswith(("否", "不", "没")):
        return False
    return None


def main() -> None:
    rows = json.loads((RESULTS / "edge_mage_rerun_fall.json").read_text(encoding="utf-8"))["rows"]
    lie = [r for r in rows if r["gt"] == "Lie"][:10]
    other = [r for r in rows if r["gt"] != "Lie"][:10]
    sample = lie + other

    ext = SceneExtractor(DEFAULT_WEIGHTS, str(POSTURE_WEIGHTS), device="cuda:0")
    ext.warmup()
    desc = GemmaDescriber(DEFAULT_MODEL, "cuda:0", adapter=str(ADAPTER))
    desc.warmup()

    out_rows = []
    yes_ids = []
    for w in ("yes", "no"):
        ids = desc.tokenizer.encode(w, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"{w!r} 不是单 token：{ids}")
        yes_ids.append(ids[0])
    proc = FirstTokenYesNo(yes_ids)
    for r in sample:
        img = find_frame(r["video"], r["t"])
        t0 = time.perf_counter()
        scene = ext.extract(load_image(img), posture_whole=True, posture_min_conf=0.25)
        t1 = time.perf_counter()
        messages = [{"role": "system", "content": PROMPT_SYS},
                    {"role": "user", "content": PROMPT_USR.format(
                        data=json.dumps(scene.to_prompt_data(), ensure_ascii=False))}]
        g = desc.generate(messages, max_new_tokens=1, greedy=True,
                          logits_processor=[proc])
        t2 = time.perf_counter()
        pred = yes(g["description"])
        out_rows.append({
            "video": r["video"], "t": r["t"], "gt": r["gt"],
            "pred": pred, "raw": g["description"],
            "posture": scene.persons[0].get("姿态") if scene.persons else None,
            "scene_ms": round((t1 - t0) * 1000, 1),
            "gen_ms": round((t2 - t1) * 1000, 1),
            "gen_tokens": g["gen_tokens"],
        })

    tp = sum(1 for x in out_rows if x["gt"] == "Lie" and x["pred"])
    fn = sum(1 for x in out_rows if x["gt"] == "Lie" and not x["pred"])
    fp = sum(1 for x in out_rows if x["gt"] != "Lie" and x["pred"])
    tn = sum(1 for x in out_rows if x["gt"] != "Lie" and not x["pred"])
    gen = [x["gen_ms"] for x in out_rows]
    total = [x["scene_ms"] + x["gen_ms"] for x in out_rows]
    s = sorted(gen)
    out = {
        "n": len(out_rows), "lie": len(lie), "non_lie": len(other),
        "prompt_sys": PROMPT_SYS,
        "gen_ms": {"median": round(statistics.median(gen), 1),
                   "mean": round(statistics.mean(gen), 1),
                   "std": round(statistics.stdev(gen), 1),
                   "p95": round(s[int(round(0.95 * (len(s) - 1)))], 1)},
        "total_ms_median": round(statistics.median(total), 1),
        "tokens": statistics.mean(x["gen_tokens"] for x in out_rows),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": round(tp / (tp + fn), 3) if tp + fn else 0,
        "precision": round(tp / (tp + fp), 3) if tp + fp else 0,
        "rows": out_rows,
    }
    (RESULTS / "edge_yesno20.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=2))
    for x in out_rows:
        print(f'{x["gt"]:<5} posture={x["posture"]} pred={x["pred"]} '
              f'gen={x["gen_ms"]}ms total={round(x["scene_ms"]+x["gen_ms"],1)}ms raw={x["raw"]!r}')


if __name__ == "__main__":
    main()
