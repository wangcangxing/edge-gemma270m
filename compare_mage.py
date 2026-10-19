r"""与 Mage-VL 对齐的控制变量实验：同一批帧、同一条“一句话描述”指令。

    python compare_mage.py --mage <Mage-VL 目录> --n 100

变量控制：
- 帧：dataset/scenes.jsonl 的 test 划分、seed=1 抽 100 帧（与 edge 评估同一批）
- 任务：一句中文描述，20~40 字
- 指标：人数一致率、姿态提及（对/错/未提）、平均字数、单帧延迟、峰值显存
- edge 侧结果直接取 results/edge_bf16_greedy.json（同为 GPU bf16、贪心）
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from eval_tuned import person_ok
from paths import SCENES
from prompting import clean_output
from targets import _clean_obj

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROMPT = "用一句客观的中文描述这张画面，20~40字，只输出这一句，不要解释。"

POSTURE_WORDS = {
    "站立": ("站立", "站着", "站"),
    "坐": ("坐",),
    "躺卧": ("躺卧", "横卧", "躺着", "躺", "卧"),
}
NO_PERSON = ("没有", "无人", "没人", "空无一人", "看不到")
PERSON_WORDS = ("人", "男子", "女子", "男人", "女人", "人员", "身影",
                "老人", "小孩", "孩子", "顾客", "学生")


def person_ok_fair(desc: str, n: int) -> bool:
    """与 edge 评估同口径，但承认“男子/女子/身影”等指人的说法。"""
    if n == 0:
        return any(k in desc for k in NO_PERSON)
    if any(k in desc for k in NO_PERSON):
        return False
    return any(w in desc for w in PERSON_WORDS)


def posture_status(desc: str, gt: str | None) -> str:
    if not gt:
        return "无标注"
    if any(w in desc for w in POSTURE_WORDS[gt]):
        return "正确"
    others = [w for k, ws in POSTURE_WORDS.items() if k != gt for w in ws]
    if any(w in desc for w in others):
        return "错误"
    return "未提"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mage", required=True, help="Mage-VL 权重/代码目录")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--tag", default="edge_vs_mage")
    args = ap.parse_args()

    recs = [json.loads(l) for l in SCENES.read_text(encoding="utf-8").splitlines()]
    test = [r for r in recs if r["split"] == "test"]
    random.Random(1).shuffle(test)
    test = test[: args.n]

    edge_rows = json.loads((RESULTS / "edge_bf16_greedy.json").read_text(encoding="utf-8"))["tuned_rows"]
    edge_by_name = {r["image"]: r for r in edge_rows}

    from transformers import AutoModelForCausalLM, AutoProcessor

    print("loading Mage-VL ...", flush=True)
    t_load = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.mage, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.mage, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
    ).eval()
    load_s = time.perf_counter() - t_load
    print(f"loaded {load_s:.1f}s  device={model.device}", flush=True)

    rows = []
    t_all = 0.0
    for i, rec in enumerate(test, 1):
        name = Path(rec["image"]).name
        img = Image.open(rec["image"]).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0
        t_all += gen_s
        new = out[0, inputs["input_ids"].shape[1]:]
        desc = clean_output(processor.tokenizer.decode(new, skip_special_tokens=True))
        gt_n = len(rec["persons"])
        gt_pos = rec["persons"][0].get("姿态") if rec["persons"] else None
        edge = edge_by_name.get(name, {})
        edge_desc = edge.get("desc", "")
        rows.append({
            "image": name,
            "gt_n": gt_n,
            "gt_posture": gt_pos,
            "mage_desc": desc,
            "edge_desc": edge_desc,
            "mage_person_ok": person_ok_fair(desc, gt_n),
            "edge_person_ok": person_ok_fair(edge_desc, gt_n),
            "mage_posture": posture_status(desc, gt_pos),
            "edge_posture": posture_status(edge_desc, gt_pos),
            "mage_len": len(desc),
            "edge_len": len(edge_desc),
            "mage_gen_s": round(gen_s, 3),
            "edge_gen_s": edge.get("gen_s", 0),
            "mage_peak_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        })
        if i % 10 == 0:
            print(f"{i}/{len(test)}  {gen_s:.2f}s  {desc[:30]}", flush=True)

    def summary(side: str) -> dict:
        return {
            "person_ok": round(sum(r[f"{side}_person_ok"] for r in rows) / len(rows), 3),
            "posture_correct": round(sum(r[f"{side}_posture"] == "正确" for r in rows) / len(rows), 3),
            "posture_wrong": round(sum(r[f"{side}_posture"] == "错误" for r in rows) / len(rows), 3),
            "posture_missed": round(sum(r[f"{side}_posture"] == "未提" for r in rows) / len(rows), 3),
            "avg_len": round(statistics.mean(r[f"{side}_len"] for r in rows), 1),
            "avg_gen_s": round(statistics.mean(r[f"{side}_gen_s"] for r in rows), 3),
        }

    out = {
        "n": len(rows),
        "load_s": round(load_s, 1),
        "prompt": PROMPT,
        "mage": summary("mage"),
        "edge": summary("edge"),
        "mage_peak_gb": round(max(r["mage_peak_gb"] for r in rows), 2),
        "edge_peak_gb": 0.55,
        "rows": rows,
    }
    (RESULTS / f"edge_{args.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=2))
    for r in rows[:6]:
        print(f"- {r['image']} n={r['gt_n']} gt={r['gt_posture']}")
        print(f"  mage: {r['mage_desc']}")
        print(f"  edge: {r['edge_desc']}")


if __name__ == "__main__":
    main()
