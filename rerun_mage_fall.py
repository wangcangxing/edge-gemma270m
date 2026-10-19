r"""重跑 Mage-VL：在 207 帧对齐测试集上回答“是否有人躺在地上”。

    python rerun_mage_fall.py --mage <Mage-VL> --probe <probe_vlm_crossdomain.json>

问题与旧 probe 完全一致（Q_LIE / Q_PERSON），用于替换上一轮复用旧答案的做法。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from compare_mage_fall import find_frame, metrics

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
Q_LIE = "Is there a person lying on the ground in this image? Answer yes or no."
Q_PERSON = "Is there a person in this image? Answer yes or no."


def yes(ans: str) -> bool | None:
    a = ans.strip().lower()
    if re.match(r"^(yes|y)\b", a):
        return True
    if re.match(r"^(no|n)\b", a):
        return False
    if a.startswith(("是", "有")):
        return True
    if a.startswith(("否", "没", "不")):
        return False
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mage", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    args = ap.parse_args()

    rows_src = json.loads(Path(args.probe).read_text(encoding="utf-8"))["rows"]

    from transformers import AutoModelForCausalLM, AutoProcessor

    print("loading Mage-VL ...", flush=True)
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.mage, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.mage, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
    ).eval()
    load_s = time.perf_counter() - t0
    print(f"loaded {load_s:.1f}s", flush=True)

    rows, missing = [], 0
    t_lie, t_person = [], []
    peak = 0.0
    for i, r in enumerate(rows_src, 1):
        img_path = find_frame(r["video"], r["t"])
        if img_path is None:
            missing += 1
            continue
        img = Image.open(img_path).convert("RGB")
        answers = {}
        for key, question in (("lie", Q_LIE), ("person", Q_PERSON)):
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": question},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            peak = max(peak, torch.cuda.max_memory_allocated() / 1024 ** 3)
            ans = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                             skip_special_tokens=True).strip()
            answers[key] = {"raw": ans, "yes": yes(ans), "s": round(dt, 3)}
            (t_lie if key == "lie" else t_person).append(dt)
        rows.append({"video": r["video"], "t": r["t"], "gt": r["gt"],
                     "old_mage_lie": bool(r["vlm_lie"]),
                     "new_mage_lie": answers["lie"]["yes"],
                     "new_mage_person": answers["person"]["yes"],
                     "raw_lie": answers["lie"]["raw"],
                     "raw_person": answers["person"]["raw"]})
        if i % 20 == 0:
            print(f"{i}/{len(rows_src)} lie={answers['lie']['yes']} "
                  f"({answers['lie']['s']}s) gt={r['gt']} raw={answers['lie']['raw']!r}",
                  flush=True)

    lie_rows = [{"gt": r["gt"], "lie": bool(r["new_mage_lie"])} for r in rows]
    old_rows = [{"gt": r["gt"], "lie": r["old_mage_lie"]} for r in rows]
    out = {
        "n": len(rows), "missing": missing, "load_s": round(load_s, 1),
        "counts": {k: sum(1 for r in rows if r["gt"] == k) for k in ("Lie", "Stand", "Sit")},
        "new": metrics(lie_rows, "lie"),
        "old": metrics(old_rows, "lie"),
        "agree_old_new": round(sum(1 for r in rows if bool(r["new_mage_lie"]) == r["old_mage_lie"]) / len(rows), 3),
        "lie_latency_median": round(statistics.median(t_lie), 3),
        "person_latency_median": round(statistics.median(t_person), 3),
        "peak_gb": round(peak, 2),
        "rows": rows,
    }
    (RESULTS / "edge_mage_rerun_fall.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
