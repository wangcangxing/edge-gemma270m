r"""对 Mage 重跑中 5 个翻转帧做确定性检验。

    python rerun_mage_deterministic.py --mage <Mage-VL> --probe <probe.json>

对每个翻转帧：
- deterministic：torch.use_deterministic_algorithms(True) + cudnn.deterministic + 固定种子，问 3 次
- default：关闭确定性算法，问 3 次
比较是否稳定、以及是否与上一轮重跑（new）一致。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from compare_mage_fall import find_frame  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
Q_LIE = "Is there a person lying on the ground in this image? Answer yes or no."


def yes(ans: str) -> bool | None:
    a = ans.strip().lower()
    if re.match(r"^(yes|y)\b", a):
        return True
    if re.match(r"^(no|n)\b", a):
        return False
    return None


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mage", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    prev = json.loads((RESULTS / "edge_mage_rerun_fall.json").read_text(encoding="utf-8"))["rows"]
    diff = [r for r in prev if bool(r["new_mage_lie"]) != r["old_mage_lie"]]

    from transformers import AutoModelForCausalLM, AutoProcessor

    print("loading ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.mage, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.mage, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
    ).eval()

    def ask(img, deterministic: bool) -> str:
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": Q_LIE},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        return processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip()

    rows = []
    for r in diff:
        img_path = find_frame(r["video"], r["t"])
        img = Image.open(img_path).convert("RGB")
        det = [ask(img, True) for _ in range(args.runs)]
        default = [ask(img, False) for _ in range(args.runs)]
        rows.append({
            "video": r["video"], "t": r["t"], "gt": r["gt"],
            "old": bool(r["old_mage_lie"]), "new": bool(r["new_mage_lie"]),
            "deterministic": det, "default": default,
        })
        print(f'{r["video"]} t={r["t"]} gt={r["gt"]} old={r["old_mage_lie"]} new={r["new_mage_lie"]} '
              f'det={det} default={default}', flush=True)

    out = {"runs": args.runs,
           "deterministic_all_stable": all(len(set(r["deterministic"])) == 1 for r in rows),
           "default_all_stable": all(len(set(r["default"])) == 1 for r in rows),
           "rows": rows}
    (RESULTS / "edge_mage_determinism.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
