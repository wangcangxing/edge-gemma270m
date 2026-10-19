r"""验证旧 probe 的帧来源：从 MP4 按整数秒解码，再问同样的 Q_LIE。

旧 probe 用 cv2 从视频取 `round(sec*fps)` 帧；新重跑用的是已抽取的 JPG（t+0.25/0.75s）。
这里对 5 个翻转帧复现旧取帧方式，看答案是否回到旧记录。

    python check_mage_old_protocol.py --mage <Mage-VL>
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
Q_LIE = "Is there a person lying on the ground in this image? Answer yes or no."
VIDEO_ROOTS = [Path(r"E:\MageVL\eval\posture\train"), Path(r"E:\MageVL\eval\posture\valid")]


def video_path(name: str) -> Path:
    for root in VIDEO_ROOTS:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


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
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    prev = json.loads((RESULTS / "edge_mage_rerun_fall.json").read_text(encoding="utf-8"))["rows"]
    diff = [r for r in prev if bool(r["new_mage_lie"]) != r["old_mage_lie"]]

    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.mage, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.mage, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
    ).eval()
    torch.use_deterministic_algorithms(True)

    def ask(img) -> str:
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
        name = r["video"].replace(".mp4", "")
        cap = cv2.VideoCapture(str(video_path(f"{name}.mp4")))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(r["t"] * fps)))
        ok, frame = cap.read()
        cap.release()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ans = [ask(img) for _ in range(args.runs)]
        rows.append({"video": r["video"], "t": r["t"], "gt": r["gt"],
                     "old": bool(r["old_mage_lie"]), "new": bool(r["new_mage_lie"]),
                     "mp4_frame": ans})
        print(f'{r["video"]} t={r["t"]} gt={r["gt"]} old={r["old_mage_lie"]} '
              f'new={r["new_mage_lie"]} mp4={ans}', flush=True)

    (RESULTS / "edge_mage_old_protocol.json").write_text(
        json.dumps({"runs": args.runs, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
