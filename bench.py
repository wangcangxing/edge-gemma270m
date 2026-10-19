r"""复现延迟 / 显存 / 体积：GPU bf16、CPU fp32、CPU int8 三档。

    python bench.py --adapter out/lora_gemma270m_zh
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from gemma_describe import DEFAULT_MODEL, GemmaDescriber
from paths import ADAPTER

HERE = Path(__file__).resolve().parent
SCENE = {"画面": "1280x720，正常",
         "人物": [{"位置": "左侧下", "身高占比": 0.43, "脚部接近地面": True, "姿态": "躺卧"}],
         "物体": ["沙发x2", "茶几"]}


def bytes_of(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def one(tag: str, device: str, int8: bool, adapter: str, runs: int) -> dict:
    d = GemmaDescriber(DEFAULT_MODEL, device, int8, adapter=adapter)
    d.warmup()
    times, toks, outs = [], [], []
    for _ in range(runs):
        r = d.describe(SCENE, "zh", max_new_tokens=48, greedy=True)
        times.append(r["gen_s"])
        toks.append(r["gen_tokens"])
        outs.append(r["description"])
    return {"tag": tag, "device": device, "int8": int8,
            "load_s": round(d.load_s, 2),
            "gen_s_median": round(statistics.median(times), 3),
            "tok_s": round(statistics.mean(t for t in toks) / statistics.median(times), 1),
            "model_mem_gb": round(bytes_of(d.model) / 1024 ** 3, 2),
            "sample": outs[-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ADAPTER))
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    rows = [
        one("gpu_bf16", "cuda:0", False, args.adapter, args.runs),
        one("cpu_fp32", "cpu", False, args.adapter, args.runs),
        one("cpu_int8", "cpu", True, args.adapter, args.runs),
    ]
    base_size = Path(DEFAULT_MODEL, "model.safetensors").stat().st_size / 1024 ** 3
    adapter_size = Path(args.adapter, "adapter_model.safetensors").stat().st_size / 1024 ** 2
    out = {"base_weights_gb": round(base_size, 3),
           "adapter_mb": round(adapter_size, 1), "rows": rows}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    (res / "edge_bench.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
