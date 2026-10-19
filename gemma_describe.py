"""Gemma-3-270M-IT：把场景 JSON 说成一句中文。

这是语义层的第二半，也是被压缩候选：270M 纯文本、无视觉塔，权重约 0.54 GB(bf16)。
它读不到像素，只能读 scene_yolo.py 给出的符号事实 —— 这正是能在端侧跑起来的原因。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompting import build_messages, clean_output
from paths import GEMMA_MODEL

DEFAULT_MODEL = str(GEMMA_MODEL)


class GemmaDescriber:
    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "cuda:0",
                 int8_cpu: bool = False, threads: int = 14,
                 adapter: str = "") -> None:
        self.device = device
        if device == "cpu":
            torch.set_num_threads(threads)
            dtype = torch.float32
        else:
            dtype = torch.bfloat16
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        self.model.to(device).eval()
        self.adapter = adapter
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.merge_and_unload()
            self.model.to(device, dtype=dtype).eval()
        self.load_s = time.perf_counter() - t0
        self.int8 = False
        if int8_cpu and device == "cpu":
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8)
            self.int8 = True
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            self.mem_after_load_gb = torch.cuda.memory_allocated() / 1024 ** 3
        else:
            self.mem_after_load_gb = 0.0

    def warmup(self) -> None:
        enc = self.tokenizer("hi", return_tensors="pt").to(self.device)
        with torch.inference_mode():
            self.model.generate(**enc, max_new_tokens=2, do_sample=False,
                                pad_token_id=self.tokenizer.pad_token_id or 0)

    def describe(self, scene_data: dict, lang: str = "zh", max_new_tokens: int = 64,
                 temperature: float = 0.4, greedy: bool = False) -> dict:
        messages = build_messages(scene_data, lang)
        r = self.generate(messages, max_new_tokens=max_new_tokens,
                          temperature=temperature, greedy=greedy)
        r["lang"] = lang
        return r

    def generate(self, messages: list[dict], max_new_tokens: int = 64,
                 temperature: float = 0.4, greedy: bool = False,
                 prefix: str = "") -> dict:
        enc = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, tokenize=True)
        if prefix:
            pids = self.tokenizer(prefix, add_special_tokens=False,
                                  return_tensors="pt").input_ids
            enc["input_ids"] = torch.cat([enc["input_ids"], pids], dim=1)
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        enc = {k: v.to(self.device) for k, v in enc.items()}
        n_in = int(enc["input_ids"].shape[1])

        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=not greedy,
                      pad_token_id=self.tokenizer.pad_token_id or 0)
        if not greedy:
            kwargs.update(temperature=temperature, top_p=0.9,
                          repetition_penalty=1.05)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**enc, **kwargs)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0

        new_ids = out[0, n_in:]
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        n_out = int(new_ids.shape[0])
        peak_gb = (torch.cuda.max_memory_allocated() / 1024 ** 3
                   if self.device.startswith("cuda") else 0.0)
        return {
            "description": clean_output(raw),
            "raw": raw,
            "prompt_tokens": n_in,
            "gen_tokens": n_out,
            "gen_s": round(gen_s, 3),
            "tok_s": round(n_out / gen_s, 2) if gen_s > 0 else None,
            "peak_gpu_gb": round(peak_gb, 2),
            "device": self.device,
            "int8": self.int8,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="场景 JSON → 一句语义描述（Gemma-3-270M）")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--scene-json", required=True, help="JSON 字符串或 .json 文件路径")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--adapter", default="", help="LoRA adapter 目录")
    ap.add_argument("--int8", action="store_true", help="CPU 动态 int8 量化")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    p = Path(args.scene_json)
    data = json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else json.loads(args.scene_json)
    # 兼容 Scene 完整 JSON：只取提示词用的精简版
    if "persons" in data and "人物" not in data:
        data = {"画面": f'{data["width"]}x{data["height"]}，{data["light"]}',
                "人物": data["persons"], "物体": data["objects"]}

    d = GemmaDescriber(args.model, args.device, args.int8, adapter=args.adapter)
    d.warmup()
    t0 = time.perf_counter()
    r = d.describe(data, args.lang, args.max_new_tokens, greedy=args.greedy)
    r["wall_s"] = round(time.perf_counter() - t0, 3)
    r["load_s"] = round(d.load_s, 2)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
