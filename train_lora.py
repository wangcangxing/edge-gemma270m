r"""LoRA 微调 Gemma-3-270M：场景 JSON → 一句中文描述。

基座实测不会按指令改写 JSON（probe_quality.py），所以用模板造监督数据教这条映射。
只训练 LoRA（约 2% 参数），基座权重不动，输出 adapter 约 30 MB。

    python train_lora.py --epochs 2 --out out/lora_gemma270m_zh
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma_describe import DEFAULT_MODEL
from paths import ADAPTER, SCENES
from prompting import build_messages
from targets import make_target

HERE = Path(__file__).resolve().parent


def lm_loss(model, batch: dict) -> torch.Tensor:
    """只在目标 token 上算 lm_head + CE。

    整段 forward 只为 262k 词表的 lm_head 会在 8x320 上吃掉 >10 GB 显存；
    这里取 hidden[:, :-1] 后按 label 掩码只算 ~30 个位置，显存降到 MB 级。
    """
    base = model.base_model.model
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = base.model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            use_cache=False).last_hidden_state
    h = hidden[:, :-1]
    y = batch["labels"][:, 1:]
    mask = y != -100
    if not bool(mask.any()):
        return torch.zeros((), device=hidden.device, requires_grad=True)
    return F.cross_entropy(base.lm_head(h[mask]).float(), y[mask])


def to_prompt_data(rec: dict) -> dict:
    return {"画面": f'{rec["width"]}x{rec["height"]}，{rec["light"]}',
            "人物": rec["persons"], "物体": rec["objects"]}


class Batcher:
    def __init__(self, scenes: list[dict], tok, max_len: int) -> None:
        self.scenes = scenes
        self.tok = tok
        self.max_len = max_len
        self.eos = tok.eos_token_id
        self._pids: dict[str, list[int]] = {}

    def prompt_ids(self, rec: dict) -> list[int]:
        key = rec["image"]
        if key not in self._pids:
            msgs = build_messages(to_prompt_data(rec), "zh")
            pids = self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True)["input_ids"]
            self._pids[key] = pids.tolist() if hasattr(pids, "tolist") else list(pids)
        return self._pids[key]

    def encode(self, rec: dict, seed: int) -> tuple[list[int], list[int]]:
        data = to_prompt_data(rec)
        target = make_target(data, random.Random(seed))
        pids = self.prompt_ids(rec)
        tids = self.tok(target, add_special_tokens=False).input_ids + [self.eos]
        ids = (pids + tids)[: self.max_len]
        labels = ([-100] * len(pids) + tids)[: self.max_len]
        return ids, labels

    def batch(self, recs: list[dict], epoch: int) -> dict:
        seqs, labs = [], []
        for i, r in enumerate(recs):
            ids, lb = self.encode(r, hash((r["image"], epoch, i)) & 0xFFFF)
            seqs.append(ids)
            labs.append(lb)
        n = max(len(x) for x in seqs)
        pad = self.tok.pad_token_id or 0
        input_ids = torch.tensor([x + [pad] * (n - len(x)) for x in seqs])
        labels = torch.tensor([x + [-100] * (n - len(x)) for x in labs])
        attn = (input_ids != pad).long()
        return {"input_ids": input_ids.cuda(), "attention_mask": attn.cuda(),
                "labels": labels.cuda()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--scenes", default=str(SCENES))
    ap.add_argument("--out", default=str(ADAPTER))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    recs = [json.loads(l) for l in Path(args.scenes).read_text(encoding="utf-8").splitlines()]
    train = [r for r in recs if r["split"] == "train"]
    val = [r for r in recs if r["split"] == "val"]
    if args.max_samples:
        train = train[: args.max_samples]
    print(f"train {len(train)} / val {len(val)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    base.config.use_cache = False
    cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, cfg)
    # LoRA 参数保持 fp32，前向用 bf16 autocast，优化器状态稳定
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01, eps=1e-8)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = int(steps_per_epoch * args.epochs)
    warmup = max(1, total_steps // 20)

    def lr_at(s: int) -> float:
        if s < warmup:
            return args.lr * (s + 1) / warmup
        p = (s - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    batcher = Batcher(train, tok, args.max_len)
    val_batcher = Batcher(val, tok, args.max_len) if val else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model.train()
    t0 = time.perf_counter()
    step = 0
    best_val = float("inf")
    for epoch in range(math.ceil(args.epochs)):
        rng = random.Random(args.seed + epoch)
        rng.shuffle(train)
        for i in range(0, len(train), args.batch_size):
            batch = batcher.batch(train[i:i + args.batch_size], epoch)
            loss = lm_loss(model, batch) / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step % 20 == 0:
                print(f"step {step}/{total_steps} loss {loss.item()*args.grad_accum:.3f} "
                      f"lr {lr_at(step):.1e} {time.perf_counter()-t0:.0f}s", flush=True)
            step += 1
            if val_batcher and step % args.eval_every == 0:
                model.eval()
                with torch.inference_mode():
                    vloss = 0.0
                    for j in range(0, len(val), args.batch_size):
                        vb = val_batcher.batch(val[j:j + args.batch_size], 0)
                        vloss += lm_loss(model, vb).item()
                    vloss /= math.ceil(len(val) / args.batch_size)
                print(f"  val loss {vloss:.3f}", flush=True)
                if vloss < best_val:
                    best_val = vloss
                    model.save_pretrained(out)
                    tok.save_pretrained(out)
                model.train()
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    model.save_pretrained(out)
    tok.save_pretrained(out)
    meta = {"train": len(train), "val": len(val), "steps": step,
            "epochs": args.epochs, "lr": args.lr, "batch": args.batch_size,
            "grad_accum": args.grad_accum, "best_val": best_val,
            "seconds": round(time.perf_counter() - t0, 1)}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {out}  {meta}")


if __name__ == "__main__":
    main()
