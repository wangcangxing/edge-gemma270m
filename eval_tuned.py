r"""测试集评估：基座 vs LoRA 微调（人数一致性、物体召回、幻觉、耗时）。

    python eval_tuned.py --adapter out/lora_gemma270m_zh --n 100
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from gemma_describe import DEFAULT_MODEL, GemmaDescriber
from paths import ADAPTER, SCENES
from scene_yolo import COCO_ZH
from targets import _clean_obj
from train_lora import to_prompt_data

HERE = Path(__file__).resolve().parent
ALL_OBJECTS = {v for v in COCO_ZH.values()} - {"人"}
NO_PERSON = ("没有", "无人", "没人", "空无一人", "看不到")


def allowed_objects(rec: dict) -> set[str]:
    """场景允许出现的物体：全局物体 + 每个人的旁边物体 + 人。"""
    out = {_clean_obj(o) for o in rec.get("objects", [])}
    for p in rec.get("persons", []):
        out |= {_clean_obj(x) for x in p.get("旁边物体", [])}
    out.add("人")
    return out


def person_ok(desc: str, n: int) -> bool:
    if n == 0:
        return any(k in desc for k in NO_PERSON)
    if any(k in desc for k in NO_PERSON):
        return False
    if n == 1:
        return "人" in desc
    words = {2: ("两人", "2人", "两个人", "二人"), 3: ("三人", "3人"),
             4: ("四人", "4人")}
    return any(w in desc for w in words.get(n, (f"{n}人", "人")))


def mentioned(desc: str, objs: list[str]) -> list[str]:
    return [o for o in objs if o and o in desc]


def run(d: GemmaDescriber, recs: list[dict], greedy: bool) -> dict:
    rows = []
    t_all = 0.0
    for rec in recs:
        data = to_prompt_data(rec)
        t0 = time.perf_counter()
        r = d.describe(data, "zh", max_new_tokens=48, greedy=greedy)
        t_all += time.perf_counter() - t0
        objs = [_clean_obj(o) for o in rec["objects"]]
        desc = r["description"]
        allowed = allowed_objects(rec)
        hall = [w for w in mentioned(desc, sorted(ALL_OBJECTS)) if w not in allowed]
        rows.append({"image": Path(rec["image"]).name, "n": len(rec["persons"]),
                     "scene": data, "desc": desc,
                     "person_ok": person_ok(desc, len(rec["persons"])),
                     "obj_recall": (len(mentioned(desc, objs)) / len(objs)) if objs else 1.0,
                     "hallucinated": hall, "len": len(desc),
                     "gen_s": r["gen_s"], "tok_s": r["tok_s"]})
    ok = sum(r["person_ok"] for r in rows) / len(rows)
    recall = sum(r["obj_recall"] for r in rows) / len(rows)
    hall = sum(1 for r in rows if r["hallucinated"]) / len(rows)
    return {"person_ok": round(ok, 3), "obj_recall": round(recall, 3),
            "halluc_rate": round(hall, 3),
            "avg_len": round(sum(r["len"] for r in rows) / len(rows), 1),
            "avg_gen_s": round(t_all / len(rows), 3),
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--scenes", default=str(SCENES))
    ap.add_argument("--adapter", default=str(ADAPTER))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--int8", action="store_true", help="CPU 动态 int8")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--skip-base", action="store_true", help="只评微调模型（省一半时间）")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--tag", default="eval_lora")
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.scenes).read_text(encoding="utf-8").splitlines()]
    test = [r for r in recs if r["split"] == "test"]
    random.Random(1).shuffle(test)
    test = test[: args.n]
    print(f"test scenes {len(test)}  persons={sum(1 for r in test if r['persons'])}")

    b = None
    if not args.skip_base:
        base = GemmaDescriber(args.model, args.device, int8_cpu=args.int8,
                              threads=args.threads)
        base.warmup()
        b = run(base, test, args.greedy)
        del base
        import torch
        torch.cuda.empty_cache()

    tuned = GemmaDescriber(args.model, args.device, int8_cpu=args.int8,
                           threads=args.threads, adapter=args.adapter)
    tuned.warmup()
    t = run(tuned, test, args.greedy)

    out = {"test_n": len(test), "greedy": args.greedy,
           "base": None if b is None else {k: v for k, v in b.items() if k != "rows"},
           "tuned": {k: v for k, v in t.items() if k != "rows"},
           "samples": [{k: s[k] for k in ("image", "n", "desc", "person_ok")} for s in t["rows"][: args.sample]]}
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    (res / f"edge_{args.tag}.json").write_text(
        json.dumps({"summary": out,
                    "base_rows": b["rows"] if b else [],
                    "tuned_rows": t["rows"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    for s in t["rows"][: args.sample]:
        print(f"  n={s['n']} ok={s['person_ok']} | {s['desc']}")


if __name__ == "__main__":
    main()
