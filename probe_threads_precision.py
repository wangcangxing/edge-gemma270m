r"""线程数是否改变输出精度：同一 int8 模型在 1 / 2 / 14 线程下贪心解码对比。

    python probe_threads_precision.py --n 40

判据：以 14 线程为基线，逐字比较 1/2 线程的输出；同时统计人数一致率与物体召回。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import psutil
import torch

from eval_tuned import allowed_objects, mentioned, person_ok
from gemma_describe import DEFAULT_MODEL, GemmaDescriber
from paths import ADAPTER, SCENES
from scene_yolo import COCO_ZH
from targets import _clean_obj
from train_lora import to_prompt_data

ALL_OBJECTS = {v for v in COCO_ZH.values()} - {"人"}
CONFIGS = [("14t", list(range(20)), 14), ("2t", [0, 2], 2), ("1t", [0], 1)]
HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--scenes", default=str(SCENES))
    ap.add_argument("--adapter", default=str(ADAPTER))
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.scenes, encoding="utf-8")]
    test = [r for r in recs if r["split"] == "test"]
    random.Random(1).shuffle(test)
    test = test[: args.n]
    print(f"scenes {len(test)}  persons0={sum(1 for r in test if not r['persons'])}")

    d = GemmaDescriber(DEFAULT_MODEL, "cpu", int8_cpu=True, threads=1,
                       adapter=args.adapter)
    d.warmup()

    outs: dict[str, list[str]] = {}
    stats: dict[str, dict] = {}
    for tag, cpus, threads in CONFIGS:
        psutil.Process().cpu_affinity(cpus)
        torch.set_num_threads(threads)
        d.warmup()
        texts, ok, recall, times = [], [], [], []
        for r in test:
            data = to_prompt_data(r)
            res = d.describe(data, "zh", max_new_tokens=48, greedy=True)
            texts.append(res["description"])
            ok.append(person_ok(res["description"], len(r["persons"])))
            objs = [_clean_obj(o) for o in r["objects"]]
            recall.append(len(mentioned(res["description"], objs)) / len(objs) if objs else 1.0)
            times.append(res["gen_s"])
        outs[tag] = texts
        stats[tag] = {"person_ok": round(sum(ok) / len(ok), 3),
                      "obj_recall": round(sum(recall) / len(recall), 3),
                      "avg_gen_s": round(sum(times) / len(times), 3)}

    base = outs["14t"]
    summary = {"n": len(test), "configs": stats,
               "exact_match_vs_14t": {t: sum(a == b for a, b in zip(outs[t], base)) for t in ("2t", "1t")}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    (res / "edge_thread_precision.json").write_text(
        json.dumps({"summary": summary, "outputs": outs}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    # 展示差异样本
    for t in ("2t", "1t"):
        diff = [(i, base[i], outs[t][i]) for i in range(len(base)) if base[i] != outs[t][i]]
        print(t, "diffs", len(diff))
        for i, a, b in diff[:3]:
            print(" ", i, "|", a, "||", b)


if __name__ == "__main__":
    main()
