r"""多人帧（2 人 / 3 人）补充评估：同一 LoRA、GPU bf16、贪心。

test 592 帧里实际只有 6 个 2 人帧、0 个 3 人帧。因此：
- test_2p：全部 6 帧（test 内全部 2 人帧）
- extra_2p：从非 test 部分随机抽 20 帧（仅补样本量，单独报）
- all_3p：全数据集仅 2 帧 3 人（都不在 test），单独报

    python eval_multi_person.py
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from eval_tuned import ALL_OBJECTS, allowed_objects, mentioned, person_ok
from gemma_describe import DEFAULT_MODEL, GemmaDescriber
from paths import ADAPTER, SCENES
from targets import _clean_obj
from train_lora import to_prompt_data

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def evaluate(d: GemmaDescriber, recs: list[dict]) -> dict:
    rows = []
    for r in recs:
        data = to_prompt_data(r)
        out = d.describe(data, "zh", max_new_tokens=48, greedy=True)["description"]
        objs = [_clean_obj(o) for o in r["objects"]]
        allowed = allowed_objects(r)
        hall = [w for w in mentioned(out, sorted(ALL_OBJECTS)) if w not in allowed]
        rows.append({
            "image": Path(r["image"]).name, "video": r["video"], "split": r["split"],
            "n": len(r["persons"]), "desc": out,
            "person_ok": person_ok(out, len(r["persons"])),
            "obj_recall": (len(mentioned(out, objs)) / len(objs)) if objs else 1.0,
            "hallucinated": hall,
        })
    return {
        "n": len(rows),
        "person_ok": round(sum(r["person_ok"] for r in rows) / len(rows), 3),
        "obj_recall": round(statistics.mean(r["obj_recall"] for r in rows), 3),
        "halluc_rate": round(sum(1 for r in rows if r["hallucinated"]) / len(rows), 3),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-2p", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="multi_person")
    args = ap.parse_args()

    recs = [json.loads(l) for l in SCENES.read_text(encoding="utf-8").splitlines()]
    test = [r for r in recs if r["split"] == "test"]
    rest = [r for r in recs if r["split"] != "test"]
    test_2p = [r for r in test if len(r["persons"]) == 2]
    rest_2p = [r for r in rest if len(r["persons"]) == 2]
    random.Random(args.seed).shuffle(rest_2p)
    extra_2p = rest_2p[: args.extra_2p]
    all_3p = [r for r in recs if len(r["persons"]) == 3]

    d = GemmaDescriber(DEFAULT_MODEL, "cuda:0", adapter=str(ADAPTER))
    d.warmup()

    out = {"test_total": len(test),
           "test_counts": {"2p": len(test_2p), "3p": len([r for r in test if len(r['persons']) == 3])},
           "groups": {}}
    for name, group in (("test_2p", test_2p), ("extra_2p", extra_2p), ("all_3p", all_3p)):
        out["groups"][name] = evaluate(d, group) if group else None
        if group:
            g = out["groups"][name]
            print(f"{name}: n={g['n']} person_ok={g['person_ok']} "
                  f"recall={g['obj_recall']} halluc={g['halluc_rate']}", flush=True)

    (RESULTS / f"edge_{args.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("test_2p", "extra_2p", "all_3p"):
        g = out["groups"][name]
        if not g:
            print(f"{name}: 无样本")
            continue
        print(f"\n[{name}] n={g['n']}")
        for r in g["rows"]:
            print(f"- {r['image']} n={r['n']} ok={r['person_ok']} | {r['desc']}")


if __name__ == "__main__":
    main()
