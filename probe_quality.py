"""一次性探针：270M 在几种提示词下的实际输出与稳定延迟（不进管线）。"""
import json
import time
from pathlib import Path

from gemma_describe import GemmaDescriber
from prompting import build_messages, scene_json

HERE = Path(__file__).resolve().parent
SCENES = [
    {"画面": "1280x720，正常",
     "人物": [{"位置": "左侧下", "身高占比": 0.43, "脚部接近地面": True, "姿态": "躺卧"}],
     "物体": ["沙发x2", "茶几"]},
    {"画面": "1280x720，偏暗",
     "人物": [{"位置": "右中", "身高占比": 0.55, "脚部接近地面": False, "姿态": "站立"}],
     "物体": ["床", "盆栽"]},
    {"画面": "1920x1080，正常", "人物": [], "物体": ["椅子x3", "餐桌"]},
]


def run(d, name, messages, prefix="", greedy=False):
    t0 = time.perf_counter()
    r = d.generate(messages, max_new_tokens=64, greedy=greedy, prefix=prefix)
    print(f"--- {name} ({r['gen_s']}s, {r['tok_s']} tok/s) ---")
    print(r["description"] or "(空)")
    return r


def main():
    d = GemmaDescriber(device="cuda:0")
    d.warmup()
    t0 = time.perf_counter()
    for i, scene in enumerate(SCENES):
        run(d, f"fewshot zh #{i}", build_messages(scene, "zh"))
    run(d, "fewshot zh greedy", build_messages(SCENES[0], "zh"), greedy=True)
    run(d, "fewshot zh + prefix", build_messages(SCENES[0], "zh"), prefix="描述：")
    run(d, "fewshot en", build_messages(SCENES[0], "en"))

    mini = [{"role": "system", "content": "只输出一句中文描述，不超过40字，不解释。"},
            {"role": "user", "content": f"场景：{scene_json(SCENES[0])}\n描述："}]
    run(d, "minimal zh", mini)

    direct = [{"role": "user",
               "content": f"根据下面的检测数据写一句中文描述：{scene_json(SCENES[0])}\n描述："}]
    run(d, "direct zh", direct)
    print(f"total {time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    main()
