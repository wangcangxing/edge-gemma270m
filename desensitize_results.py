r"""把 results/*.json 里的本机绝对路径替换成占位符（发布前脱敏）。

    $env:EDGE_ASSETS_PREFIX='<你的模型/数据根目录>'; python desensitize_results.py

规则：
    仓库根目录下的路径           -> <REPO>
    EDGE_ASSETS_PREFIX 前缀路径   -> <ASSETS>
    Windows 用户目录             -> <HOME>
    其余盘符路径                  -> <DRIVE>\...
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = str(HERE.parent)
ASSETS = os.environ.get("EDGE_ASSETS_PREFIX", "").rstrip("\\/")
RESULTS = HERE / "results"
USER_RE = re.compile(r"[A-Za-z]:\\Users\\[^\\\"]+")
DRIVE_RE = re.compile(r"^[A-Za-z]:\\")


def clean(text: str) -> str:
    text = text.replace(REPO, "<REPO>")
    if ASSETS:
        text = text.replace(ASSETS, "<ASSETS>")
    text = USER_RE.sub("<HOME>", text)
    text = re.sub(r"[A-Za-z]:\\pagefile\.sys", "<PAGEFILE>", text, flags=re.I)
    return DRIVE_RE.sub(lambda m: "<DRIVE>\\", text)


def walk(obj):
    if isinstance(obj, str):
        return clean(obj)
    if isinstance(obj, list):
        return [walk(x) for x in obj]
    if isinstance(obj, dict):
        return {k: walk(v) for k, v in obj.items()}
    return obj


def main() -> None:
    files = sorted(RESULTS.glob("*.json"))
    for p in files:
        data = json.loads(p.read_text(encoding="utf-8"))
        p.write_text(json.dumps(walk(data), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"cleaned {p.name}")
    print(f"done {len(files)} files")


if __name__ == "__main__":
    main()
