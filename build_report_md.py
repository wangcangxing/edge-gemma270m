r"""把 Word 报告转成 Markdown（仓库入库的副本）。

    python build_report_md.py

依赖 pandoc（PATH 里或设 PANDOC 环境变量）。生成后用几处后处理修正：
标题层级、附录里的代码块（pandoc 会把 # / $ / <> 转义）。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCX = HERE / "端侧语义描述实测报告.docx"
MD = HERE / "端侧语义描述实测报告.md"

APPENDIX_CODE = """```powershell
# 在仓库根目录执行；$py 换成你的 Python/venv
$py = 'python'
$adapter = 'out/lora_gemma270m_zh'

# 权重：下载到 models/gemma-3-270m-it（或设 EDGE_GEMMA_MODEL）
# 数据：把抽帧目录设到 EDGE_DOMAIN_IMAGES，再构建 scenes.jsonl
& $py build_dataset.py
& $py train_lora.py --epochs 3 --out $adapter

# 评估 / 基准
& $py eval_tuned.py --adapter $adapter --n 100
& $py bench.py --runs 3
$env:EDGE_CPUS = '0,2'; & $py bench_2c.py --image <IMAGE>

# 端到端
& $py pipeline.py --source <IMAGE> --adapter $adapter
& $py pipeline.py --source <IMAGE> --yolo-device cpu --gemma-device cpu --int8 --adapter $adapter
```"""


def main() -> None:
    pandoc = os.environ.get("PANDOC") or shutil.which("pandoc")
    if not pandoc:
        raise SystemExit("找不到 pandoc：装好或设 PANDOC 指向 pandoc.exe")
    if not DOCX.exists():
        raise SystemExit(f"缺 {DOCX.name}：先跑 build_report.py")

    subprocess.run([pandoc, str(DOCX), "-f", "docx", "-t", "gfm",
                    "--wrap=none", "-o", str(MD)], check=True)
    lines = MD.read_text(encoding="utf-8").splitlines()

    # 标题：首行加粗标题 → H1，副标题 → 引用
    if lines and lines[0].startswith("**端侧语义描述实测报告**"):
        lines[0] = "# 端侧语义描述实测报告"
        for i in range(1, min(5, len(lines))):
            if lines[i].strip():
                lines[i] = "> " + lines[i].strip()
                break

    # 标题降一级（原来的 H1→H2、H2→H3；保留首行标题）
    shifted = [lines[0]] if lines else []
    for ln in lines[1:]:
        if ln.startswith("## "):
            shifted.append("#" + ln)
        elif ln.startswith("# "):
            shifted.append("#" + ln)
        else:
            shifted.append(ln)
    lines = shifted

    # 附录 A 用干净的代码块替换 pandoc 的转义文本
    start = next((i for i, ln in enumerate(lines) if ln.startswith("## 附录 A")), None)
    end = next((i for i, ln in enumerate(lines) if ln.startswith("## 附录 B")), None)
    if start is not None and end is not None:
        lines = lines[: start + 1] + ["", APPENDIX_CODE, ""] + lines[end:]

    text = "\n".join(lines).rstrip() + "\n"
    text = re.sub(r"\n{3,}", "\n\n", text)
    MD.write_text(text, encoding="utf-8")
    print(f"saved {MD} ({MD.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
