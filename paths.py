"""集中管理可配置路径：仓库内全部用相对路径，外部资源用环境变量覆盖。

默认值都相对本目录，不写死任何机器上的绝对路径：
    EDGE_GEMMA_MODEL      Gemma-3-270M 权重目录（默认 models/gemma-3-270m-it）
    EDGE_ADAPTER          LoRA adapter 目录（默认 out/lora_gemma270m_zh）
    EDGE_DOMAIN_IMAGES    域内抽帧图片目录（默认 dataset/images）
    EDGE_POSTURE_WEIGHTS  姿态检测器权重（默认 weights/posture_best.pt）
    EDGE_SAMPLE_IMAGE     bench_2c.py 用的测试图（默认 samples/sample.jpg）
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


GEMMA_MODEL = env_path("EDGE_GEMMA_MODEL", HERE / "models" / "gemma-3-270m-it")
ADAPTER = env_path("EDGE_ADAPTER", HERE / "out" / "lora_gemma270m_zh")
DOMAIN_IMAGES = env_path("EDGE_DOMAIN_IMAGES", HERE / "dataset" / "images")
POSTURE_WEIGHTS = env_path("EDGE_POSTURE_WEIGHTS", HERE / "weights" / "posture_best.pt")
SAMPLE_IMAGE = env_path("EDGE_SAMPLE_IMAGE", HERE / "samples" / "sample.jpg")
SCENES = HERE / "dataset" / "scenes.jsonl"
