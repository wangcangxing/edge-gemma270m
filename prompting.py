"""场景 JSON → Gemma 对话提示词。

没有 few-shot：实测 few-shot 里的"沙发和茶几"会被 270M 直接抄进无关场景。
指令保持极简，让 LoRA 学"数据 → 句子"的映射，而不是学"抄示例"。
"""
from __future__ import annotations

import json
from typing import Any

ZH_SYSTEM = (
    "你是家庭监控的语义描述引擎。输入是检测器给出的结构化场景数据，不是图片。"
    "根据数据写一句客观的中文描述，20~40字，只输出这一句话，"
    "不解释、不复述输入、不编造数据里没有的人和物体。"
)

EN_SYSTEM = (
    "You are a semantic description engine for home surveillance. "
    "The input is structured scene data from a detector, not an image. "
    "Write one objective English sentence (15-30 words). Output only the sentence."
)


def scene_json(scene_data: dict[str, Any]) -> str:
    return json.dumps(scene_data, ensure_ascii=False, separators=(",", ":"))


def build_messages(scene_data: dict[str, Any], lang: str = "zh") -> list[dict[str, str]]:
    system = ZH_SYSTEM if lang == "zh" else EN_SYSTEM
    body = scene_json(scene_data)
    if lang == "zh":
        user = f"数据：{body}\n描述："
    else:
        user = f"Data: {body}\nDescription:"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def clean_output(text: str) -> str:
    """去掉特殊 token、前缀和多余句子，只留第一句。"""
    for token in ("<end_of_turn>", "<eos>", "<bos>", "<start_of_turn>model"):
        text = text.replace(token, "")
    text = text.strip().strip('"').strip()
    for prefix in ("输出：", "描述：", "Output:", "Description:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.splitlines()[0].strip() if text else ""
    for i, ch in enumerate(text):
        if ch in "。！？!?":
            text = text[: i + 1]
            break
    return text
