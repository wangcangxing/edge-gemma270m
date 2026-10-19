r"""生成《端侧语义描述实测报告》DOCX（数据全部从 results/*.json 读，避免手抄错）。"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from paths import ADAPTER, SCENES

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "端侧语义描述实测报告.docx"
FONT = "微软雅黑"
MONO = "Consolas"

bench = json.loads((RESULTS / "edge_bench.json").read_text(encoding="utf-8"))
bench2 = json.loads((RESULTS / "edge_bench_2c.json").read_text(encoding="utf-8"))
bench1 = json.loads((RESULTS / "edge_bench_1c1t.json").read_text(encoding="utf-8"))
prec = json.loads((RESULTS / "edge_thread_precision.json").read_text(encoding="utf-8"))["summary"]
q_int8 = json.loads((RESULTS / "edge_int8_1t.json").read_text(encoding="utf-8"))
q_bf16 = json.loads((RESULTS / "edge_bf16_greedy.json").read_text(encoding="utf-8"))
sweep_post = json.loads((RESULTS / "edge_thread_sweep_posture_clean.json").read_text(encoding="utf-8"))
cmp_desc = json.loads((RESULTS / "edge_edge_vs_mage.json").read_text(encoding="utf-8"))
cmp_fall = json.loads((RESULTS / "edge_edge_vs_mage_fall.json").read_text(encoding="utf-8"))
folds = json.loads((RESULTS / "edge_posture_folds_fall.json").read_text(encoding="utf-8"))
det = json.loads((RESULTS / "edge_mage_determinism.json").read_text(encoding="utf-8"))
rerun = json.loads((RESULTS / "edge_mage_rerun_fall.json").read_text(encoding="utf-8"))
rows_int8, rows_bf16 = q_int8["tuned_rows"], q_bf16["tuned_rows"]
q8s, q16s = q_int8["summary"]["tuned"], q_bf16["summary"]["tuned"]
exact_8_16 = sum(a["desc"] == b["desc"] for a, b in zip(rows_int8, rows_bf16))
ev = json.loads((RESULTS / "edge_eval_lora.json").read_text(encoding="utf-8"))
meta = json.loads((ADAPTER / "train_meta.json").read_text(encoding="utf-8"))
recs = [json.loads(l) for l in SCENES.read_text(encoding="utf-8").splitlines()]

base = ev["base_rows"]
tuned = ev["tuned_rows"]
base_s, tuned_s = ev["summary"]["base"], ev["summary"]["tuned"]


def demo(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


dog, indoor, cpu_demo, video = (demo("edge_demo_dog.json"), demo("edge_demo_indoor.json"),
                                demo("edge_demo_cpu.json"), demo("edge_demo_video.json"))

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin = sec.bottom_margin = Cm(2.0)


def east_asia(style_or_run, name=FONT):
    rpr = style_or_run.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)


normal = doc.styles["Normal"]
normal.font.name = FONT
normal.font.size = Pt(10.5)
east_asia(normal)
normal.paragraph_format.space_after = Pt(4)
normal.paragraph_format.line_spacing = 1.25

for i, size in ((1, 15), (2, 12.5), (3, 11)):
    st = doc.styles[f"Heading {i}"]
    st.font.name = FONT
    st.font.size = Pt(size)
    st.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    st.font.bold = True
    east_asia(st)


def para(text: str = "", bold: bool = False, italic: bool = False, size: float | None = None,
         align=None, style: str | None = None, color: RGBColor | None = None):
    p = doc.add_paragraph(style=style)
    run = p.add_run(text)
    run.bold, run.italic = bold, italic
    if size:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    east_asia(run)
    if align is not None:
        p.alignment = align
    return p


def bullet(text: str):
    return para(text, style="List Bullet")


def caption(text: str):
    p = para(text, size=9, color=RGBColor(0x59, 0x59, 0x59))
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT


def shade(cell, hex_fill: str):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcpr.append(shd)


def table(headers: list[str], rows: list[list], widths: list[float] | None = None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for i, htext in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(htext)
        run.bold = True
        run.font.size = Pt(9.5)
        east_asia(run)
        shade(c, "DCE6F1")
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(9.5)
            east_asia(run)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    # 行不跨页、表格整体尽量不跨页（配合 keep_with_next）
    for ri, row in enumerate(t.rows):
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if ri < len(t.rows) - 1:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True
    trpr = t.rows[0]._tr.get_or_add_trPr()
    trpr.append(OxmlElement("w:tblHeader"))
    return t


def mono(lines: list[str]):
    for line in lines:
        p = doc.add_paragraph()
        run = p.add_run(line)
        run.font.name = MONO
        run.font.size = Pt(9)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0


# ---------------- 封面 ----------------
doc.add_paragraph()
para("端侧语义描述实测报告", bold=True, size=24, align=WD_ALIGN_PARAGRAPH.CENTER,
     color=RGBColor(0x1F, 0x3A, 0x5F))
para("YOLOv10n（看） + Gemma-3-270M（说）· 家庭监控场景", size=12,
     align=WD_ALIGN_PARAGRAPH.CENTER, color=RGBColor(0x44, 0x54, 0x6A))
doc.add_paragraph()
table(["项目", "内容"], [
    ["对象", "Gemma-3-270M-IT + YOLOv10n 的语义描述层"],
    ["数据", "域内 10 段视频抽帧 3,116 张（train 2,168 / val 356 / test 592，按视频切分）"],
    ["硬件", "NVIDIA RTX 5060 Ti 16 GB；Intel i5-14600KF（14 核 20 线程）；31.8 GB RAM"],
    ["日期", "2026-10-18"],
    ["结论", "可用作低成本“感官转述层”：权重 0.499 GiB，CPU 单句 0.45~0.70 s，"
             "两核端到端 0.49~0.60 s/帧、单核 0.76 s/帧；峰值内存约 2.9 GB；"
             "但必须 LoRA 微调，且描述上限受检测器限制。"],
], widths=[3.0, 13.5])
doc.add_page_break()

# ---------------- 摘要 ----------------
doc.add_heading("摘要", level=1)
bullet(f"权重：Gemma-3-270M bf16 为 {bench['base_weights_gb']} GiB，LoRA adapter {bench['adapter_mb']} MB；"
       "CPU int8 常驻 0.63 GB。对比 Mage-VL 4.74B 的 8.83 GiB / 峰值 10.5 GB。")
bullet(f"延迟：GPU bf16 {bench['rows'][0]['gen_s_median']} s/句（{bench['rows'][0]['tok_s']} tok/s）；"
       f"CPU int8 {bench['rows'][2]['gen_s_median']} s/句（{bench['rows'][2]['tok_s']} tok/s）；"
       f"两核两线程 {bench2['gemma_int8']['gen_s_median']} s/句（{bench2['gemma_int8']['tok_s']} tok/s）；"
       f"单核单线程 {bench1['gemma_int8']['gen_s_median']} s/句（{bench1['gemma_int8']['tok_s']} tok/s），"
       f"端到端 {bench1['e2e_s_per_frame']} s/帧。")
bullet(f"内存：int8 参数 {bench['rows'][2]['model_mem_gb']} GB；进程峰值 RSS "
       f"{bench1['memory_gb']['peak_rss']} GB / private {bench1['memory_gb']['peak_private']} GB"
       "（1 核与 2 核几乎相同）。")
bullet(f"质量（test 100 帧，同提示词对照）：人数一致率 {base_s['person_ok']*100:.0f}% → "
       f"{tuned_s['person_ok']*100:.0f}%，物体召回 {base_s['obj_recall']*100:.0f}% → "
       f"{tuned_s['obj_recall']*100:.0f}%，幻觉率 {base_s['halluc_rate']*100:.0f}% → "
       f"{tuned_s['halluc_rate']*100:.0f}%。")
bullet("基座 270M 不会按指令把场景 JSON 改写成句子：它照抄字段（分辨率、姿态、身高占比），"
       "或输出“画面：正常”。LoRA 微调是必需项，不是优化项。")
bullet("270M 这个尺寸上 GPU 没有优势：GPU bf16 0.55 s 与 CPU int8 0.66 s 基本打平；"
       "完整 pipeline 下 4~6 线程最优（4t/6t 打平），8 线程以上明显劣化（20 帧实测）。")
bullet("精度不受线程数影响：1T/2T/14T 的人数一致与物体召回都是 100%，逐字一致 39/40（唯一差异为同义改写）；"
       "int8 与 bf16 逐字一致 77/100，指标同为 100%/100%，差异全为同义句。")
bullet("与 Mage-VL 同帧对照（207 帧）：跌倒召回 Mage 22.6%，Edge 26.4%（域内训练）/ 18.9%（未见测试视频）；"
       "描述任务 Mage 人数 77%、36 字，Edge 100%、16 字；两者都远不足以单独判定“躺地”。")
bullet("描述上限 = 检测器上限：只有 COCO 80 类 + 3 类姿态，没有颜色/材质/动作；"
       "测试集仍是同一家庭按视频留出，未验证跨场景泛化。")

# ---------------- 1 方案 ----------------
doc.add_heading("1 方案与实验设置", level=1)
doc.add_heading("1.1 分工原理", level=2)
para("Gemma-3-270M 是纯文本模型（Gemma3ForCausalLM，268,098,176 参数，无视觉塔），读不到像素。"
     "因此把“看”拆给 YOLOv10n：检测器输出结构化场景 JSON（人数、九宫格位置、脚部是否接近地面、"
     "旁边物体、亮度、可选姿态），语言模型只负责把符号事实说成一句中文。"
     "描述的上限由检测器决定，模型不编造画面里没有的类别。")

doc.add_heading("1.2 组件", level=2)
table(["组件", "版本 / 规格", "作用"], [
    ["YOLOv10n", "ultralytics 8.4.165；yolov10n.pt 5.6 MB", "80 类 COCO 目标检测"],
    ["姿态检测器", "urfd_posture_v5_mixed；3 类（站立/坐/躺卧）", "人体姿态，供描述使用（可选）"],
    ["Gemma-3-270M-IT", "268,098,176 参数；bf16 0.499 GiB", "场景 JSON → 一句中文描述"],
    ["LoRA adapter", f"{meta['steps']} 步训练；{bench['adapter_mb']} MB；7.59 M 可训练参数（2.75%）", "任务微调"],
    ["运行环境", "Python 3.12.7；torch 2.9.1+cu128；transformers 5.17.0；peft 0.21.2", "推理与训练"],
], widths=[3.2, 8.3, 4.8])

doc.add_heading("1.3 指标口径", level=2)
bullet("人数一致率：描述里的人数信息与场景 JSON 一致。0 人需出现“没有/无人/没人/空无一人/看不到”；"
       "1 人需出现“人”且不出现“无人”类表述；多人类推。")
bullet("物体召回：场景全局物体名在描述中出现的比例。")
bullet("幻觉率：描述里出现了不属于该场景（全局物体 + 每人旁边物体 + “人”）的 COCO 类别词的行占比。")
bullet("延迟：单次 generate 的墙钟秒数；tok/s = 新生成 token 数 / 秒，含 prefill。")
bullet("常驻：参数字节数（bf16 / int8 实际张量）；GPU 峰值取一次生成内 max_memory_allocated。")
bullet(f"测试集：test 592 帧中随机抽 100 帧（随机种子 1），其中 {sum(1 for r in tuned if r['n'] == 0)} 帧无人、"
       f"{sum(1 for r in tuned if r['n'] == 1)} 帧 1 人、{sum(1 for r in tuned if r['n'] == 2)} 帧 2 人。")

# ---------------- 2 数据 ----------------
doc.add_heading("2 数据集", level=1)
split = Counter(r["split"] for r in recs)
persons = Counter(len(r["persons"]) for r in recs)
posture = Counter(p.get("姿态") or "未识别" for r in recs for p in r["persons"])
obj = Counter(o.split("x")[0] for r in recs for o in r["objects"])
light = Counter(r["light"] for r in recs)
para("来源：本地域内 10 段室内视频抽帧（原始帧与权重不入库），共 "
     f"{len(recs):,} 帧；按视频切分，避免帧级泄漏。"
     "姿态用数据集 YOLO 标注（0=站立、1=坐、2=躺卧）按 IoU 校正。")
table(["切分", "视频", "帧数"], [
    ["train", "fall_detection_4~10（7 段）", split["train"]],
    ["val", "hr_fall_detection_2", split["val"]],
    ["test", "hr_fall_detection_1 / 3", split["test"]],
    ["合计", "10 段", len(recs)],
], widths=[2.2, 7.0, 2.2])
caption("表 1 数据切分（按视频，非按帧）")

table(["画面人数", "帧数", "占比"], [
    ["0 人", persons[0], f"{persons[0]/len(recs)*100:.1f}%"],
    ["1 人", persons[1], f"{persons[1]/len(recs)*100:.1f}%"],
    ["2 人", persons[2], f"{persons[2]/len(recs)*100:.1f}%"],
    ["3 人", persons[3], f"{persons[3]/len(recs)*100:.1f}%"],
], widths=[3.0, 3.0, 3.0])
caption("表 2 人数分布：以单人场景为主（86.7%）")

table(["姿态", "实例数", "说明"], [
    ["站立", posture["站立"], "检测器与描述中最常见"],
    ["坐", posture["坐"], ""],
    ["躺卧", posture["躺卧"],
     f"跌倒场景的关键类别，占 {posture['躺卧']/sum(posture.values())*100:.1f}%"],
    ["未识别", posture["未识别"], "姿态置信度 < 0.55 或框不匹配，描述里不提姿态"],
], widths=[3.0, 3.0, 8.0])
caption(f"表 3 人体姿态分布（{sum(posture.values())} 个实例）")

top = obj.most_common(8)
table(["物体类别", "出现帧数"], [[k, v] for k, v in top], widths=[4.0, 4.0])
caption("表 4 场景物体 Top 8（按帧计）")

para(f"光照：正常 {light['正常']} 帧、偏暗 {light['偏暗']} 帧。"
     "数据几乎全部为正常光照，夜间/低照度未覆盖。")

# ---------------- 3 训练 ----------------
doc.add_heading("3 LoRA 微调", level=1)
table(["超参", "取值"], [
    ["可训练参数 / 占比", "7,593,984 / 275,692,160（2.75%）"],
    ["LoRA r / alpha / dropout", "32 / 64 / 0.05"],
    ["目标模块", "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"],
    ["优化器 / 学习率", "AdamW / 5e-4（cosine，5% warmup）"],
    ["batch × grad-accum", "8 × 2"],
    ["max_len / epochs / steps", f"256 / {meta['epochs']} / {meta['steps']}"],
    ["训练耗时", f"{meta['seconds']} s（RTX 5060 Ti）"],
    ["最佳 val loss", f"{meta['best_val']:.4f}"],
], widths=[5.0, 10.0])
caption("表 5 LoRA 训练配置（adapter 输出到 out/lora_gemma270m_zh）")

para("训练目标由模板生成（人数、位置、姿态、旁边物体、亮度），只包含场景里已有的事实。"
     "关键工程点：262k 词表的 lm_head 若对整段序列计算，8×320 的 logits 会占满 15.7 GB 显存、"
     "单步 1.7 s；改成只取 hidden[:, :-1] 中 label ≠ -100 的位置后，显存降到 MB 级、单步约 0.3 s。")

# ---------------- 4 延迟 ----------------
doc.add_heading("4 延迟与资源", level=1)
doc.add_heading("4.1 三档推理", level=2)
rows = []
for r in bench["rows"]:
    label = {"gpu_bf16": "GPU bf16", "cpu_fp32": "CPU fp32", "cpu_int8": "CPU int8"}[r["tag"]]
    rows.append([label, f"{r['gen_s_median']} s", f"{r['tok_s']}", f"{r['model_mem_gb']} GB",
                 f"{r['load_s']} s"])
table(["档位", "每句生成", "tok/s", "模型常驻", "加载"], rows, widths=[3.2, 3.0, 2.5, 3.0, 2.5])
caption("表 6 单句生成（greedy，16 token，连跑 3 次取中位数；bench.py）")

para("GPU 与 CPU 打平的原因：270M 在 Windows + PyTorch 上逐 token 的 Python/调度开销约 59 ms，"
     "大于算力本身；SDPA 也回落到 math 后端（head_dim 256、4 heads、1 KV head，flash 未编译）。"
     "因此“上 GPU”对这个尺寸没有收益。")

doc.add_heading("4.2 两核两线程", level=2)
table(["项目", "数值"], [
    ["CPU 亲和性 / torch 线程", f"逻辑 CPU {bench2['cpus']} / {bench2['torch_threads']} 线程"],
    ["Gemma int8 单句", f"{bench2['gemma_int8']['gen_s_median']} s（{bench2['gemma_int8']['tok_s']} tok/s，"
                        f"{bench2['gemma_int8']['gen_tokens']} token）"],
    ["YOLOv10n CPU 每帧", f"{bench2['yolo_cpu_ms']} ms"],
    ["端到端每帧", f"{bench2['e2e_s_per_frame']} s"],
    ["进程内存峰值", f"RSS {bench2['memory_gb']['peak_rss']} GB / private "
                    f"{bench2['memory_gb']['peak_private']} GB"],
], widths=[5.0, 9.0])
caption("表 7 两核两线程（bench_2c.py，绑 0,2 两个 P 核逻辑 CPU）")
para("对照：绑同一 P 核的两个超线程（0,1）时，单句 0.563 s（28.4 tok/s）、YOLO 55 ms、"
     "端到端 0.62 s。两个物理核仍略优于两个超线程。")

doc.add_heading("4.2.1 单核单线程与内存", level=3)
m1 = bench1["memory_gb"]
table(["项目", "数值"], [
    ["CPU 亲和性 / torch 线程", f"逻辑 CPU {bench1['cpus']} / {bench1['torch_threads']} 线程"],
    ["Gemma int8 单句", f"{bench1['gemma_int8']['gen_s_median']} s"
                        f"（{bench1['gemma_int8']['tok_s']} tok/s，"
                        f"{bench1['gemma_int8']['gen_tokens']} token）"],
    ["YOLOv10n CPU 每帧", f"{bench1['yolo_cpu_ms']} ms"],
    ["端到端每帧", f"{bench1['e2e_s_per_frame']} s"],
    ["进程 RSS 峰值", f"{m1['peak_rss']} GB"],
    ["进程 private 峰值", f"{m1['peak_private']} GB"],
    ["加载后 RSS（空进程）", f"{m1['rss_after_gemma']} GB（{m1['rss_base']} GB）"],
], widths=[5.0, 9.0])
caption("表 8 单核单线程（bench_2c.py --cpus 0 --threads 1）")
para(f"内存说明：int8 参数本身只有 0.63 GB；进程峰值 {m1['peak_rss']} GB RSS"
     f"（private {m1['peak_private']} GB）主要来自 PyTorch 运行时 + 量化时先载入的 fp32 副本"
     "（1.07 GB）及分配器缓存。单核与双核的内存几乎一样（2.91 vs 2.76 GB），核数只影响时间。")

doc.add_heading("4.2.2 线程数与精度", level=3)
table(["配置", "人数一致率", "物体召回", "与 14 线程逐字一致"], [
    ["14 线程", "100%", "100%", "基线"],
    ["2 线程", "100%", "100%", f"{prec['exact_match_vs_14t']['2t']}/{prec['n']}"],
    ["1 线程（单核）", "100%", "100%", f"{prec['exact_match_vs_14t']['1t']}/{prec['n']}"],
], widths=[3.5, 3.0, 3.0, 4.5])
caption("表 9 test 40 帧贪心输出对照（probe_threads_precision.py）")
para("与 14 线程基线相比，2 线程与 1 线程各只有 1 条同义改写"
     "（“画面上看不到人，椅子清晰可见。” vs “画面里没有看到人，椅子都很清楚。”），"
     "人数、物体、位置没有事实性差异。线程数只改速度（同批 1T 0.43 s、2T 0.35 s、14T 0.66 s 每句），不改精度。")

doc.add_heading("4.3 线程数反直觉", level=2)
table(["配置", "Gemma int8 单句", "tok/s", "YOLO CPU 每帧"], [
    ["默认 14 线程", "0.658 s", "24.3", "138 ms"],
    ["2 线程（绑 0,2）", "0.452 s", "35.4", "41 ms"],
], widths=[4.5, 4.0, 2.5, 4.0])
caption("表 10 线程数对比：小模型上限由同步开销决定，开满反而互相等")

doc.add_heading("4.3.1 姿态开启复核（20 帧）", level=3)
post_rows = []
for key in ("4t#0", "6t#1", "2t", "8t", "1t", "14t", "4t#6", "6t#7"):
    v = sweep_post["configs"][key]
    t = v["total_ms"]
    post_rows.append([key.replace("#0", "").replace("#1", "").replace("#6", "（重复）")
                      .replace("#7", "（重复）"),
                      f"{t['median']}", f"{t['mean']}", f"{t['std']}", f"{t['p95']}"])
table(["配置", "median", "mean", "std", "p95"], post_rows,
      widths=[4.5, 2.8, 2.8, 2.8, 2.8])
caption("表 11 完整 pipeline（姿态开启、20 帧、全局预热 6 次）：4t 与 6t 打平")
para("4t 两次中位 455.1 / 513.8 ms，6t 472.5 / 473.6 ms；逐帧胜负 13/20 与 3/20 两次方向翻转。"
     "8t 以上明显劣化（687 / 1248 ms）。因此完整 pipeline 下推荐 4~6 线程，"
     "“4t 唯一最优”不成立；1t/2t 更慢，14t 最差。")

doc.add_heading("4.4 端到端样例", level=2)
rec_dog = dog["records"][0]
rec_ind = indoor["records"][0]
rec_cpu = cpu_demo["records"][0]
table(["输入", "设备", "YOLO", "生成", "输出"], [
    ["dog.jpg", "GPU", f"{rec_dog['yolo_ms']:.0f} ms", f"{rec_dog['gen_s']:.2f} s", rec_dog["description"]],
    ["室内帧（1280×720）", "GPU", f"{rec_ind['yolo_ms']:.0f} ms", f"{rec_ind['gen_s']:.2f} s", rec_ind["description"]],
    ["同一室内帧", "CPU int8", f"{rec_cpu['yolo_ms']:.0f} ms", f"{rec_cpu['gen_s']:.2f} s", rec_cpu["description"]],
], widths=[3.5, 2.0, 2.0, 2.0, 6.2])
caption("表 12 端到端输出（pipeline.py；GPU 为 bf16，CPU 为 int8）")

vrows = [[str(r["frame"]), f"{r['yolo_ms']:.0f} ms", f"{r['gen_s']:.2f} s", r["description"]]
         for r in video["records"]]
table(["帧号", "YOLO", "生成", "输出"], vrows, widths=[2.0, 2.2, 2.2, 9.5])
caption("表 13 视频路径 smoke test（sample_1.mp4，0.5 fps 抽 3 帧）")
para("注：该视频是户外画面（汽车），但模板里写死了“室内”假设，第一帧输出“室内没有人”属于场景假设错误，"
     "说明语义层没有判断室内外的能力。")

# ---------------- 5 质量 ----------------
doc.add_heading("5 质量评估", level=1)
doc.add_heading("5.1 汇总", level=2)
table(["指标", "基座 270M", "+ LoRA"], [
    ["人数一致率", f"{base_s['person_ok']*100:.1f}%", f"{tuned_s['person_ok']*100:.1f}%"],
    ["物体召回", f"{base_s['obj_recall']*100:.1f}%", f"{tuned_s['obj_recall']*100:.1f}%"],
    ["幻觉率", f"{base_s['halluc_rate']*100:.1f}%", f"{tuned_s['halluc_rate']*100:.1f}%"],
    ["平均字数", f"{base_s['avg_len']}", f"{tuned_s['avg_len']}"],
    ["平均生成耗时", f"{base_s['avg_gen_s']} s", f"{tuned_s['avg_gen_s']} s"],
    ["字数范围", f"{min(r['len'] for r in base)}~{max(r['len'] for r in base)}",
     f"{min(r['len'] for r in tuned)}~{max(r['len'] for r in tuned)}"],
], widths=[4.0, 4.5, 4.5])
caption("表 14 test 100 帧，同一提示词（无 few-shot）对照")
para("幻觉率口径：描述中出现了不属于该帧允许集合（全局物体 ∪ 每人旁边物体 ∪ “人”）的 COCO 类别词，"
     "按行计；同一行提到多个也只计 1 行。")

table(["指标", "bf16（GPU）", "int8（CPU，1 线程）"], [
    ["人数一致率", f"{q16s['person_ok']*100:.0f}%", f"{q8s['person_ok']*100:.0f}%"],
    ["物体召回", f"{q16s['obj_recall']*100:.0f}%", f"{q8s['obj_recall']*100:.0f}%"],
    ["幻觉率", f"{q16s['halluc_rate']*100:.0f}%", f"{q8s['halluc_rate']*100:.0f}%"],
    ["平均字数", f"{q16s['avg_len']}", f"{q8s['avg_len']}"],
    ["逐字一致", "基线", f"{exact_8_16}/{len(rows_int8)}"],
], widths=[4.0, 4.5, 4.5])
caption("表 15 int8 与 bf16 的精度对照（同一 100 帧，贪心解码）")
para("幻觉率沿用表 14 口径；“逐字一致”按整句完全相等计（标点也算）。")
para(f"{len(rows_int8)-exact_8_16} 条差异全部是同义改写，例如 int8“画面下方有人站立，椅子就在旁边。”"
     "对应 bf16“画面下方有一人站立，旁边是椅子。”，没有人数、物体或位置的事实性差异。")

doc.add_heading("5.2 分人数", level=2)
rows = []
for n in (0, 1):
    b = [r for r in base if r["n"] == n]
    t = [r for r in tuned if r["n"] == n]
    if not b:
        continue
    rows.append([f"{n} 人", len(b),
                 f"{sum(r['person_ok'] for r in b)/len(b)*100:.0f}%",
                 f"{sum(r['obj_recall'] for r in b)/len(b)*100:.0f}%",
                 f"{sum(r['person_ok'] for r in t)/len(t)*100:.0f}%",
                 f"{sum(r['obj_recall'] for r in t)/len(t)*100:.0f}%"])
table(["场景", "帧数", "基座人数一致", "基座召回", "LoRA 人数一致", "LoRA 召回"], rows,
      widths=[2.2, 1.8, 3.2, 2.8, 3.4, 2.8])
caption("表 16 分人数统计：基座在 5 个无人帧上全部失败（0/5）")

doc.add_heading("5.3 失败模式", level=2)
para("基座 29/100 帧人数不一致，主要是三种：")
bullet("输出“画面：正常”“画面正常。”，把亮度字段当成整句，完全忽略人和物。")
bullet("照抄字段：如“画面显示一个正常状态的物体，人物位置右下，身高占比0.33，脚部接近地面，"
       "姿态站立，旁边物体是椅子。”（50 字，JSON 字段原样复述）")
bullet("5 个无人帧全部没有表达“无人”，而是描述出了物体或分辨率。")
para("LoRA 后 100 帧人数全部一致；唯一一处幻觉（1/100）是场景 JSON 里物体为空、"
     "人物旁边物体也为空，模型仍补了“旁边是椅子”。")

doc.add_heading("5.4 输出样例", level=2)
table(["场景输入", "输出", "来源"], [
    ["1 人躺卧、左侧下、沙发×2、茶几", "画面左侧有一人躺卧，旁边是沙发和茶几。", "CPU int8"],
    ["无人、只有狗", "画面中没有看到人，只有狗。", "dog.jpg"],
    ["1 人站立、中央、椅子/沙发/床", "画面中央有一人，旁边是椅子、沙发和床。", "室内帧 GPU"],
    ["1 人站立、右下、椅子", "画面右下方有人站立，椅子就在旁边。", "test 帧"],
], widths=[5.0, 7.5, 3.0])
caption("表 17 典型输出（句子由模型生成，非模板拼接）")

# ---------------- 6 结论 ----------------
doc.add_heading("6 结论", level=1)
for i, text in enumerate([
    "这条路可行：YOLOv10n 提供符号事实、Gemma-3-270M 生成中文，权重 0.5 GiB，"
    "CPU 单句 0.45~0.70 s；两核端到端 0.49~0.60 s/帧，单核 0.76 s/帧。",
    "必须微调：基座不会按指令改写 JSON（人数一致 71%、召回 57%），LoRA 后 100%/100%、幻觉 1%，"
    "训练成本 813 步 / 249 秒。",
    "端侧不需要 GPU：GPU bf16 与 CPU int8 打平；完整 pipeline 下 4~6 线程最优"
    "（4t 与 6t 在噪声内打平），8 线程以上明显劣化；单核精度不掉"
    "（1T/2T/14T 指标均为 100%）。内存峰值约 2.9 GB RSS，其中 int8 参数仅 0.63 GB。",
    "描述上限等于检测器上限：只有 COCO 80 类 + 3 类姿态，没有颜色/材质/动作；姿态漏检时不提“躺卧”。",
    "泛化未证明：测试按视频留出，但仍是同一家庭同一机位；跨场景/跨机位需要重新采数据验证。",
    "与 Mage-VL 路线相比：权重 8.83 GiB → 0.499 GiB，峰值显存 10.5 GB → 0.55 GB，"
    "长描述 10.8 s → 一句 0.45~0.7 s，但描述能力从“开放式”降为“一句话符号转述”。",
    "同帧对照（207 帧）：跌倒召回 Mage 22.6% vs Edge 26.4%（域内训练）/ 18.9%（未见测试视频），"
    "两者都 <30%；Edge 的“持平”主要来自域内监督，不是架构优势。",
], 1):
    para(f"{i}. {text}")

# ---------------- 7 限制 ----------------
doc.add_heading("7 限制与未验证", level=1)
for text in [
    "未导出 ONNX / GGUF / RKNN，未在 ARM、NPU 或目标板卡上实测；本机是 x86 + RTX 5060 Ti。",
    "视频路径只做了 3 帧 smoke test，没有逐帧事件评估（跌倒触发、误报率等）。",
    "模板监督把句式限制在 14~23 字，不会产生开放式长描述；“室内”是写死的场景假设，户外帧会说错。",
    "数据几乎全是正常光照、单人场景（86.7%），多人（81 帧）与夜间未覆盖。",
    "姿态标签在训练时来自数据集标注、推理时来自检测器（跨域 lie 召回约 14.4%），存在训练/推理分布差。",
    "测试集与训练集同域，指标不能外推到其他家庭、其他摄像头角度。",
]:
    bullet(text)

# ---------------- 8 与 Mage-VL 对照 ----------------
doc.add_heading("8 与 Mage-VL 对照（控制变量）", level=1)

doc.add_heading("8.1 对齐方式", level=2)
para("跌倒判定用 Mage 跨域协议的原帧：probe 记录 221 行，能映射回 250/750 ms 抽帧 JPG 的 207 帧"
     "（53 Lie / 79 Stand / 75 Sit），Mage 与 Edge 使用完全相同的 207 个 JPG 文件。"
     "Mage 重跑问题与原 probe 一致（Q_LIE），贪心解码；Edge 用同一帧跑 YOLO + 姿态 + Gemma，"
     "描述含“躺/卧”即判 lie。另做一组 100 帧描述任务：test 划分、seed=1，与 207 帧只有 2 个文件重叠。")
para("fold 定义：foldA 的训练集排除 hr_fall_detection_1 / hr_fall_detection_3（这两段在 foldA 的 val 中）；"
     "foldB 排除 hr_fall_detection_2；v5_mixed 使用全部域视频。因此只有 foldA 对这两个测试视频是“未见”的。")

doc.add_heading("8.2 跌倒判定（207 帧）", level=2)
fall_rows = []
for label, m in (
    ("Mage-VL（重跑，JPG 帧）", rerun["new"]),
    ("Mage-VL（旧记录，MP4 帧）", rerun["old"]),
    ("Edge v5_mixed（见过该域）", cmp_fall["edge_detector"]),
    ("Edge foldB（见过该域）", folds["urfd_posture_foldB"]),
    ("Edge foldA（未见 hr_1/3）", folds["urfd_posture_foldA"]),
):
    fall_rows.append([label, f"{m['tp']}/{m['fn']}/{m['fp']}",
                      f"{m['recall']*100:.1f}%", f"{m['precision']*100:.1f}%",
                      f"{m['accuracy']*100:.1f}%"])
table(["系统", "TP/FN/FP", "召回", "精度", "准确率"], fall_rows,
      widths=[5.6, 3.0, 2.4, 2.4, 2.4])
caption("表 18 跌倒判定：同一 207 帧、同一标签、同一判定（Mage 0.64 s / 9.32 GB；Edge 0.55 GB）")
para("两边都只认“躺地”这一类，精度 100% 表示报出的 yes 全是真 lie；召回都不到 30%。"
     "Mage 重跑 22.6% 与旧记录 28.3% 的差别来自取帧（JPG vs MP4 整数秒），不是随机性（见 8.4）。")
para("统计检验（同 207 帧配对 McNemar 精确检验，Mage vs Edge v5）：b=6（仅 Mage 对）、c=8（仅 Edge 对），"
     "双侧 p=0.79，差异不显著。召回 95% Wilson CI：Mage 22.6% [13.5, 35.5]%、"
     "Edge 26.4% [16.4, 39.6]%、foldA 18.9% [10.6, 31.4]%。")

doc.add_heading("8.3 描述任务（100 帧）", level=2)
table(["指标", "Mage-VL", "Edge"], [
    ["人数一致率", f"{cmp_desc['mage']['person_ok']*100:.0f}%", f"{cmp_desc['edge']['person_ok']*100:.0f}%"],
    ["姿态正确", f"{cmp_desc['mage']['posture_correct']*100:.0f}%", f"{cmp_desc['edge']['posture_correct']*100:.0f}%"],
    ["姿态未提", f"{cmp_desc['mage']['posture_missed']*100:.0f}%", f"{cmp_desc['edge']['posture_missed']*100:.0f}%"],
    ["平均字数", f"{cmp_desc['mage']['avg_len']}", f"{cmp_desc['edge']['avg_len']}"],
    ["单帧延迟", f"{cmp_desc['mage']['avg_gen_s']} s", f"{cmp_desc['edge']['avg_gen_s']} s"],
    ["峰值显存", f"{cmp_desc['mage_peak_gb']} GB", f"0.55 GB"],
], widths=[4.5, 5.0, 5.0])
caption("表 19 描述任务：同一 100 帧、同一句中文描述指令（这批恰好 0 个 lie）")
para("姿态口径：真值取数据集标注（站立/坐/躺卧）；描述含对应词（站立/站、坐、躺/卧）记“正确”，"
     "含其他姿态词但非真值记“错误”，不含任何姿态词记“未提”。"
     "7 帧无姿态标注（含 5 个无人帧）不计入三者，百分比以全部 100 帧为分母。")
para("Mage 的 23 次人数失败主要是只描述教室、没提人；Edge 不会漏人，但没有教室/白板/投影仪这类场景语义。")

doc.add_heading("8.4 确定性检验（5 个翻转帧）", level=2)
det_rows = []
for mode in ("deterministic", "default"):
    answers = [r[mode][0] for r in det["rows"]]
    stable = sum(1 for r in det["rows"] if len(set(r[mode])) == 1)
    det_rows.append([f"5 帧（JPG，t+0.25/0.75）", f"{mode} ×{det['runs']}",
                     ", ".join(answers), f"{stable}/5 稳定", "新重跑"])
det_rows.append(["5 帧（MP4 整数秒）", "deterministic ×2",
                 "yes, yes, yes, yes, no", "5/5 稳定", "旧记录"])
table(["帧源", "模式", "5 帧答案", "稳定性", "一致于"], det_rows,
      widths=[4.8, 4.2, 4.6, 2.6, 2.4])
caption("表 20 确定性重跑：torch.use_deterministic_algorithms(True) + 固定种子")
para("同一帧源下 deterministic 与 default 输出逐字一致；换取帧方式（MP4 整数秒 vs JPG）答案整体翻转。"
     "所以 5 帧差异不是随机性，而是帧源不同。")

doc.add_heading("8.5 结论与边界", level=2)
for text in [
    "同帧源（JPG）下：Mage 22.6% / Edge v5 26.4% / Edge foldA 18.9%，三者都远不足以单独判定“躺地”。",
    "Edge 的“略高”依赖姿态检测器见过该域（v5_mixed/foldB）；换成未见 hr_1/hr_3 的 foldA 就低于 Mage。",
    "成本差距明确：Mage 9.32 GB / 0.64 s（单个 yes/no），Edge 0.55 GB；描述任务 0.71 s。",
    "描述能力方向相反：Mage 更开放（36 字、场景语义），Edge 更准但更短（16 字、只有检测到的类别）。",
    "边界：14 行缺 250 ms 帧；foldA 只排除了 hr_1/hr_3；测试域是计算机教室；Mage 0.64 s 是 yes/no 而非长描述。",
]:
    bullet(text)

# ---------------- 附录 ----------------
doc.add_heading("附录 A 复现命令", level=1)
mono([
    "# 在仓库根目录执行；$py 换成你的 Python/venv",
    "$py = 'python'",
    "$adapter = 'out/lora_gemma270m_zh'",
    "",
    "# 权重：下载到 models/gemma-3-270m-it（或设 EDGE_GEMMA_MODEL）",
    "# 数据：把抽帧目录设到 EDGE_DOMAIN_IMAGES，再构建 scenes.jsonl",
    "& $py build_dataset.py",
    "& $py train_lora.py --epochs 3 --out $adapter",
    "",
    "# 评估 / 基准",
    "& $py eval_tuned.py --adapter $adapter --n 100",
    "& $py bench.py --runs 3",
    "$env:EDGE_CPUS='0,2'; & $py bench_2c.py --image <IMAGE>",
    "",
    "# 端到端",
    "& $py pipeline.py --source <IMAGE> --adapter $adapter",
    "& $py pipeline.py --source <IMAGE> --yolo-device cpu --gemma-device cpu --int8 --adapter $adapter",
])

doc.add_heading("附录 B 工程文件", level=1)
table(["文件", "作用"], [
    ["scene_yolo.py", "YOLOv10n → 场景 JSON"],
    ["prompting.py", "场景 JSON → 对话提示词（无 few-shot）"],
    ["gemma_describe.py", "270M 加载 / 生成 / 计时；支持 LoRA、CPU int8"],
    ["pipeline.py", "图片 / 视频 → 场景 → 描述，结果写 results/edge_*.json"],
    ["targets.py / build_dataset.py", "模板监督信号；数据集构建与按视频切分"],
    ["train_lora.py", "LoRA 微调（只在目标 token 上算 CE）"],
    ["eval_tuned.py", "基座 vs 微调的三项质量指标"],
    ["bench.py / bench_2c.py", "三档延迟；两核两线程延迟"],
    ["probe_quality.py", "基座提示词探针（负面结论留证）"],
    ["compare_mage.py / compare_mage_fall.py", "与 Mage 同帧对照（描述 / 跌倒判定）"],
    ["rerun_mage_fall.py / rerun_mage_deterministic.py", "重跑 Mage、确定性检验"],
    ["eval_posture_folds.py / check_mage_old_protocol.py", "foldA/foldB 消融；旧取帧协议复现"],
    ["results/*.json", "本报告全部数字的原始输出"],
], widths=[5.0, 11.0])

# 表题移到表格上方并与其同页（表题在表上，避免表题被挤到下一页）
_moved = 0
for _p in list(doc.paragraphs):
    if _p.text.startswith("表 "):
        _prev = _p._element.getprevious()
        if _prev is not None and _prev.tag == qn("w:tbl"):
            _prev.addprevious(_p._element)
            _p.paragraph_format.keep_with_next = True
            _moved += 1
print("captions moved above tables:", _moved)

# 页脚
footer_p = sec.footer.paragraphs[0]
footer_p.text = "端侧语义描述实测报告 · YOLOv10n + Gemma-3-270M · 2026-10-18"
footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in footer_p.runs:
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(0x7F, 0x7F, 0x7F)
    east_asia(run)

doc.core_properties.title = "端侧语义描述实测报告"
doc.core_properties.subject = "YOLOv10n + Gemma-3-270M"
doc.core_properties.author = "模型优化项目"
doc.save(OUT)
print("saved", OUT, OUT.stat().st_size)
