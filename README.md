# 端侧语义描述层：YOLOv10n 看 + Gemma-3-270M 说

本目录是「另起一个思路」的独立工程：用**目标检测器提供视觉事实**、**270M 纯文本模型生成中文描述**，
替代 4.74B 的 Mage-VL 在端侧承担语义描述。权重约 **0.5 GiB**（对比 Mage-VL bf16 的 8.83 GiB）。
完整实测数据见 **[端侧语义描述实测报告.md](端侧语义描述实测报告.md)**。

## 为什么这样分工

- `google/gemma-3-270m-it` 是**纯文本**模型（`Gemma3ForCausalLM`，268,098,176 参数，无视觉塔）。
  它读不到像素，只能读结构化场景 JSON。
- YOLOv10n 只认 80 类 COCO 目标 + 可选的 3 类姿态模型，输出「人有几个、在哪、旁边有什么」。
- 因此语义层的输入是符号事实，输出是一句中文；**描述的上限 = 检测器的上限**，不编造。

## 文件

| 文件 | 作用 |
| --- | --- |
| `scene_yolo.py` | YOLOv10n 检测 → 场景 JSON（九宫格位置、脚部接地、旁边物体、亮度、可选姿态） |
| `prompting.py` | 场景 JSON → 对话提示词（无 few-shot，避免 270M 抄示例） |
| `gemma_describe.py` | 加载 Gemma-3-270M（可挂 LoRA），生成一句描述并计时；`--int8` 走 CPU 动态量化 |
| `pipeline.py` | 端到端：图片/视频 → 场景 → 描述 → `results/edge_<tag>.json` |
| `targets.py` | 模板监督信号：场景 → 目标句（只含场景里已有的事实） |
| `build_dataset.py` | 域内视频帧 → `dataset/scenes.jsonl`（按视频切 train/val/test，用标注补姿态） |
| `train_lora.py` | LoRA 微调（只在目标 token 上算 lm_head，显存 MB 级） |
| `eval_tuned.py` | 基座 vs 微调：人数一致 / 物体召回 / 幻觉率 / 耗时 |
| `bench.py` | 复现 GPU bf16 / CPU fp32 / CPU int8 三档延迟与体积 |
| `bench_2c.py` | 绑 2 核 2 线程的端侧档位（亲和性 + torch 线程数） |
| `build_report.py` | 生成 `端侧语义描述实测报告.docx`（本地，不入库） |
| `build_report_md.py` | 用 pandoc 把 Word 报告转成 `端侧语义描述实测报告.md`（入库副本） |
| `端侧语义描述实测报告.md` | **实测报告正文**（GitHub 上直接读） |
| `paths.py` | 路径统一入口：仓库内相对路径 + 环境变量覆盖 |
| `desensitize_results.py` | 把 `results/*.json` 里的本机绝对路径替换成占位符 |
| `probe_quality.py` | 基座提示词探针（记录基座不会改写 JSON 的负面结论） |

## 快速开始

```powershell
# 依赖：pip install -r requirements.txt
# 权重：下载到 models/gemma-3-270m-it（或设 EDGE_GEMMA_MODEL）
python -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/gemma-3-270m-it', local_dir='models/gemma-3-270m-it')"
# adapter：训练输出到 out/lora_gemma270m_zh（或设 EDGE_ADAPTER）
# 数据：把抽帧目录设到 EDGE_DOMAIN_IMAGES，再生成 dataset/scenes.jsonl

$py = 'python'
$adapter = 'out/lora_gemma270m_zh'

# 1) 单图端到端（微调后）
& $py pipeline.py --source <IMAGE> --adapter $adapter

# 2) 纯 CPU 端侧档位（int8）
& $py pipeline.py --source <IMAGE> --yolo-device cpu --gemma-device cpu --int8 --adapter $adapter

# 3) 只看检测器的场景 JSON
& $py scene_yolo.py --image <IMAGE>

# 4) 复现训练
& $py build_dataset.py
& $py train_lora.py --epochs 3 --out $adapter

# 5) 评估 / 基准
& $py eval_tuned.py --adapter $adapter --n 100
& $py bench.py --runs 3
$env:EDGE_CPUS = '0,2'; & $py bench_2c.py --image <IMAGE>

# 6) 重新生成报告（docx 本地；md 入库）
& $py build_report.py
& $py build_report_md.py      # 需要 pandoc
```

## 实测（2026-10-18，本机 RTX 5060 Ti + i5-14600KF）

### 体积

| 组件 | 体积 |
| --- | --- |
| YOLOv10n（`yolov10n.pt`） | 5.6 MB |
| Gemma-3-270M bf16 | **0.499 GiB** |
| LoRA adapter | **29 MB** |
| CPU int8 常驻（含 fp32 词嵌入） | 0.63 GB |

### 延迟（单句，greedy，输出 15~16 token；`bench.py` 连跑 3 次取中位数）

| 档位 | 每句生成 | tok/s | 常驻 |
| --- | --- | --- | --- |
| GPU bf16 | 0.55 s | 27.3 | 0.50 GB |
| CPU fp32 | 1.01 s | 14.9 | 1.00 GB |
| CPU int8 | 0.66 s | 24.3 | 0.63 GB |
| YOLO（GPU / CPU） | 20~50 / 138 ms 每帧 | — | <1 GB |
| **端到端（CPU int8 + CPU YOLO）** | **≈0.8~1.2 s/帧** | — | — |

**关键观察**：270M 这个尺寸上，CPU int8（0.66 s）和 GPU bf16（0.55 s）几乎打平。
瓶颈是**逐 token 的 Python/调度开销**，不是算力——端侧直接 CPU 就够，不必为它上 GPU。

### 两核两线程（`bench_2c.py`，亲和性绑 2 个 P 核逻辑 CPU、torch 2 线程）

| 项目 | 数值 |
| --- | --- |
| Gemma int8 单句（16 token） | **0.45~0.56 s**（28~36 tok/s） |
| YOLO CPU 每帧 | 33~55 ms |
| **端到端每帧** | **0.48~0.62 s** |
| 进程内存峰值 | RSS **2.76 GB** / private **3.59 GB** |

**反直觉但实测如此**：2 线程比 14 线程**更快**——
14 线程时 Gemma 0.66 s/24.3 tok/s、YOLO 138 ms；2 线程时 Gemma 0.45 s/35.6 tok/s、YOLO 33~43 ms。
270M 和 640 输入的小检测器都是**同步开销 > 并行收益**，线程开满只是在互相等。
（两组测量条件略有差异：14 线程那组是 pipeline 里含模型加载后的稳定值，2 线程这组关掉了姿态模型，
但 Gemma 的差距与 YOLO 的量级差都在 2~4 倍，方向一致。）

### 单核单线程（`bench_2c.py --cpus 0 --threads 1`）

| 项目 | 数值 |
| --- | --- |
| Gemma int8 单句（16 token） | **0.70 s**（22.9 tok/s） |
| YOLO CPU 每帧 | 58 ms |
| **端到端每帧** | **0.76 s** |
| 进程 RSS 峰值 | **2.91 GB** |
| 进程 private 峰值 | **3.68 GB** |
| 加载后 RSS | 2.88 GB（空进程 0.56 GB） |

**内存结论**：int8 参数本身只有 **0.63 GB**；进程峰值 ~2.9 GB RSS（private ~3.7 GB）主要来自
PyTorch 运行时 + 量化时先载入的 fp32 副本（1.07 GB）及其分配器缓存。
**单核与双核的内存几乎一样**（2.91 vs 2.76 GB），核数只影响时间：单核端到端 0.76 s/帧，双核 0.49~0.60 s/帧。

**精度不受线程数影响**（`probe_threads_precision.py`，test 40 帧，贪心解码）：
1T / 2T / 14T 三档的人数一致率与物体召回**都是 100%**；与 14T 基线逐字一致 **39/40**，
唯一差异是同义改写（“画面上看不到人，椅子清晰可见。” vs “画面里没有看到人，椅子都很清楚。”）。
线程数只改速度（同批 40 帧：1T 0.43 s、2T 0.35 s、14T 0.66 s 每句；1C 基准另一批为 0.70 s，批次间有波动），不改精度。

### 质量（test 100 帧，按视频留出 `hr_fall_detection_1/3`）

| 指标 | 基座 270M | +LoRA |
| --- | --- | --- |
| 人数一致率 | 71% | **100%** |
| 物体召回 | 57% | **100%** |
| 幻觉率（提到场景里没有的物体） | 0% | 1% |
| 平均字数 | 19.7 | 16.6 |
| 平均生成耗时 | 1.28 s | **0.70 s** |

同一提示词（无 few-shot）下的对照；基座的问题是**不会按指令把 JSON 改写成句子**：
它会照抄字段（`"沙发x2"`、`"1280x720像素，正常"`），或把 few-shot 里的「沙发和茶几」抄进无关场景
（证据见 `results/edge_eval_lora.json` 的 `base_rows`）。LoRA 只训练 7.59M 参数（2.75%），
813 步 / 249 秒收敛。

**int8 与 bf16 的精度对照**（同一 100 帧、贪心解码）：

| 指标 | bf16（GPU） | int8（CPU，1 线程） |
| --- | --- | --- |
| 人数一致率 | 100% | **100%** |
| 物体召回 | 100% | **100%** |
| 幻觉率 | 1% | **0%** |
| 平均字数 | 16.1 | 16.3 |
| 逐字一致 | 基线 | **77/100** |

23 条差异全部是同义改写（如“画面中没有看到人，只有椅子。” vs “画面里没有看到人，椅子都很清楚。”），
没有人数、物体或位置的事实性差异。

### 实际输出样句

```
输入 1 人躺卧在左下方，旁边是沙发、茶几
→ 画面左侧有一人躺卧，旁边是沙发和茶几。          （CPU int8）

输入 无人，画面里只有狗（Mage-VL 自带 examples/dog.jpg）
→ 画面中没有看到人，只有狗。

输入 1 人站立在画面中央，旁边有椅子、沙发、床（真实室内帧）
→ 画面中央有一人，旁边是椅子、沙发和床。

输入 1 人站立在右下方，旁边有椅子（test 帧）
→ 画面下方有人站立，椅子就在旁边。
```

## 与 Mage-VL 路线对比

| | Mage-VL 4.74B | 本方案 |
| --- | --- | --- |
| 权重 | 8.83 GiB（INT4 3.53 GiB） | **0.499 GiB** |
| 能否直接读图 | 能（视觉塔） | 不能，靠 YOLO 转符号 |
| 描述粒度 | 开放式、可长可细 | 一句话、只有检测到的类别 |
| 单次成本 | 0.58 s（yes/no）/ 10.8 s（长描述） | ~0.9 s/句 |
| 端侧可行性 | 需 GPU / 大内存 | CPU 可跑 |

## 边界（未验证的部分）

- 检测器是 COCO 80 类，描述只能出现这些类别，没有颜色/材质/动作等细粒度属性。
- 姿态来自 `urfd_posture_v5_mixed`（3 类），跨域 `lie` 召回约 14.4%（历史报告）；
  描述里的「躺卧」只在检测器给置信度时出现，漏检时不提。
- 训练/测试同域（同一家庭、同一机位），**不是跨场景泛化证据**；`hr_1/hr_3` 只是按视频留出。
- 模板监督决定句式分布：输出稳定在 15~20 字，不会产生开放式长描述；「室内」是写死的场景假设，户外帧会说错。
- 视频路径只做了 3 帧 smoke test，没有逐帧事件评估。
- 端侧导出（ONNX / GGUF / RKNN）**未做、未验证**；本机 x86 + RTX 5060 Ti，不是目标硬件。
- 权重放 `models/gemma-3-270m-it`、adapter 放 `out/`，均不入库；可用 `EDGE_GEMMA_MODEL` / `EDGE_ADAPTER` 覆盖。
