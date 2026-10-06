# ZH-HUMOR：中文多模态幽默检测数据集（Multimodal Chinese Humor Detection Dataset）

`ZH-HUMOR` 是首个面向**中文多模态幽默检测**的开源数据集，包含 **文本（Text）、声学（Acoustic）、视觉（Visual）** 三种模态，其数据格式与公开的英文多模态幽默数据集 [UR-FUNNY](https://github.com/ROC-HCI/UR-FUNNY) 完全同构，可直接复用其模型与评测流程。

本数据集以**观众笑声**作为幽默的监督信号，通过全自动标注管线从 Bilibili 脱口秀节目中构建，每个样本由「上文讲授（context，3 句）+ 笑点句（punchline，一句）」构成，标注该笑点句是否为幽默（正样本，伴随真实观众笑声）。

**ZH-HUMOR（中文多模态幽默检测数据集）· Multimodal Chinese Humor Detection Dataset**

> - 领域：多模态情感计算 / 幽默检测 / Computational Humor
> - 语言：中文（简体）
> - 模态：文本 300 维 · 声学 65 维 · 视觉 178 维（词级对齐）
> - 规模：12 个视频 · **1,998 个样本**（正 942 / 负 1,056）
> - 格式：UR-FUNNY 同构 pkl，含划分好的 train / dev / test
> - 用途：仅限学术研究

---

## Table of Contents

- [News / 更新](#news--更新)
- [1. 数据集概览 (Overview)](#1-数据集概览-overview)
- [2. 数据格式 (Data Format & Schema)](#2-数据格式-data-format--schema)
- [3. 特征说明 (Features)](#3-特征说明-features)
- [4. 目录结构 (Repository Structure)](#4-目录结构-repository-structure)
- [5. 快速开始 (Quick Start)](#5-快速开始-quick-start)
  - [5.1 加载数据](#51-加载数据)
  - [5.2 训练 C-MFN 模型](#52-训练-c-mfn-模型)
  - [5.3 评估与复现](#53-评估与复现)
- [6. 实验结果 (Results)](#6-实验结果-results)
- [7. 标注方法 (Annotation Pipeline)](#7-标注方法-annotation-pipeline)
- [8. 质量筛选子集 (Quality Filtering)](#8-质量筛选子集-quality-filtering)
- [9. 可视化工具 (Visualization)](#9-可视化工具-visualization)
- [10. 数据来源与许可 (Source & License)](#10-数据来源与许可-source--license)
- [11. 引用 (Citation)](#11-引用-citation)
- [12. 致谢 (Acknowledgement)](#12-致谢-acknowledgement)

---

## News / 更新

- **v1.0（2026-10）**：发布完整 12 视频 / 1,998 样本数据集，含三模态特征与 3 个质量筛选子集（`pkl_q1` / `pkl_q1z` / `pkl_q2`），随附复现脚本与留一视频（LOVO）评估代码。

---

## 1. 数据集概览 (Overview)

### 1.1 领域动机

幽默是沟通中最微妙也最复杂的交际行为之一，常依赖**台词**（文本）、**语气音调**（声学）与**表情体态**（视觉）三者共同传递。公开幽默数据集（UR-FUNNY、MHD、HaHackathon 等）以英文为主，**中文多模态幽默数据长期空白**。ZH-HUMOR 旨在填补这一空白。

### 1.2 构建思路

我们观察到：**真实观众的笑声**是幽默最自然、最客观的监督信号。因此采用「笑声事件对齐到笑点句」的全自动标注方案（详见第 7 节），无需人工逐条标注，可快速扩展。

### 1.3 统计信息

**表 1：总量统计**

| 统计项 | 数值 |
|---|---|
| 样本总数 | **1,998** |
| 正样本（幽默，伴随笑声） | 942（47.1%） |
| 负样本（非幽默，无笑声） | 1,056（52.9%） |
| 标注视频数 | 12 |
| 中文词表大小 | 5,313 |
| punchline 平均词数 | 4.8 |
| 平均 context 句数 | 3 |
| 模态维度（文本 / 视觉 / 声学） | 300 / 178 / 65（总 543） |
| 划分（train / dev / test） | 1,258 / 125 / 615 |

**表 2：与 UR-FUNNY（英文）对比**

| 统计项 | UR-FUNNY | ZH-HUMOR |
|---|---|---|
| 样本总数 | 10,166 | 1,998 |
| 视频来源 | TED 演讲 | Bilibili 脱口秀 |
| punchline 平均词数 | 15.9 | 4.8 |
| context 平均句数 | 5.1 | 3 |
| 特征提取 | FastText + OpenFace + COVAREP | jieba + MediaPipe FaceMesh + openSMILE |

**表 3：12 个标注视频明细**

| BV 号 | 标题 | 时长(min) | 样本 | 正/负 |
|---|---|---|---|---|
| BV1Xb411d7RJ | 徐志胜 ·《脱口秀大会S5》 | 62.5 | 424 | 196/228 |
| BV1kk4y1H7WM | 童漠男 ·【倾听】北下关大爷 | 63.8 | 381 | 179/202 |
| BV1he4y1q7Mw | 何广智 · 躺平 vs 漂亮妹妹 | 55.4 | 296 | 142/154 |
| BV1cG4y1h7Uj | 徐志胜 · 暴力丑学 | 42.2 | 242 | 108/134 |
| BV1DN411j7xo | 刘旸教主 ·《伊卡洛斯》专场 | 86.5 | 234 | 113/121 |
| BV1E24y1i7wm | 脱口秀S6冠军预定剪辑 | 50.4 | 184 | 88/96 |
| BV1CY4y187Ur | 刘旸教主 ·《天生有意思》专场 | 79.4 | 115 | 57/58 |
| BV1e94y1F7V8 | 罗永浩 · 爆笑脱口秀 | 26.7 | 60 | 29/31 |
| BV1Fk4y1Q7Xr | 刘震云 · 哲学式脱口秀 | 25.5 | 31 | 15/16 |
| BV1m24y1C7DZ | 傅首尔 · 辩手脱口秀 | 9.0 | 13 | 6/7 |
| BV1QD4y1Y74r | 小沈龙经典脱口秀 | 15.9 | 10 | 5/5 |
| BV1Qe4y1T7Rc | 小沈龙搞笑段子 | 37.5 | 8 | 4/4 |

> 注：`BV1kk4y1H7WM` 为纯音频配图视频，`BV1DN411j7xo` 画面过暗，二者视觉通道几乎无可用特征（人脸检测率 0%），主要用于文本/声学实验，在新版本中建议启用 **质量筛选子集**（见第 8 节）。

---

## 2. 数据格式 (Data Format & Schema)

数据以 **pkl**（Python pickle, protocol 4）存储，与 `UR-FUNNY` 数据接口一致。目录：`zh_talkshow/pkl/`。

### 2.1 文件清单

| 文件 | 说明 | 键 → 值 |
|---|---|---|
| `language_sdk.pkl` | 文本内容与词级时间戳 | `样本id` → 字典（见下） |
| `humor_label_sdk.pkl` | 标签 | `样本id` → `0`/`1` |
| `data_folds.pkl` | 数据划分 | `{'train': [id], 'dev': [id], 'test': [id]}` |
| `word_to_idx.pkl` | 词表 | `词` → `索引` |
| `word_embedding_list.pkl` | 词向量 | `[向量, ...]`，索引与词表一致 |
| `video_sdk.pkl` | 视觉特征 | `样本id` → `punchline_features` / `context_features` |
| `acou_sdk.pkl` | 声学特征 | `样本id` → `punchline_features` / `context_features` |

### 2.2 `language_sdk.pkl` 单样本结构

```python
{
    'root': 'BV1CY4y187Ur',                # 来源视频
    'context_sentences': ['我当时想的是', '我先干一年全职脱口秀演员', '我不行我再回去嘛'],
    'punchline_sentence': '培训行业还能完了是怎么着',
    'context_intervals': [ [[s,e],...],     # 每句内的词级时间戳（秒）
                           [[s,e],...],
                           [[s,e],...] ],
    'punchline_intervals': [[s,e], ...],    # 笑点句的词级时间戳
    'punchline_embedding_indexes': [],      # 词索引（特征阶段回填）
    'context_embedding_indexes': [[...], [...], [...]],  # 每句词索引
}
```

### 2.3 特征拼接顺序（重要）

`dataloader.py` 中每词特征按以下顺序拼接：

```
[ 文本 300 维 | 声学 65 维 | 视觉 178 维 ]
```

- Punchline：`(K, 543)`，`K` = 笑点句词数（pad 至 20）
- Context：`(N, W, 543)`，`N` = 3 句，`W` = 每句词数（pad 至 20）

> 训练时若使用自定义拆分，务必注意**视觉/声学通道顺序**，避免模态错位（详见本仓库 `reproduce/cmfn.py` 的注释与修复说明）。

---

## 3. 特征说明 (Features)

### 3.1 文本（300 维）

- 分词：`jieba`
- 词向量：中文词表随机初始化 300 维（v1.0）；建议后续替换为预训练词向量（word2vec / Tencent AI Lab Embedding）以获得更优效果。

### 3.2 声学（65 维）

- 工具：`openSMILE` 提取 **ComParE-2016** 词级 LLD 特征
- 处理：对每个词覆盖的音频帧特征取**平均值**，得到词级 65 维特征。

### 3.3 视觉（178 维）

- 工具：`MediaPipe FaceMesh`（模型 `face_landmarker.task`）
- 组成：40 个人脸关键点 × 3 坐标（120）+ 52 个 blendshape 表情系数 + 6 维 pose = **178 维**
- 说明：原英文数据集使用 OpenFace 提取 371 维；因 Windows 下 OpenFace 依赖过重，本数据集改用 MediaPipe FaceMesh 作为跨平台替代。

---

## 4. 目录结构 (Repository Structure)

```
ZH-HUMOR/
├── zh_talkshow/
│   ├── pkl/                  # 主数据集（UR-FUNNY 同构 pkl）
│   │   ├── language_sdk.pkl
│   │   ├── humor_label_sdk.pkl
│   │   ├── data_folds.pkl
│   │   ├── word_to_idx.pkl
│   │   ├── word_embedding_list.pkl
│   │   ├── video_sdk.pkl
│   │   └── acou_sdk.pkl
│   ├── pkl_q1/               # 质量筛选 · 人脸率≥50%（3 视频 / 697 样本）
│   ├── pkl_q1z/              # 质量筛选 · q1 再去零脸样本（490 样本）
│   ├── pkl_q2/               # 质量筛选 · 人脸率≥40%（5 视频 / 1,177 样本，推荐）
│   ├── audio/                # 提取的 16kHz 单声道音频（wav，按 BV 号）
│   ├── asr/                  # 词级转写 JSON（faster-whisper）
│   ├── laughter/             # YAMNet 笑声区间检测结果
│   ├── samples/
│   │   └── samples.csv       # 标注样本（context + punchline + label + 时间戳）
│   ├── manifests/            # 视频候选清单 candidates.csv
│   └── models/               # 特征提取模型（face_landmarker.task）
├── reproduce/                # 复现与训练代码
│   ├── dataloader.py         # 数据加载（HumorDataset）
│   ├── cmfn.py               # C-MFN 模型（已修复模态拆分）
│   ├── train_cmfn.py         # 训练脚本（urfunny / zh）
│   ├── train_lovo.py         # 留一视频交叉验证（LOVO）
│   ├── select_videos.py      # 视频质量筛选
│   ├── build_text_feat.py    # 文本特征
│   ├── build_audio_feat.py   # 声学特征
│   ├── build_visual_feat.py  # 视觉特征
│   ├── export_observable.py  # pkl -> 可观察文本/音频/视频
│   └── ...
└── README.md
```

---

## 5. 快速开始 (Quick Start)

环境要求：`Python 3.10+`，`torch`，`scikit-learn`，`numpy`。

### 5.1 加载数据

```python
import pickle
pkl = "zh_talkshow/pkl"

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

lang  = load(f"{pkl}/language_sdk.pkl")
label = load(f"{pkl}/humor_label_sdk.pkl")
folds = load(f"{pkl}/data_folds.pkl")

print("样本数:", len(lang))                      # 1998
print("正样本:", sum(v == 1 for v in label.values()))  # 942
print("划分:", {k: len(v) for k, v in folds.items()})  # {'train':1258,'dev':125,'test':615}

# 查看一个样本
sid = folds["test"][0]
s = lang[sid]
print("punchline:", s["punchline_sentence"], "| label:", label[sid])
```

### 5.2 训练 C-MFN 模型

```bash
cd reproduce
python train_cmfn.py --dataset zh --epochs 15 --lr 1e-3
```

若要使用质量筛选子集，可改用 `train_lovo.py`：

```bash
# 以 q2（推荐）做留一视频交叉验证
python train_lovo.py --data-dir "../zh_talkshow/pkl_q2" --dims 300,178,65
```

### 5.3 评估与复现

`eval_zh.py` 用于评估 `zh` 数据集上训练的已保存模型：

```bash
python eval_zh.py          # 默认加载 zh_cmfn_best.pt
```

`eval_urfunny_lowmem.py` 用于低内存评估 UR-FUNNY 官方数据集上的复现结果。

---

## 6. 实验结果 (Results)

在相同配置下，以 **留一视频交叉验证（LOVO）** 公平评估 C-MFN 在三类数据集上的表现：

| 数据集 | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| **UR-FUNNY（修复后复现）** | **64.59%** | 66.98% | 61.98% | 72.86% |
| UR-FUNNY（论文原文） | 65.23% | — | — | — |
| ZH-HUMOR 全量 12 视频 | 57.11% | 47.33% | 56.20% | 40.87% |
| **ZH-HUMOR 质量集 q2（5/1177）** | **58.54%** | **58.64%** | 54.83% | **63.02%** |
| ZH-HUMOR 质量集 q1（3/697） | 59.25% | 47.79% | 57.78% | 40.75% |

> 结论：修复模态拆分后 UR-FUNNY 复现与论文（65.23%）高度对齐；中文数据集采用**质量筛选 + LOVO 评估**可将 Recall 提升约 22pp（40.87% → 63.02%）。

---

## 7. 标注方法 (Annotation Pipeline)

以「真实观众笑声」作为幽默监督信号，全自动构建样本。流程如下：

```
视频下载 → 音频提取(ffmpeg, 16kHz/单声道)
        → 词级转写(faster-whisper / WhisperX, 词级时间戳)
        → 笑声检测(YAMNet, 检测笑声事件并合并相邻<1s片段)
        → 样本生成(以笑声锚点对齐笑点句, 生成 context+punchline+label)
        → 特征提取(文本 jieba / 声学 openSMILE / 视觉 MediaPipe)
        → pkl 打包(UR-FUNNY 同构)
```

- **正样本**：紧随观众笑声、以笑声结束的笑点句（`label=1`）
- **负样本**：无笑声的对照句（`label=0`），与正样本同视频、同长度分布
- **防泄漏**：train/dev/test 按**视频（root）分层**划分，同一视频的样本不会跨越多折

对应脚本：`reproduce/` 下的 `extract_audio.py` / `transcribe.py` / `detect_laughter.py` / `build_samples.py` / `build_pkl.py`。

---

## 8. 质量筛选子集 (Quality Filtering)

不同视频的**视觉质量**差异较大（人脸入画率、光照、机位）。提供按「人脸检测率 / 样本量」筛选的子集，供需要视觉模态的任务使用：

| 子集 | 筛选条件 | 视频/样本 | 备注 |
|---|---|---|---|
| `pkl_q1` | 人脸率 ≥ 50% 且样本 ≥ 15 | 3 / 697 | 高质量 |
| `pkl_q1z` | q1 再丢视觉全零样本 | 3 / 490 | 过小，不推荐 |
| `pkl_q2` | 人脸率 ≥ 40% 且样本 ≥ 15 | 5 / 1,177 | **推荐**：数据量与纯度均衡 |

重新筛选：

```bash
cd reproduce
python select_videos.py --min-face 0.4 --min-n 15 --out pkl_q2
```

---

## 9. 可视化工具 (Visualization)

将 pkl 中的特征/时间戳转回可观察的文本、音频与视频片段，方便人工核验标注质量：

```bash
cd reproduce
python export_observable.py
```

> 输出的文本可在终端查看；音频/视频片段需配合原始 Bilibili 视频文件（按 BV 号定位到 `punchline_intervals` 时间区间）截取。

---

## 10. 数据来源与许可 (Source & License)

- **数据来源**：Bilibili 平台公开的脱口秀/单口喜剧节目视频，原始版权归 Up 主及出品方所有。
- **本数据集包含**：转写文本、笑声标注、派生特征（文本/声学/视觉向量）与样本划分，**不含**原始视频/音频文件的版权内容。
- **许可**：本数据集**仅限学术研究使用**，禁止商业用途。请遵守 Bilibili 用户协议及原作者版权。引用本数据集时请注明（见第 11 节）。
- **伦理说明**：标注基于「是否有观众笑声」这一客观信号，不涉及对表演者个人价值的主观评判。

> 如需在论文中使用，建议在数据描述部分明确数据来源与使用限制，并考虑依法获得相关授权。

---

## 11. 引用 (Citation)

若本数据集或代码对你的研究有帮助，请引用：

```bibtex
@misc{zhhumor2026,
  title  = {ZH-HUMOR: A Multimodal Chinese Humor Detection Dataset},
  author = {Yanyu Dong},
  year   = {2026},
  howpublished = {\url{https://github.com/yourname/ZH-HUMOR}}
}
```

本工作基于以下多模态幽默检测方法：

```bibtex
@inproceedings{hasan2019urfunny,
  title     = {{UR-FUNNY}: A Multimodal Language Dataset for Understanding Humor},
  author    = {M. K. Hasan and Wasifur Rahman and AmirAli Bagher Zadeh and Jianyuan Zhong and Md Iftekhar Tanveer and Louis-Philippe Morency and Mohammed (Ehsan) Hoque},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  year      = {2019},
  pages     = {6306--6312},
  address   = {Hong Kong, China}
}
```

---

## 12. 致谢 (Acknowledgement)

- 方法框架：`C-MFN` / `UR-FUNNY`（Hasan et al., 2019）
- 特征工具：`openSMILE`、`MediaPipe FaceMesh`、`faster-whisper` / `WhisperX`、`YAMNet`
- 数据来源：Bilibili 各 Up 主

---

如有问题或合作意向，欢迎提交 [Issue](<仓库链接>) 或邮件联系（448429930@qq.coms）。