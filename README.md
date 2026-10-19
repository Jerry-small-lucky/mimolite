# MiMoLite —— 106M 自研大语言模型：无蒸馏规则化数据工程

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

从零预训练、参数仅 106M 的代码生成语言模型。**不蒸馏、不改模型结构、评测为唯一准绳**，
全部实验在一张 **RTX 5070 8GB** 笔记本上完成。核心成果是"数据侧能力迭代"的方法论：
8 版同口径实验、4 个根因修复、2 次负优化的归因，以及一份完整可复现的研究报告。

> 本项目定位为**可复现的科研叙事**，而非模型表演：106M 参数在标准英文基准上
> 能力有限（见"外部基准锚点"），真正的贡献在于把"从 0 到 29% pass@1"的
> 数据工程过程做成透明、诚实、可复核的完整记录。

## 核心结果（自建 100 题代码评测集，同训练量只换数据）

| 版本 | 做法 | pass@1 / pass@8 / pass@16 |
|---|---|---|
| 基座 a001_B1_v2 | 预训练 4500 步 | 0 / 0 / 0 |
| **sft_v8（当前最佳）** | 600 条规则化模板 + 评测频次加权（HOT16/其他 6），10 epochs | **29 / 57 / 59** |
| sft_v9 | 删 4 干扰变体 + 极端强化 3 函数 | 20（负优化 → 预算守恒铁律） |
| sft_v10 | 确定性配额 + 扩大总量 | 28 / 51（负优化 → 低样本函数采样命中下降） |

### 外部基准锚点（泛化鸿沟，诚实披露）

| 评测集 | 题量 | pass@1 | pass@8 |
|---|---:|---:|---:|
| HumanEval（英文标准） | 164 | 0/164 (0%) | 0/164 (0%) |
| MBPP（英文标准） | 257 | 0/257 (0%) | 0/257 (0%) |
| 自建 100 题（内分布） | 100 | 29/100 (29%) | 57/100 (57%) |

内分布 29/57 与外分布 0 之间的**泛化鸿沟**说明：数据侧迭代把能力绑定在训练分布的
题面—函数名组合上，尚未形成跨分布泛化。判定经逐例诊断确认真实；评测口径
（指令式适配、低温单采样）已在研究报告中完整披露。

### 多 seed 稳健性（结论非单点噪声）

| 版本 | 题量 | pass@1 | pass@8 |
|---|---:|---:|---:|
| sft_v8（seed 7，基线） | 100 | 29/100 (29%) | 57/100 (57%) |
| sft_v8_seed8 | 100 | 31/100 (31%) | 58/100 (58%) |
| sft_v8_seed9 | 100 | 24/100 (24%) | 52/100 (52%) |

同一数据链配置 3 个随机种子重复：pass@1 均值 **28%**、pass@8 均值 **56%**。

### 负结果（完整记录，不回避）

- **GRPO 四轮实验**：在 106M + 8GB 规模下 RL 暂不可行（无强教师信号、奖励稀疏），
  负结果与归因完整记录在报告第八章，资产保留。
- **负优化实验**：v9/v10 证明"有限预算内强化 A 必然削弱 B"（预算守恒律），
  由此提出研究提案（见下方文档）。

## 为什么值得看

1. **几乎没人占的格子**：106M + 无教师蒸馏 + 纯数据工程 + 8GB 单卡。
2. **诚实负结果**：外分布 0%、GRPO 不可行——不藏、不夸。
3. **完整复现链**：数据合成 → 去重合并 → SFT → 评测，全部命令可复现。
4. **研究方法论产出**：预算守恒律 / 函数名竞争律 / 多采样恢复（研究提案）。

## 文档

- 研究报告：`docs/自研106M大模型数据侧能力迭代研究报告.docx`（PDF 同目录）
- 研究提案：`docs/研究提案-预算守恒与函数名竞争.docx`
- 数据许可声明：[`NOTICE.md`](NOTICE.md)

## 目录结构

```
<项目根>/
├── 核心脚本（根目录，主线可复现链）
│   ├── model.py                 架构定义（MiMoLite）
│   ├── pretrain.py              阶段一：预训练
│   ├── sft.py                   阶段二：指令微调
│   ├── grpo.py / grpo_code.py   阶段三：RL（GSM8K 数字版 / 代码版沙箱奖励）
│   ├── select_grpo_data.py      GRPO 数据筛选
│   ├── eval_code.py             评测：自建 100 题（p1/p8/p16）
│   ├── eval_stdbench.py         评测：标准基准（HumanEval / MBPP）
│   ├── export_gguf.py           导出 GGUF
│   └── ablate.py                消融实验
├── data/                        数据资产（分类索引见 data/README.md）
│   ├── eval_code.jsonl          自建评测集 100 题（随仓库发布，MIT）
│   ├── synth_code_*.jsonl       自研合成代码数据（随仓库发布，MIT）
│   └── stdbench/                标准基准 parquet（不提交，见 NOTICE）
├── scripts/                     数据链与报告维护脚本
│   └── archive/                 （probe 探针 / fix 修复 / ops 运维 / legacy 早期流水线）
├── docs/                        报告与提案（docx + pdf）
├── logs/                        评测证据档案（eval_*.json）
├── runs/                        权重（不提交，见 NOTICE）
├── out/                         GGUF（不提交，见 NOTICE）
├── tokenizer/                   tokenizer.json + 元信息
└── research/                    调研资料
```

## 复现主线（从数据到结果）

```powershell
cd mimolite   # 克隆本仓库后

# 1) 数据链：模板合成 → 去重合并（泄漏 0）→ SFT 格式
python scripts/synth_code_tpl.py --n 600 --seed 7
python scripts/build_train_v2.py --tpl data/synth_code_tpl.jsonl --out data/synth_code_v2.jsonl
python scripts/prepare_sft.py --raw data/synth_code_v2.jsonl --min 1

# 2) 训练（10 epochs，330 步，GPU 空闲时约 2 分钟）
python sft.py --ckpt runs/a001_B1_v2/final.pt --data data/synth_code_sft.jsonl --run-name sft_vX --epochs 10 --val-n 60

# 3) 评测（自建 100 题，pass@1/8/16）
python eval_code.py --ckpt runs/sft_vX/final.pt --out logs/eval_code_sftvX.json --samples 1
python eval_code.py --ckpt runs/sft_vX/final.pt --out logs/eval_code_sftvX_p8.json --samples 8

# 4) 标准基准锚点（需自行获取 HumanEval / MBPP parquet，见 NOTICE）
python eval_stdbench.py --ckpt runs/sft_v8/final.pt --bench humaneval --samples 8
python eval_stdbench.py --ckpt runs/sft_v8/final.pt --bench mbpp --samples 8
```

完整跑一遍（预训练 → tokenizer → SFT → GRPO → GGUF）见文末「附：完整跑一遍」。

## 环境要点（实测，避免重复踩坑）

| 项 | 情况 |
|---|---|
| GPU | RTX 5070 Laptop，**8GB** 显存，compute capability 12.0 |
| 训练栈 | torch 2.12.1+cu130 / transformers 5.17 / trl 1.13 / peft 0.20 |
| 推理栈 | llama.cpp（0.4.0-dev，CUDA 已编译） |
| 网络 | huggingface.co 直连超时，用 `https://hf-mirror.com` |
| 微基准 | encode_batch 13.7M 字符/s；训练 14.1k token/s（bs=8 + 梯度检查点，显存 3.7GB） |

已踩坑记录（详见报告 7.2 与 README 下方）：HF 直连不通、RoPE 必须对齐 llama.cpp
（interleaved 风格）、GGUF 张量必须真 F16、llama-cli 交互式需 stdin=DEVNULL、
PowerShell Out-File 缓冲、byte-level BPE 内存膨胀、MTP targets 别切两次。

## 模型架构

108.16M 参数（训练口径，不含 MTP），8 层 / 隐藏 960 / GQA（15 query 头 / 5 KV 头）/
SWA:GA = 6:1（128 窗口）/ SwiGLU / RoPE(interleaved) / MTP 1 层（权重 0.3，导出后
总参 110.12M）。预训练 4500 步，中英 75:25 动态混合。架构口径细节见研究报告。

## 消融实验（ablate.py）

SWA/GA 比例与位置（a003）、深度 vs 宽度（a001）、WSD×GradPower（a005）——
等参数量、等 token 预算、同验证集三条纪律强制，结论见报告。

## 许可

- 代码与自研数据：**MIT**（[LICENSE](LICENSE)）
- 数据来源与合规说明：[NOTICE](NOTICE.md)（第三方数据不随仓库分发）

## 联系

GitHub 仓库：<https://github.com/Jerry-small-lucky/mimolite>
模型权重：已发布（Release v0.1.0，FP16 GGUF 210.7 MiB + Q4_K_M 79.1 MiB）：
<https://github.com/Jerry-small-lucky/mimolite/releases/latest>
如需 checkpoint 或其他格式，请在仓库 Issues 留言。

---

# 附：完整跑一遍（预训练 → 推理）

```powershell
cd mimolite

# 0) 语料（约 10 分钟，1.3GB 中文 + 0.85GB 英文维基；脚本已归档）
python scripts/archive/legacy/prepare_corpus.py --zh 400000 --en 200000

# 1) tokenizer（约 2 分钟；脚本已归档）
python scripts/archive/legacy/train_tokenizer.py --vocab 16384 --sample 80000

# 2) token 缓存（约 6 分钟，产出 .bin）
python pretrain.py --prepare

# 3) 阶段一：预训练（--steps 决定时长）
python pretrain.py --steps 9000 --micro-bs 8 --accum 4 --lr 1.5e-3

# 4) 阶段二：SFT
python scripts/archive/fix/prepare_sft_data.py
python sft.py --epochs 2 --lr 1e-4

# 5) 阶段三：GRPO 强化学习
python grpo.py --groups 300 --group-size 8

# 6) 导出 GGUF 并用 llama.cpp 跑
python export_gguf.py --ckpt runs/sft/final.pt --out out/mimolite
llama-cli -m out/mimolite.gguf -p "你好" -n 128   # 路径以你的 llama.cpp 安装为准
```

### 时长换算（按实测 14.1k token/s）

一个 step = micro_bs × accum × (seq_len+1) = 8×4×513 = **16,416 token**

| steps | 训练 token | 预计时长 |
|---:|---:|---:|
| 500 | 8M | 10 分钟（试水） |
| 3,000 | 49M | 1 小时 |
| 9,000 | 148M | 3 小时 |
| 20,000 | 328M | 6.5 小时（约 1 个 epoch） |

### GGUF 导出与推理要点

- `tokenizer.ggml.pre` 必须是 `"gpt-2"`（与训练一致），写成 `"default"` 不报错但切分错乱。
- 张量必须真 F16（早期全部 F32 声明 F16 导致体积翻倍）。
- llama-perplexity / llama-bench 打不开含中文路径，需设工作目录 + ASCII 相对名。
- llama-cli 是交互式，脚本里必须 `stdin=DEVNULL`。

## MiMo 资料来源

- MiMo-V2-Flash 技术报告：<https://mimo.xiaomi.com/zh/paper/mimo-v2-flash>
- MiMo-7B 技术报告：<https://mimo.xiaomi.com/zh/paper/mimo-reasoning>
- MiMo 官方站：<https://mimo.xiaomi.com> · GitHub：<https://github.com/XiaomiMiMo/MiMo>
