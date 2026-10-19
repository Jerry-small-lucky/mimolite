# NOTICE — 数据与许可声明

本仓库代码以 **MIT License** 授权（见 `LICENSE`）。

## 数据合规策略（重要）

**本仓库不随附任何第三方数据集的原文。** 仓库只包含自研产物（代码、自建评测集、
自研合成数据、评测结果档案与文档）。第三方数据仅在本机实验中使用，开源时
**不重新分发**其原文；使用者如需复现，请自行获取数据并遵守各来源许可。

## 数据来源许可一览（已逐项核实）

| 数据 | 来源 | 许可 | 使用方式 | 仓库内状态 |
|---|---|---|---|---|
| HumanEval（164 题） | OpenAI, Chen et al. 2021 | **MIT** | 仅评测锚点 | 不提交 parquet |
| MBPP（257 题） | Google Research, Austin et al. 2021 | **CC-BY-4.0** | 仅评测锚点 | 不提交 parquet |
| GSM8K | OpenAI, Cobbe et al. 2021 | **MIT** | GRPO 奖励数据 | 不提交 jsonl |
| 维基百科语料 | Wikimedia (HF `wikimedia/wikipedia`) | **CC BY-SA 4.0**（旧版本 3.0 兼容；另有 GFDL） | 预训练语料 | 不提交 jsonl/bin |
| 古腾堡英文语料 | Project Gutenberg | **美国版权法下公共领域**（绝大多数作品） | 预训练补充语料 | 不提交 jsonl |
| 自建评测集（100 题） | 本项目原创 | **MIT** | 内分布评测（唯一准绳） | `data/eval_code.jsonl` 随仓库发布 |
| 自研合成数据（seed 7/8/9） | 本项目规则化模板程序生成 | **MIT** | 数据侧迭代主线 | `data/synth_code_*.jsonl` 随仓库发布 |
| GRPO 训练/筛选数据 | 本项目基于模板题筛选（通过率 ≤0.5 / =0） | **MIT** | GRPO 负结果实验资产 | `data/grpo_train.jsonl`、`data/grpo_hard.jsonl` 随仓库发布 |

## 注意事项

1. **CC BY-SA 的说明**：维基语料按 CC BY-SA 授权。本项目不直接分发维基文本，
   仅提供获取与清洗脚本；使用维基语料训练模型的衍生作品义务，请按
   [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 与 Wikimedia
   使用条款自行评估。模型权重若对外发布（如 Hugging Face），建议在模型卡中
   如实披露训练数据来源与许可。
2. **CC-BY 的说明**：MBPP 按 CC-BY-4.0 授权，若再分发其内容须署名
   （Austin et al., "Program Synthesis with Large Language Models", 2021）。
   本仓库不直接分发，仅报告在其上的评测结果。
3. **权重文件**：`runs/` 下的 checkpoint（约 17 GB）与 `out/` 下的 GGUF
   不随 Git 仓库提交；如需模型权重，请通过仓库 Issues 联系作者或等待
   Hugging Face 发布。

## 引用

若在研究中引用本项目，建议引用研究报告：

> 自研 106M 大语言模型数据侧能力迭代研究报告（MiMoLite），2026。
> `docs/自研106M大模型数据侧能力迭代研究报告.docx`
