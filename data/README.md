# data/ 数据资产分类索引

> 文件名遵循既有约定（被脚本/README/报告硬编码引用，**不可改名**）。本文件仅说明各文件的角色与归属，便于检索。

## 1. 评测数据（唯一准绳）

| 文件 | 说明 |
|---|---|
| `eval_code.jsonl` | **自建评测集 100 题**（中文 83 / 英文 17，16 主题簇），与训练严格无泄漏，所有迭代的 pass@1/8/16 均出自此题集 |
| `stdbench/` | P0 外部基准锚点：`humaneval.parquet`（164 题）、`mbpp_test.parquet`（257 题），英文函数级代码生成基准 |

## 2. 合成代码数据（数据侧迭代主线）

| 文件 | 说明 |
|---|---|
| `synth_code_tpl*.jsonl` | 模板合成样本（600 条，seed 7/8/9 三版） |
| `synth_code.jsonl` | 早期合成（35 变体阶段） |
| `synth_code_distill.jsonl` | 蒸馏部分（510 条，仅数据侧合并用） |
| `synth_code_v2*.jsonl` | 去重合并后训练集（1110 条 = 蒸馏 510 + 模板 600，泄漏 0，seed 7/8/9） |
| `synth_code_sft*.jsonl` | SFT 格式（1110 条，10 epochs 训练输入，seed 7/8/9） |

## 3. 原始语料与预训练中间产物（早期流水线，勿动）

| 文件 | 说明 |
|---|---|
| `a001_en/zh.bin/.jsonl` | A001 预训练语料（bin 为 token 缓存） |
| `clean_en/zh.jsonl` | 清洗后语料 |
| `packed_en/zh.bin` | 打包 token 缓存 |
| `pretrain_en/zh.orig.jsonl` | 原始抓取语料 |
| `hq_gutenberg_en.jsonl` | 英文古腾堡高质量语料 |
| `gsm8k.jsonl` | GSM8K（GRPO 数字版奖励数据） |
| `sft.jsonl` | 早期 SFT 数据（字段 messages） |

## 4. GRPO 训练数据

| 文件 | 说明 |
|---|---|
| `grpo_train.jsonl` | GRPO 训练数据 |
| `grpo_hard.jsonl` | 顽固区难例（OOD 筛选） |

## 5. 生成规则

- 合成/合并/评测脚本：`scripts/synth_code_tpl.py`、`scripts/gen_synth*.py`、`scripts/build_train_v2.py`、`scripts/prepare_sft.py`、`eval_code.py`、`eval_stdbench.py`
- 历史数据生成脚本已归档至 `scripts/archive/`（legacy / fix）
