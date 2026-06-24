# ReTool：面向代码增强数学推理的 SFT + Sandbox RL 训练链路

## 摘要

本项目实现了一条面向数学推理模型的 ReTool 训练链路：先用带代码执行痕迹的
监督微调数据让模型学会“写代码、读执行结果、给最终答案”的格式，再在 veRL
中接入异步 Python sandbox，让模型在 RL rollout 阶段真正执行生成的代码，并用
最终答案 reward 训练模型。最终产物是一个可用 Hugging Face/Transformers 加载
的融合后 RL checkpoint：

```text
retool-math-l1-l3-dapo-lora-r64-global_step_200-fused-hf
```

当前仓库记录的是完整工程链路和可复现实验配置。正式 benchmark 还没有作为结果
报告；目前可确认的结果来自训练日志、checkpoint 产物、sandbox smoke，以及少量
人工对话观察。

## 1. 研究目标

普通数学模型经常会“口头写代码”，但并没有真实执行，也不会稳定地把执行结果
反馈进推理过程。这个项目关注的问题是：

1. 如何把 ReTool 风格的代码辅助推理变成可训练的数据格式；
2. 如何在 RL rollout 阶段接入真实 sandbox，而不是让模型伪造
   `<interpreter>...</interpreter>`；
3. 如何把最终答案格式和 reward 对齐，减少“推理过程看起来对但最终答案不可判分”
   的情况；
4. 如何在有限显存和磁盘预算下完成 SFT、RL、checkpoint merge、发布和复现。

核心约束是所有模型响应最终必须落到同一个答案协议：

```text
Answer: <final answer>
```

reward 以最后的 `Answer:` 行为主要判分对象。代码执行只是帮助模型得到答案，
不会替代最终答案。

## 2. 完整链路

整体流程如下：

```text
问题数据
  -> 生成/整理 ReTool SFT messages
  -> veRL SFT，得到 merged HF SFT checkpoint
  -> 构造 MATH level 1-3 RL prompt parquet
  -> veRL DAPO/GRPO 风格 rollout
  -> 异步 Python sandbox 执行代码
  -> math reward 根据最终答案判分
  -> LoRA/FSDP checkpoint merge + fuse
  -> 发布最终 HF checkpoint
```

### 2.1 SFT 数据

SFT 数据位于：

```text
data/sft/train.jsonl
```

它使用 Hugging Face chat messages 格式：

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

当前文件包含 2002 行：主体来自 `JoeYing/ReTool-SFT`，并合入少量本项目生成样例。
只有 2 行的临时样例文件已经删除，避免把 smoke 输出误当成训练集。

如需继续生成 SFT 样例，默认写入 ignored 文件：

```bash
set -a; source .env; set +a
python gen_data.py \
  --question "What is the sum of all integers from 1 to 100?" \
  --out data/sft/generated.jsonl \
  --model "$GEN_MODEL"
```

`gen_data.py` 会执行生成的 Python 代码， materialize 真实
`<interpreter>...</interpreter>` 输出，并要求末尾存在唯一的 `Answer:` 行。

### 2.2 SFT 训练

SFT 基座为 Qwen2.5-3B，训练入口是：

```bash
DATA_JSONL=/root/autodl-tmp/retool/data/sft/train.jsonl \
MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-3B \
bash scripts/run_verl_sft.sh
```

训练后使用 veRL/FSDP checkpoint merge，得到可直接加载的 HF checkpoint：

```text
/root/autodl-tmp/retool/runs/merged/retool-qwen2_5-3b-sft-epoch3-global_step_941-hf
```

### 2.3 RL 数据

RL prompt 数据定义在：

```text
data/rl/math_l1_l3/
```

提交到仓库中的文件是：

```text
prompt.txt   # MATH level 1-3 prompt 模板
meta.json    # 数据来源、level/subject 统计、生成命令
README.md    # 数据说明
```

生成的 parquet 文件不进 git。默认生成命令：

```bash
python scripts/prepare_math_level_data.py
```

当前 RL 数据规格：

| 字段 | 值 |
| --- | --- |
| 数据源 | `EleutherAI/hendrycks_math` |
| level | 1 到 3 |
| unique rows before split | 3504 |
| repeat | 10 |
| train rows after repeat | 33760 |
| val rows | 128 |
| subjects | prealgebra, algebra, number theory, counting/probability, geometry, intermediate algebra, precalculus |

### 2.4 Sandbox RL

RL 训练使用 `scripts/run_dapo_smoke.sh`。这个脚本名字里仍有 smoke，但已经参数化，
可以通过环境变量启动 DAPO/GRPO 风格训练：

```bash
export MODE=lora
export LORA_RANK=64
export MODEL_PATH=/root/autodl-tmp/retool/runs/merged/retool-qwen2_5-3b-sft-epoch3-global_step_941-hf
export TRAIN_FILE=/root/autodl-tmp/retool/data/rl/math_l1_l3/train.parquet
export VAL_FILE=/root/autodl-tmp/retool/data/rl/math_l1_l3/val.parquet
export USE_RETOOL_SANDBOX=True
export ROLLOUT_N=8
export TRAIN_BATCH_SIZE=2
export MAX_RESPONSE_LENGTH=2048
export SAVE_FREQ=50

mkdir -p logs
nohup bash scripts/run_dapo_smoke.sh \
  reward.custom_reward_function.path=/root/autodl-tmp/retool/retool_sandbox/math_reward.py \
  reward.custom_reward_function.name=compute_score \
  > logs/${EXPERIMENT_NAME:-retool-dapo}.log 2>&1 &
```

真实工具链路由 `retool_sandbox/` 提供：

```text
模型生成到 </code>
  -> AsyncPythonSandboxPool 执行代码
  -> 真实 stdout 写回 <interpreter>...</interpreter>
  -> 模型继续生成
  -> reward 读取最后的 Answer: ...
```

这点很重要：`scripts/infer_hf.py` 是普通 generation，里面出现的
`<interpreter>` 文本并不代表真实执行。真实工具调用必须使用
`scripts/infer_hf_with_sandbox.py` 或 veRL agent loop。

## 3. 训练结果与证据

### 3.1 SFT loss

SFT 训练形成 3 个连续阶段的 merged checkpoint，验证 loss 持续下降：

| 阶段 | 训练步 | final `val/loss` |
| --- | ---: | ---: |
| epoch 1 | global_step 941 | 0.5525964499 |
| epoch 2 | global_step 941 | 0.5473101735 |
| epoch 3 | global_step 941 | 0.5376862288 |

![SFT validation loss](docs/assets/sft_val_loss.png)

这个曲线说明模型学到了 ReTool 输出分布和答案格式，但不能单独证明数学能力提升。

### 3.2 RL 训练信号

最终保留的 RL run 是：

```text
retool-math-l1-l3-dapo-lora-r64-sandbox-b2-n8-r2048-s50-1000-fixreward-20260624_023756
```

关键配置：

| 字段 | 值 |
| --- | --- |
| base model | epoch-3 SFT merged HF checkpoint |
| algorithm path | DAPO/GRPO 风格 veRL PPO trainer |
| mode | LoRA |
| LoRA rank | 64 |
| train batch size | 2 |
| rollout n | 8 |
| max response length | 2048 |
| save freq | 50 |
| final saved step | global_step 200 |

核心信号：

1. reward 有区分度：同组 rollout 中同时存在正负样本；
2. agent loop 确实运行：`num_turns/mean` 非零，说明不是纯文本 generation；
3. 最终公开 checkpoint 来自 `global_step_200`，是阶段性产物，不是完整收敛模型。

| step | `critic/score/mean` | `critic/score/max` | `critic/score/min` | `response_length/mean` | `num_turns/mean` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | -0.625 | 1.0 | -1.0 | 802.75 | 2.8125 |
| 100 | -0.500 | 1.0 | -1.0 | 1122.38 | 2.6250 |
| 150 | -0.250 | 1.0 | -1.0 | 703.69 | 2.9375 |
| 199 | -0.500 | 1.0 | -1.0 | 817.63 | 2.6875 |

![RL reward mean](docs/assets/rl_rewards_mean.png)

![RL response length mean](docs/assets/rl_response_length_mean.png)

![RL number of turns mean](docs/assets/rl_num_turns_mean.png)

### 3.3 当前可发布 checkpoint

最终可加载 checkpoint 已发布为 GitHub Release：

```text
https://github.com/flower418/retool/releases/tag/retool-rl-gs200-fused-hf
```

恢复说明和 checksum 见：

```text
docs/checkpoints.md
```

## 4. 手工对话评测与核心现象

`scripts/manual_dialog_eval.py` 使用 6 道可核验小题，对 base / SFT / RL 使用同一个
真实 sandbox prompt。它不是正式 benchmark，作用是观察模型是否真的学会把工具反馈
纳入推理。

| model | correct | accuracy | tool calls | avg tool calls |
| --- | ---: | ---: | ---: | ---: |
| base Qwen2.5-3B | 1 / 6 | 16.7% | 0 | 0.00 |
| SFT checkpoint | 3 / 6 | 50.0% | 5 | 0.83 |
| RL checkpoint | 4 / 6 | 66.7% | 7 | 1.17 |

逐题结果如下：

| question | expected | base | SFT | RL |
| --- | ---: | ---: | ---: | ---: |
| `sum_1_to_100` | 5050 | 5050 | 5050 | 5050 |
| `stairs_6_steps_123` | 24 | 20 | missing answer | 21 |
| `mod_17017_power` | 1 | 11 | 1 | 1 |
| `square_divisors` | 24 | 12 | 24 | 24 |
| `ordered_gcd_pairs` | 19 | 10 | 25 | 25 |
| `digit_sum_to_500` | 30 | 10 | 28 | 30 |

核心现象是 `digit_sum_to_500`：RL 模型先用组合计数得到错误中间结论 `372`，随后
主动用 sandbox 枚举，读取 `<interpreter>30</interpreter>`，最终把答案改成 30。

```text
How many positive integers n <= 500 have digit sum 7?
```

````text
So, the total is 372.

But wait, let me verify with code.

<code>
```python
count = 0
for n in range(1, 501):
    digit_sum = sum(int(d) for d in str(n))
    if digit_sum == 7:
        count += 1
print(count)
```
</code>
<interpreter>30</interpreter>

The code output is 30. So, the answer is 30.
````

这就是当前最核心的结论：RL 不只是让模型“会写代码块”，而是在强化“用外部执行结果
覆盖错误先验”的行为。它不是稳定的数学反思能力，因为 `ordered_gcd_pairs` 里模型
同样调用了 sandbox，但代码没有真正检查 `gcd(a, b) == 6`，所以仍错成 25。

## 5. 边界

1. 当前只有手工对话评测，还没有正式 AIME/MATH benchmark 分数。
2. sandbox 只能验证代码执行结果，不能保证代码表达了正确数学问题。
3. RL checkpoint 是 `global_step_200` 的阶段性模型，不是完整收敛后的最优模型。

## 6. 复现命令

### 6.1 安装

```bash
conda create -n retool python=3.11 -y
conda activate retool
pip install -r requirements.txt
```

### 6.2 生成 SFT 样例

```bash
set -a; source .env; set +a
python gen_data.py \
  --question "What is the sum of all integers from 1 to 100?" \
  --out data/sft/generated.jsonl \
  --model "$GEN_MODEL"
```

### 6.3 准备 RL 数据

```bash
python scripts/prepare_math_level_data.py
```

### 6.4 sandbox 推理

```bash
python scripts/infer_hf_with_sandbox.py \
  --model /path/to/hf-checkpoint \
  --question "Compute the sum of all integers from 1 to 100." \
  --max-new-tokens 4096 \
  --step-max-new-tokens 1024 \
  --max-tool-calls 4 \
  --max-model-calls 8
```

### 6.5 测试

```bash
python -m py_compile gen_data.py retool_sandbox/*.py scripts/*.py tests/*.py
python -m pytest tests
```
