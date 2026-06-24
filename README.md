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

## 3. 训练结果

### 3.1 SFT loss

SFT 训练共形成 3 个连续阶段的 merged checkpoint。最终阶段日志显示验证 loss
持续下降：

| 阶段 | 训练步 | final `val/loss` |
| --- | ---: | ---: |
| epoch 1 | global_step 941 | 0.5525964499 |
| epoch 2 | global_step 941 | 0.5473101735 |
| epoch 3 | global_step 941 | 0.5376862288 |

epoch 3 内部验证 loss 也从 step 100 的 `0.54754` 降到 step 941 的
`0.53769`。这个结果说明模型确实学到了 SFT 数据分布和输出格式，但它不等价于
数学能力已经可靠提升。

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

训练日志中能确认几件事：

1. reward 不是常数。若干保存点上 `critic/score/min=-1.0`、
   `critic/score/max=1.0`，说明同组采样里同时存在正负样本。
2. agent loop 确实在运行。后期日志中 `num_turns/mean` 约为 2.7，
   `tool_calls` 非零，说明 rollout 不是纯文本 generation。
3. 训练没有跑满原计划 1000 step，而是在 `global_step_200` 处保存并 merge
   了最终 checkpoint。这个 checkpoint 是当前公开产物，不应把它描述成完整
   收敛后的模型。

部分日志点：

| step | `critic/score/mean` | `critic/score/max` | `critic/score/min` | `response_length/mean` | `num_turns/mean` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | -0.625 | 1.0 | -1.0 | 802.75 | 2.8125 |
| 100 | -0.500 | 1.0 | -1.0 | 1122.38 | 2.6250 |
| 150 | -0.250 | 1.0 | -1.0 | 703.69 | 2.9375 |
| 199 | -0.500 | 1.0 | -1.0 | 817.63 | 2.6875 |

这些指标只能说明训练链路、reward、sandbox 和 checkpoint 保存路径打通；不能直接
说明模型在 held-out benchmark 上变强。

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

为了检查训练是否真的改变了模型和工具的关系，使用
`scripts/manual_dialog_eval.py` 跑了一组小规模手工对话评测。这个评测不是正式
benchmark，只包含 6 道可人工核验的数学题，目的是比较 base / SFT / RL 三个
checkpoint 在同一 sandbox prompt 下的行为差异。

评测使用同一个答案协议和同一个真实 sandbox loop：模型生成 `<code>` 代码块后，
Python sandbox 执行代码，并把 stdout 写回 `<interpreter>...</interpreter>`。
最终仍以最后一行 `Answer: ...` 作为判分对象。

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

这组结果最重要的现象不是 6 题上的小样本准确率，而是 RL checkpoint 出现了更明确
的“工具反馈校正”行为。以 `digit_sum_to_500` 为例，题目是：

```text
How many positive integers n <= 500 have digit sum 7?
```

RL 模型先用组合计数得到了错误中间结论 `372`，随后主动转向 sandbox 枚举验证：

````text
But wait, let me verify with code. I'll use the Python sandbox to compute the
digit sums for numbers up to 500 and count how many have a sum of 7.

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

Answer: 30
````

这个案例体现了 ReTool RL 的本质目标：模型不只是学习“输出一个看起来像代码的
格式”，而是学习在推理过程中把外部执行结果当作更高优先级的证据。换句话说，RL
阶段强化的不是纯粹的内在反思能力，而是“生成可执行检验 -> 读取环境反馈 ->
覆盖错误先验 -> 写出最终答案”的行为模式。对于枚举、模运算、代数化简这类可以
直接计算的问题，这种模式能把一部分本来会错的推理拉回正确答案。

但这不等价于模型已经稳定具备数学反思能力。`ordered_gcd_pairs` 是一个反例：
RL 模型也调用了 sandbox，但它写出的代码只统计了 6 的倍数对，没有真正检查
`gcd(a, b) == 6`，因此 sandbox 只是忠实执行了错误问题建模，最终仍错成 25。
这说明 sandbox 能校验代码执行结果，却不能自动校验“代码是否表达了正确数学问题”。

因此当前结论应当谨慎表述为：

1. base 基本不会遵守工具调用协议；
2. SFT 能让模型学到部分 ReTool 格式和工具调用习惯；
3. RL 进一步提高了工具调用率，并出现了工具反馈纠错的行为；
4. 但模型仍会写出语义错误的代码，或在简单递推题上坚持错误先验。

当前 README 仍不报告 AIME/MATH benchmark 分数。仓库里有
`scripts/eval_aime2024_models.py`，但正式 benchmark 需要重新跑完并检查输出质量后
再写入结果。

## 5. 不足与风险分析

### 5.1 结果证据不足

目前最完整的证据是训练过程日志和少量人工对话。SFT loss 下降不代表推理能力提升；
RL reward 非常依赖最终答案解析，也不能直接替代独立 benchmark。

下一步应该补：

1. 固定 held-out benchmark，例如 AIME 2024 或 MATH test 子集；
2. base / SFT / RL 三列对比；
3. 每题保存原始输出、sandbox tool calls、最终答案、判分理由；
4. 同时统计格式成功率、工具调用率、最终答案正确率。

### 5.2 Reward 仍可能漏判或误判

当前 reward 已做最终答案 canonicalization，但数学表达很复杂。历史 rollout audit
已经暴露过这类问题：

1. `2/3`、`\dfrac23`、`\frac{2}{3}` 等等价表达；
2. 多个 `Answer:` 行；
3. 答案行后附带多余解释；
4. tool output 正确但最终答案缺失；
5. 单位、区间、集合、根式、模数答案等格式。

因此 reward 需要和 rollout dump 一起审计，不能只看 W&B 上的平均 reward。

### 5.3 工具使用不等于问题建模正确

手工评测显示 RL checkpoint 的工具调用率高于 SFT 和 base，但工具调用本身并不
保证正确。当前主要失败模式有三类：

1. 模型不调用工具或没有输出规范 `Answer:` 行，例如 SFT 在楼梯题上长文本发散；
2. 模型调用工具，但代码表达的是错误数学模型，例如 gcd pair 题只数了 6 的倍数对；
3. 模型把 sandbox 输出当作 ground truth，但如果代码本身漏了约束，最终答案仍会错。

这说明下一步优化不能只奖励“调用了工具”，还需要让模型学会把自然语言条件完整
翻译成可执行检查。更合理的训练信号可能包括：执行代码中是否覆盖原题约束、是否
做了 brute-force 交叉验证、是否在最终答案前比较了手推结果和工具结果。

### 5.4 当前 RL 不是完整收敛实验

最终公开 checkpoint 来自 `global_step_200`，而不是计划中的完整 1000 step。
它适合作为“链路打通后的阶段性模型”，不应被描述成最终最优模型。

## 6. 需要补充的材料

为了把 README 从“工程报告”升级成更像论文的结果页，建议补充以下材料：

1. W&B 或日志截图：SFT `val/loss` 曲线、RL `critic/score/mean` 曲线、
   response length、tool calls/turns。
2. 人工对话样例：同一题下 base / SFT / RL 的原始输出，最好包含模型成功调用
   sandbox 和失败案例各 1-2 个。
3. benchmark 输出：至少一个固定小集合，例如 AIME 2024 30 题，保留逐题 JSONL。
4. 错误分类表：`answer_mismatch`、`missing_final_answer`、sandbox error、
   truncation、reward parser mismatch。

如果有训练图像，建议放在：

```text
docs/assets/
```

如果有人工对话测评，建议放在：

```text
docs/eval_conversations/
```

README 只引用筛选后的代表性图和案例，原始材料保留在 docs 下。

## 7. 复现命令

### 7.1 安装

```bash
conda create -n retool python=3.11 -y
conda activate retool
pip install -r requirements.txt
```

### 7.2 生成 SFT 样例

```bash
set -a; source .env; set +a
python gen_data.py \
  --question "What is the sum of all integers from 1 to 100?" \
  --out data/sft/generated.jsonl \
  --model "$GEN_MODEL"
```

### 7.3 准备 RL 数据

```bash
python scripts/prepare_math_level_data.py
```

### 7.4 sandbox 推理

```bash
python scripts/infer_hf_with_sandbox.py \
  --model /path/to/hf-checkpoint \
  --question "Compute the sum of all integers from 1 to 100." \
  --max-new-tokens 4096 \
  --step-max-new-tokens 1024 \
  --max-tool-calls 4 \
  --max-model-calls 8
```

### 7.5 测试

```bash
python -m py_compile gen_data.py retool_sandbox/*.py scripts/*.py tests/*.py
python -m pytest tests
```
