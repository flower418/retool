# ReTool：面向代码增强数学推理的 SFT + Sandbox RL 训练链路

## 摘要

本项目实现了一条 ReTool 风格的数学推理训练链路：先用离线 SFT 数据教模型写
Python、读取执行结果并输出最终答案，再在 veRL 的 RL rollout 中接入真实 Python
sandbox，让模型在生成过程中真正执行代码，并用最终答案 reward 做强化学习。

一句话概括：这个项目不是只让模型“长得像会用工具”，而是把“生成代码 -> 执行代码
-> 读取环境反馈 -> 修正最终答案”做成一条可训练、可审计、可发布的链路。

最终公开产物是融合后的 Hugging Face checkpoint：

```text
retool-math-l1-l3-dapo-lora-r64-global_step_200-fused-hf
```

当前结果来自训练日志、手工对话评测和 checkpoint 产物；正式 AIME/MATH benchmark
还没有完成，因此这里不报告 benchmark 分数。

## 1. 项目主线

整条链路分成 6 个阶段：

| 阶段 | 做什么 | 关键产物 |
| --- | --- | --- |
| 数据构造 | 准备 SFT trace 数据和 RL prompt 数据 | `data/sft/train.jsonl`, `data/rl/math_l1_l3/` |
| SFT | 让 Qwen2.5-3B 学会 ReTool 输出格式 | epoch-3 merged SFT checkpoint |
| Sandbox | 在推理/RL 中真实执行 Python 代码 | `retool_sandbox/`, `infer_hf_with_sandbox.py` |
| RL | 用 veRL agent loop 做在线工具调用与最终答案 reward | global_step_200 RL checkpoint |
| 结果观察 | 对比 base / SFT / RL 的工具使用和答案变化 | manual dialog eval JSONL |
| 发布 | 将最终 fused HF checkpoint 放到 GitHub Release | release assets + `docs/checkpoints.md` |

面向面试或项目讲解时，可以围绕三个问题展开：

1. 数据上，SFT trace 和 RL prompt 为什么要分开；
2. 系统上，如何保证 `<interpreter>` 不是模型幻觉，而是真实执行结果；
3. 训练上，RL 学到的核心不是“反思话术”，而是“用工具反馈覆盖错误先验”。

## 2. 数据阶段：SFT trace 和 RL prompt 是两种数据

### 2.1 SFT 数据：离线生成完整 ReTool 轨迹

SFT 数据位于：

```text
data/sft/train.jsonl
```

每行是 Hugging Face chat messages 格式：

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

当前训练文件包含 2002 行，主体来自 `JoeYing/ReTool-SFT`，并合入少量本项目生成样例。
外部 ReTool-SFT 主体保留了原始 trace 形式，部分样本使用
`<answer>\boxed{...}</answer>`；本项目后续生成和 RL 推理统一收敛到：

```text
Answer: <final answer>
```

本项目自己的 SFT 生成器是 `gen_data.py`。它接受单题或 question-only JSONL，把问题
填入 `prompts/solve_with_code.txt`，调用 OpenAI-compatible teacher model 生成
assistant 解题轨迹。

关键点是：teacher 写出的 `<interpreter>` 不直接可信。`gen_data.py` 会重新执行所有
Python code block，并把真实 stdout materialize 回样本：

```text
question
  -> teacher generates assistant trace
  -> extract <code>```python ... ```</code>
  -> local Python executes code
  -> overwrite / fill <interpreter>stdout</interpreter>
  -> validate final answer protocol
  -> write SFT JSONL
```

校验规则包括：

1. 必须有 `<code>` 和 `<interpreter>`；
2. code block 与 interpreter block 必须一一对应；
3. 新增样本必须只有一个最终 `Answer:` 行，并且它是最后一行；
4. 对整数答案，最后的 `Answer:` 要和最后一个 interpreter 输出一致；
5. 失败样本不进入训练集，只写入 `.meta.jsonl` 方便排查。

然后 `scripts/prepare_verl_sft_data.py` 会把 messages JSONL 转成 veRL SFT parquet：
它会套目标模型 tokenizer 的 chat template，过滤超长样本，再按固定 seed 切 train/val。

**踩坑与处理：**

| 坑 | 处理 |
| --- | --- |
| teacher 会伪造 `<interpreter>` 输出 | 重新执行代码，用真实 stdout 覆盖 |
| 外部数据和新增数据答案协议不完全一致 | 文档里明确 legacy `<answer>`，新增和 RL 统一 `Answer:` |
| 临时 smoke 数据容易混进训练集 | 删除只有 2 行的临时样例，只保留整理后的 `data/sft/train.jsonl` |
| SFT 数据和 RL 数据容易混淆 | SFT 有 assistant trace；RL 只有 prompt + ground truth |

### 2.2 RL 数据：只保留题目和标准答案

RL 数据位于：

```text
data/rl/math_l1_l3/
```

它来自 `EleutherAI/hendrycks_math` train split。`scripts/prepare_math_level_data.py`
遍历 7 个 subject，过滤 level 1-3，并从原始 solution 中提取最后一个
`\boxed{...}` 或 `\fbox{...}` 作为 ground truth。

每个样本转成 veRL/DAPO 兼容格式：

```json
{
  "data_source": "math_dapo",
  "prompt": [{"role": "user", "content": "..."}],
  "ability": "MATH",
  "reward_model": {"ground_truth": "...", "style": "rule-lighteval/MATH_v2"},
  "extra_info": {"source": "...", "subject": "...", "level": 3}
}
```

这里刻意不把 ReTool 代码协议写进 parquet prompt。parquet 只描述“要解什么题”和
“标准答案是什么”；真正的工具协议由 `retool_sandbox/verl_agent_loop.py` 在 rollout
时动态注入。这样数据集不会和某一个 agent 实现绑定，后续从 DAPO 换成 PPO/GRPO 也
不用重做数据。

当前 RL 数据规模：

| 字段 | 值 |
| --- | --- |
| dataset | `EleutherAI/hendrycks_math` |
| levels | 1-3 |
| unique rows before split | 3504 |
| val rows | 128 |
| train rows after repeat | 33760 |
| repeat | 10 |
| seed | 2026 |

按 level 分布：

| level | rows |
| ---: | ---: |
| 1 | 564 |
| 2 | 1348 |
| 3 | 1592 |

按 subject 分布：

| subject | rows |
| --- | ---: |
| prealgebra | 706 |
| algebra | 910 |
| number theory | 369 |
| counting/probability | 329 |
| geometry | 270 |
| intermediate algebra | 526 |
| precalculus | 394 |

**踩坑与处理：**

| 坑 | 处理 |
| --- | --- |
| 把 RL 数据误以为也需要 response | RL 是 online rollout，数据只要 prompt 和 ground truth |
| 把当前 MATH L1-3 / DAPO 写死成框架假设 | 数据、算法、reward、tool loop 分成独立 slot |
| ground truth 解析依赖 boxed answer | 只保留能抽取 boxed/fbox 答案的样本 |

## 3. SFT 阶段：先学格式，不急着证明能力

SFT 基座是 Qwen2.5-3B。这个阶段的目标不是直接把 benchmark 做高，而是让模型稳定学会
ReTool 交互协议：

1. 面对数学题生成自包含 Python code block；
2. 在代码后读取 `<interpreter>` 输出；
3. 最后给出明确最终答案；
4. 不在最终答案后继续输出无关文本。

SFT 后 merged HF checkpoint：

```text
/root/autodl-tmp/retool/runs/merged/retool-qwen2_5-3b-sft-epoch3-global_step_941-hf
```

SFT 验证 loss：

| 阶段 | 训练步 | final `val/loss` |
| --- | ---: | ---: |
| epoch 1 | global_step 941 | 0.5525964499 |
| epoch 2 | global_step 941 | 0.5473101735 |
| epoch 3 | global_step 941 | 0.5376862288 |

![SFT validation loss](docs/assets/sft_val_loss.png)

**踩坑与处理：**

| 坑 | 处理 |
| --- | --- |
| `val/loss` 下降不等于工具推理真的可靠 | 后面必须做真实 sandbox 推理和原始输出对比 |
| veRL/FSDP 原始 checkpoint 不能直接当 HF 模型用 | 用 `verl.model_merger` merge 成 Hugging Face checkpoint |
| 普通 `infer_hf.py` 输出的 `<interpreter>` 可能只是模型幻觉 | 只把 `infer_hf_with_sandbox.py` 和 veRL agent loop 当作真实工具路径 |

## 4. Sandbox 阶段：把“看起来会用工具”变成真实执行

这是项目最关键的工程边界。普通 generation 可以让模型吐出：

```text
<interpreter>5050</interpreter>
```

但这不代表代码真的执行过。真实 ReTool loop 必须做到：

```text
model generates until </code>
  -> AsyncPythonSandboxPool executes Python
  -> stdout is appended as <interpreter>...</interpreter>
  -> model continues generation with that result
```

本项目有两条真实 sandbox 路径：

| 场景 | 入口 |
| --- | --- |
| 单题/手工推理 | `scripts/infer_hf_with_sandbox.py` |
| veRL RL rollout | `retool_sandbox/verl_agent_loop.py` |

`AsyncPythonSandboxPool` 以异步 worker pool 执行 Python 片段，并限制 timeout、输出长度、
内存和文件大小。veRL agent loop 会在 rollout 中注入 ReTool 协议，限制单步生成长度、
最大模型调用次数和最大工具调用次数，并把 tool results dump 出来做审计。

**踩坑与处理：**

| 坑 | 处理 |
| --- | --- |
| 一开始只有普通 generation，没有真正 sandbox | 明确区分 `infer_hf.py` 和 `infer_hf_with_sandbox.py` |
| `gen_data.py` 是离线 SFT 数据生成，不是 RL sandbox | 在线 sandbox 独立放在 `retool_sandbox/` 和 veRL agent loop |
| 长 rollout 可能在最终 `Answer:` 前截断 | 记录 `stop_reason`、`max_tool_calls`、`missing_final_answer` |
| 只看进程/W&B 初始化会误判“训练已开始” | 需要看 Ray/vLLM 日志、rollout dump 和真实 step 指标 |

## 5. RL 阶段：用最终答案 reward 训练工具反馈行为

RL 从 epoch-3 SFT checkpoint 开始，使用 veRL 的 DAPO/GRPO 风格训练。最终保留 run：

```text
retool-math-l1-l3-dapo-lora-r64-sandbox-b2-n8-r2048-s50-1000-fixreward-20260624_023756
```

关键配置：

| 字段 | 值 |
| --- | --- |
| base model | epoch-3 SFT merged HF checkpoint |
| mode | LoRA |
| LoRA rank | 64 |
| train batch size | 2 |
| rollout n | 8 |
| max response length | 2048 |
| save freq | 50 |
| final saved step | global_step 200 |

reward 以最后的 `Answer:` 为核心判分对象。`retool_sandbox/math_reward.py` 在
`math_dapo` 基础上做 final-answer extraction、canonicalization 和元数据返回，用来区分：

1. 数学答案真的错；
2. 答案对但格式没被提取到；
3. 工具输出对但最终 `Answer:` 缺失；
4. rollout 太长或工具调用耗尽导致没有最终答案。

RL 训练信号：

| step | `critic/score/mean` | `critic/score/max` | `critic/score/min` | `response_length/mean` | `num_turns/mean` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | -0.625 | 1.0 | -1.0 | 802.75 | 2.8125 |
| 100 | -0.500 | 1.0 | -1.0 | 1122.38 | 2.6250 |
| 150 | -0.250 | 1.0 | -1.0 | 703.69 | 2.9375 |
| 199 | -0.500 | 1.0 | -1.0 | 817.63 | 2.6875 |

![RL reward mean](docs/assets/rl_rewards_mean.png)

![RL response length mean](docs/assets/rl_response_length_mean.png)

![RL number of turns mean](docs/assets/rl_num_turns_mean.png)

这些曲线说明训练、reward、agent loop 和 checkpoint 保存链路是闭合的；它们不是
正式 benchmark 分数。

**踩坑与处理：**

| 坑 | 处理 |
| --- | --- |
| reward 看起来波动大，不知道是模型错还是解析错 | 用 rollout dump 重打分，分离 `answer_mismatch` 和 `missing_final_answer` |
| 分数均值不能说明 reward 一定正确 | 检查正负样本原文、tool results、最终答案提取字段 |
| `2/3`、`\dfrac23`、重复 `Answer:`、尾随解释会影响判分 | 在 reward 中做 canonicalization，并保留 reason / match_type |
| W&B 初始化不等于训练有效推进 | 需要看到真实 rollout、非零 turns/tool calls、step 指标 |
| checkpoint 和磁盘空间压力大 | 控制 `save_freq`、`max_ckpt_to_keep`，只保留关键 step |

## 6. 结果观察：RL 学到的是“用反馈纠错”

为了观察模型是否真的把工具反馈纳入推理，我用 `scripts/manual_dialog_eval.py` 跑了
6 道可核验小题。base / SFT / RL 使用同一个真实 sandbox prompt。

| model | correct | accuracy | tool calls | avg tool calls |
| --- | ---: | ---: | ---: | ---: |
| base Qwen2.5-3B | 1 / 6 | 16.7% | 0 | 0.00 |
| SFT checkpoint | 3 / 6 | 50.0% | 5 | 0.83 |
| RL checkpoint | 4 / 6 | 66.7% | 7 | 1.17 |

逐题结果：

| question | expected | base | SFT | RL |
| --- | ---: | ---: | ---: | ---: |
| `sum_1_to_100` | 5050 | 5050 | 5050 | 5050 |
| `stairs_6_steps_123` | 24 | 20 | missing answer | 21 |
| `mod_17017_power` | 1 | 11 | 1 | 1 |
| `square_divisors` | 24 | 12 | 24 | 24 |
| `ordered_gcd_pairs` | 19 | 10 | 25 | 25 |
| `digit_sum_to_500` | 30 | 10 | 28 | 30 |

最核心的现象是 `digit_sum_to_500`。RL 模型先用组合计数得到错误中间结论 `372`，
随后主动调用 sandbox 枚举，读到 `<interpreter>30</interpreter>`，最终把答案改成 30：

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

这说明 RL 不只是让模型“会写代码块”，而是在强化“用外部执行结果覆盖错误先验”的行为。

但这个能力有边界。`ordered_gcd_pairs` 里模型也调用了 sandbox，但代码没有真正检查
`gcd(a, b) == 6`，所以 sandbox 忠实执行了错误建模，最终仍错成 25。换句话说，
sandbox 能验证代码执行结果，不能保证代码表达了正确数学问题。

## 7. 发布与边界

最终 checkpoint 已发布为 GitHub Release：

```text
https://github.com/flower418/retool/releases/tag/retool-rl-gs200-fused-hf
```

恢复说明和 checksum 见：

```text
docs/checkpoints.md
```

当前边界：

1. 还没有正式 AIME/MATH benchmark；
2. reward 依赖最终答案解析，仍可能错判或漏判；
3. sandbox 只能执行代码，不能自动判断代码是否完整表达原题约束；
4. 公开 RL checkpoint 是 `global_step_200` 的阶段性模型，不是完整收敛最优模型。

## 8. 复现入口

正文不展开命令细节，只列入口文件：

| 目标 | 入口 |
| --- | --- |
| 生成 ReTool SFT 样例 | `gen_data.py` |
| 将 SFT messages 转为 veRL parquet | `scripts/prepare_verl_sft_data.py` |
| 准备 MATH level 1-3 RL 数据 | `scripts/prepare_math_level_data.py` |
| SFT 训练 | `scripts/run_verl_sft.sh` |
| Sandbox RL 训练 | `scripts/run_dapo_smoke.sh` |
| 单题真实 sandbox 推理 | `scripts/infer_hf_with_sandbox.py` |
| 三模型手工对话评测 | `scripts/manual_dialog_eval.py` |
| checkpoint 恢复说明 | `docs/checkpoints.md` |
