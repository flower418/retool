#!/usr/bin/env bash
set -euo pipefail

cd "${VERL_ROOT:-/root/verl}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate zero

RETOOL_REPO_DIR="${RETOOL_REPO_DIR:-/root/autodl-tmp/retool}"
export PYTHONPATH="${RETOOL_REPO_DIR}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

MODE="${MODE:-full}"
PROJECT_NAME="${PROJECT_NAME:-retool-dapo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-retool-dapo-smoke-${MODE}}"
LOGGER="${LOGGER:-[\"console\"]}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/retool/runs/merged/retool-qwen2_5-3b-sft-epoch3-global_step_941-hf}"
TRAIN_FILE="${TRAIN_FILE:-/root/autodl-tmp/retool/data/dapo/smoke/train_8.parquet}"
VAL_FILE="${VAL_FILE:-/root/autodl-tmp/retool/data/dapo/smoke/val_4.parquet}"
CKPTS_DIR="${CKPTS_DIR:-/root/autodl-tmp/retool/runs/dapo_smoke/${EXPERIMENT_NAME}}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
ROLLOUT_N="${ROLLOUT_N:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-$((TRAIN_BATCH_SIZE * ROLLOUT_N))}"
REWARD_WORKERS="${REWARD_WORKERS:-${AGENT_LOOP_WORKERS}}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-${AGENT_LOOP_WORKERS}}"
VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-}"
VLLM_LAYERED_SUMMON="${VLLM_LAYERED_SUMMON:-}"
UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB:-}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
TEST_FREQ="${TEST_FREQ:--1}"
SAVE_FREQ="${SAVE_FREQ:--1}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
MAX_CRITIC_CKPT_TO_KEEP="${MAX_CRITIC_CKPT_TO_KEEP:-1}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-0}"
USE_RETOOL_SANDBOX="${USE_RETOOL_SANDBOX:-False}"
RETOOL_AGENT_LOOP_CONFIG="${RETOOL_AGENT_LOOP_CONFIG:-${RETOOL_REPO_DIR}/configs/retool_sandbox_agent.yaml}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

RESUME_ARGS=(trainer.resume_mode="${RESUME_MODE}")
if [[ -n "${RESUME_FROM_PATH}" ]]; then
  RESUME_ARGS+=(trainer.resume_from_path="${RESUME_FROM_PATH}")
fi

case "${TRAIN_FILE} ${VAL_FILE}" in
  *[Aa][Ii][Mm][Ee]*)
    echo "Refusing to use AIME for DAPO train/val. Keep AIME only for bench." >&2
    exit 3
    ;;
esac

for DATA_FILE in "${TRAIN_FILE}" "${VAL_FILE}"; do
  if [[ ! -f "${DATA_FILE}" ]]; then
    echo "Missing data file: ${DATA_FILE}" >&2
    exit 4
  fi
done

LORA_ARGS=()
if [[ "${MODE}" == "lora" ]]; then
  LORA_ARGS+=(actor_rollout_ref.model.lora_rank="${LORA_RANK:-32}")
  LORA_ARGS+=(actor_rollout_ref.model.use_shm="${MODEL_USE_SHM:-True}")
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
  VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-True}"
  VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-safetensors}"
  VLLM_LAYERED_SUMMON="${VLLM_LAYERED_SUMMON:-False}"
  UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB:-256}"
elif [[ "${MODE}" != "full" ]]; then
  echo "MODE must be full or lora, got: ${MODE}" >&2
  exit 2
else
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
  VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-False}"
  VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-dummy}"
  VLLM_LAYERED_SUMMON="${VLLM_LAYERED_SUMMON:-False}"
  UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB:-2048}"
fi

SANDBOX_ARGS=()
if [[ "${USE_RETOOL_SANDBOX}" == "True" || "${USE_RETOOL_SANDBOX}" == "true" || "${USE_RETOOL_SANDBOX}" == "1" ]]; then
  if [[ ! -f "${RETOOL_AGENT_LOOP_CONFIG}" ]]; then
    echo "Missing ReTool sandbox agent config: ${RETOOL_AGENT_LOOP_CONFIG}" >&2
    exit 5
  fi
  SANDBOX_ARGS+=("actor_rollout_ref.rollout.agent.default_agent_loop=retool_sandbox_agent")
  SANDBOX_ARGS+=("actor_rollout_ref.rollout.agent.agent_loop_config_path=${RETOOL_AGENT_LOOP_CONFIG}")
fi

python3 -m verl.trainer.main_ppo "$@" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.prompt_key=prompt \
  data.truncation=left \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_WORKERS}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager="${VLLM_ENFORCE_EAGER}" \
  actor_rollout_ref.rollout.load_format="${VLLM_LOAD_FORMAT}" \
  actor_rollout_ref.rollout.layered_summon="${VLLM_LAYERED_SUMMON}" \
  actor_rollout_ref.rollout.max_model_len="${VLLM_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${VLLM_MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MB}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.num_workers="${REWARD_WORKERS}" \
  reward_model.reward_manager=dapo \
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable="${ENABLE_OVERLONG_BUFFER:-False}" \
  +reward_model.reward_kwargs.overlong_buffer_cfg.len="${OVERLONG_BUFFER_LEN:-128}" \
  +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor="${OVERLONG_PENALTY_FACTOR:-1.0}" \
  +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
  +reward_model.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}" \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.max_critic_ckpt_to_keep="${MAX_CRITIC_CKPT_TO_KEEP}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.default_local_dir="${CKPTS_DIR}" \
  "${RESUME_ARGS[@]}" \
  trainer.log_val_generations="${LOG_VAL_GENERATIONS}" \
  "${SANDBOX_ARGS[@]}" \
  "${LORA_ARGS[@]}"
