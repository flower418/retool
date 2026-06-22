#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/root/autodl-tmp/retool}
VERL_DIR=${VERL_DIR:-/root/verl}
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-3B}
DATA_JSONL=${DATA_JSONL:-${REPO_DIR}/retool_sft_merged.jsonl}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/data/verl_sft}
RUN_ROOT=${RUN_ROOT:-${REPO_DIR}/runs}
PROJECT_NAME=${PROJECT_NAME:-retool-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-retool-qwen2_5-3b-sft-$(date +%Y%m%d-%H%M%S)}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MAX_LENGTH=${MAX_LENGTH:-8192}
VAL_SIZE=${VAL_SIZE:-100}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}
LR=${LR:-1e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-100}
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-1}
RUN_MODE=${RUN_MODE:-train}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate zero

mkdir -p "${DATA_DIR}" "${RUN_ROOT}/logs" "${RUN_ROOT}/checkpoints"

python "${REPO_DIR}/scripts/prepare_verl_sft_data.py" \
  --input "${DATA_JSONL}" \
  --model "${MODEL_PATH}" \
  --out-dir "${DATA_DIR}" \
  --max-length "${MAX_LENGTH}" \
  --val-size "${VAL_SIZE}"

SAVE_PATH="${RUN_ROOT}/checkpoints/${EXPERIMENT_NAME}"
LOG_PATH="${RUN_ROOT}/logs/${EXPERIMENT_NAME}.log"

if [[ "${RUN_MODE}" == "smoke" ]]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-smoke"
  SAVE_PATH="${RUN_ROOT}/checkpoints/${EXPERIMENT_NAME}"
  LOG_PATH="${RUN_ROOT}/logs/${EXPERIMENT_NAME}.log"
  TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-8}
  VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-4}
  TOTAL_STEPS=${TOTAL_STEPS:-1}
  LOGGER=${LOGGER:-'["console"]'}
else
  TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
  VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
  TOTAL_STEPS=${TOTAL_STEPS:-null}
  LOGGER=${LOGGER:-'["console","wandb"]'}
fi

export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB_NAME="${EXPERIMENT_NAME}"
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

cd "${VERL_DIR}"
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.messages_key=messages \
  data.max_length="${MAX_LENGTH}" \
  data.truncation=error \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
  data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  data.num_workers=2 \
  data.ignore_input_ids_mismatch=True \
  optim.lr="${LR}" \
  optim.lr_warmup_steps_ratio=0.03 \
  engine=fsdp \
  engine.model_dtype=bf16 \
  engine.dtype=bfloat16 \
  engine.param_offload=false \
  engine.optimizer_offload=false \
  engine.use_torch_compile=false \
  engine.ulysses_sequence_parallel_size=1 \
  model.path="${MODEL_PATH}" \
  +model.override_config.attn_implementation=sdpa \
  model.enable_gradient_checkpointing=true \
  model.enable_activation_offload=false \
  model.use_remove_padding=true \
  checkpoint.save_contents='["model"]' \
  checkpoint.load_contents='["model"]' \
  trainer.default_local_dir="${SAVE_PATH}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger="${LOGGER}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.max_ckpt_to_keep="${MAX_CKPT_TO_KEEP}" \
  trainer.resume_mode=disable \
  trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
  2>&1 | tee "${LOG_PATH}"
