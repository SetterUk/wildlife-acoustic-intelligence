#!/usr/bin/env bash
# AnimalCLAP Traits Classifier training launch script (8 GPU, single node)
#
# Runs multiple trait columns sequentially.
# Before running:
#   1. Pre-download audio for the train split using download_url in train.csv
#   2. Set WANDB_API_KEY in your environment: export WANDB_API_KEY=...
#   3. Adjust paths and TASKS below

set -euo pipefail

pkill -f torchrun          || true
pkill -f train_traits.py   || true

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# W&B (set your API key in the environment, not here)
export WANDB_MODE=online
export WANDB_ENTITY=your_entity
export WANDB_PROJECT=animalclap

TRAIN_CSV=/path/to/train.csv
TEST_CSV=/path/to/test.csv
TRAITS_CSV=/path/to/species_traits.csv
AUDIO_DIR=/path/to/audio
ENCODER_CKPT=/path/to/animalclap_final.pth
SAVE_DIR=./models/traits

# Tasks: "column_name:task_type:positive_value"
# task_type: multiclass or binary
# positive_value: only used for binary tasks
TASKS=(
  "diet_type:multiclass:"
  "activity_pattern:multiclass:"
  "locomotion_mode:multiclass:"
  "social_behavior:multiclass:"
  "migratory:binary:True"
)

for task in "${TASKS[@]}"; do
  IFS=":" read -r COL TYPE POS <<< "${task}"
  echo "==== Running: col=${COL} type=${TYPE} ===="

  EXTRA_ARGS=""
  if [ "${TYPE}" = "binary" ]; then
    EXTRA_ARGS="--positive_value ${POS}"
  fi

  torchrun --nnodes=1 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 \
    --max-restarts=0 train_traits.py \
      --train_csv     ${TRAIN_CSV} \
      --test_csv      ${TEST_CSV} \
      --traits_csv    ${TRAITS_CSV} \
      --audio_dir     ${AUDIO_DIR} \
      --encoder_ckpt  ${ENCODER_CKPT} \
      --target_col    ${COL} \
      --task_type     ${TYPE} \
      --save_dir      ${SAVE_DIR} \
      --epochs        5 \
      --batch_size    128 \
      --freeze_encoder \
      ${EXTRA_ARGS}

  echo "==== Done: ${COL} ===="
done

echo "All traits training complete."
