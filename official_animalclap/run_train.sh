#!/usr/bin/env bash
# AnimalCLAP training launch script (8 GPU, single node)
#
# Before running:
#   1. Pre-download audio for the train split using download_url in train.csv
#   2. Set WANDB_API_KEY in your environment: export WANDB_API_KEY=...
#   3. Adjust paths below

set -euo pipefail

pkill -f torchrun    || true
pkill -f train.py    || true

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

# To reproduce the ablation (randomized taxonomy order):
# export TEXT_MODE=tax_perm_order

torchrun --nnodes=1 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 \
  --max-restarts=0 train.py \
    --train_csv  /path/to/train.csv \
    --audio_dir  /path/to/audio \
    --save_dir   ./models \
    --run_name   animalclap_exp1 \
    --epochs     50 \
    --batch_size 32 \
    --lr         1e-4 \
    --per_species 30
