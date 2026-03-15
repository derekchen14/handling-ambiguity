#!/usr/bin/env bash
# Test full fine-tuning SFT on flow stage (hugo+dana domains) using eval_set
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m training.train_nlu \
    --mode sft --stages flow --domain hugo,dana \
    --data_path datasets/hugo/eval_set.json,datasets/dana/eval_set.json \
    --val_ratio 0.2 \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --sft_epochs 5 --sft_lr 2e-5 --batch_size 4 \
    --eval_every 1 --eval_temperature 1.0 \
    --model_save_path ./test_full_model \
    --wandb_project NLU-Training --wandb_name test_sft_full_flow_hugo_dana
