#!/usr/bin/env bash
# Test LoRA SFT on flow stage (hugo domain) using eval_set
set -euo pipefail

cd "$(dirname "$0")/../.."

export WANDB_MODE="${WANDB_MODE:-disabled}"

uv run --extra training python -m training.train_nlu \
    --mode sft --stages flow --domain hugo \
    --data_path datasets/hugo/eval_set.json \
    --val_ratio 0.2 \
    --model_name Qwen/Qwen3.5-4B \
    --sft_epochs 15 --sft_lr 2e-4 --batch_size 4 \
    --use_lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
    --eval_every 1 --eval_temperature 1.0 \
    --model_save_path ./test_lora_model \
    --wandb_project NLU-Training --wandb_name test_sft_lora_flow_hugo
