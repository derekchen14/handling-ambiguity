#!/usr/bin/env bash
# Distillation SFT: LoRA on Qwen3-4B from ensemble pseudo-labels (Hugo + Dana)
set -euo pipefail

cd "$(dirname "$0")/../.."

export WANDB_MODE="${WANDB_MODE:-online}"

uv run --extra training accelerate launch --num_processes 8 --multi_gpu -m training.train_nlu \
    --mode sft --stages flow \
    --domain hugo,dana \
    --data_path datasets/hugo/eval_set.json,datasets/dana/eval_set.json \
    --ensemble_results_path training/distill_data/hugo_10v-1.jsonl,training/distill_data/dana_10v-1.jsonl \
    --confidence_threshold 0.7 \
    --val_ratio 0.2 \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --sft_epochs 15 --sft_lr 2e-4 --batch_size 4 \
    --use_lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
    --eval_every 1 --eval_temperature 1.0 \
    --model_save_path ./distill_lora_model \
    --wandb_project NLU-Training --wandb_name distill_sft_lora_flow_hugo_dana
