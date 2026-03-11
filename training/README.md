# NLU Training Pipeline

Training pipeline for NLU models supporting **Supervised Fine-Tuning (SFT)** and **Reinforcement Learning (PPO)** across modular pipeline stages.

## Directory Structure

```
training/
├── train_nlu.py          # CLI entry point — dispatches SFT or RL training
├── stages.py             # Pipeline stage abstraction + registry (intent, flow, tool_selection, param_extraction)
├── rewards.py            # Reward callback factories for RL scoring
├── rollouts.py           # Trajectory generation via sglang server
└── utils/
    ├── trainer.py        # PPOTrainer class + run_sft helper
    ├── sft_data.py       # SFT example generation from gold labels or ensemble results
    ├── dataset.py        # RLTrainingDataset, NLUSFTDataset, GroupBatchSampler
    ├── server.py         # sglang server launch/cleanup
    ├── losses.py         # PPO loss, KL divergence, TD returns
    ├── checkpoints.py    # Model checkpoint save/load
    └── trajectories.py   # Trajectory cleaning utilities
```

## Setup

```bash
uv sync --extra training
```

This installs: `torch`, `transformers`, `accelerate`, `wandb`, `sglang`, `nest-asyncio`.

## Stages

Four base stages are registered in `STAGE_REGISTRY`:

| Stage | Description |
|-------|-------------|
| `flow` | Flow detection |
| `intent` | Intent classification |
| `tool_selection` | Tool selection (native tool calling) |
| `param_extraction` | Parameter extraction for tools |

Stages can be **composed** with `+` notation (e.g., `tool_selection+param_extraction`). Composed stages combine rewards using configurable weights via `--stage_weights`.

## Usage

### SFT — from gold labels

```bash
python -m training.train_nlu --mode sft --stages flow --domain hugo \
    --model_name Qwen/Qwen3-0.6B \
    --data_path datasets/hugo/eval_set.json
```

### SFT — from ensemble results with confidence filtering

```bash
python -m training.train_nlu --mode sft --stages flow --domain hugo \
    --model_name Qwen/Qwen3-0.6B \
    --data_path datasets/hugo/eval_set.json \
    --ensemble_results_path results/exp1b/hugo_3v-9_seed1.jsonl \
    --confidence_threshold 0.7
```

### RL — single stage

```bash
python -m training.train_nlu --mode rl --stages flow --domain hugo \
    --model_name Qwen/Qwen3-0.6B \
    --data_path datasets/hugo/eval_set.json
```

### RL — composed stages with weighted rewards

```bash
python -m training.train_nlu --mode rl \
    --stages tool_selection+param_extraction --stage_weights 0.6,0.4 \
    --domain hugo --model_name Qwen/Qwen3-0.6B \
    --data_path datasets/hugo/eval_set.json \
    --tool_manifest_path tools/tool_manifest_hugo.json
```

## Key Hyperparameters

### Common

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `Qwen/Qwen3-0.6B` | Model name or path |
| `--domain` | *(required)* | `hugo` or `dana` |
| `--seed` | `42` | Random seed |
| `--val_ratio` | `0.2` | Fraction of data held out for validation |
| `--max_tokens` | `2048` | Max tokens for generation and training |
| `--wandb_project` | `NLU-Training` | W&B project name |

### RL (PPO)

| Flag | Default | Description |
|------|---------|-------------|
| `--lr` | `2e-6` | Learning rate |
| `--batch_size` | `8` | Batch size |
| `--num_rollouts` | `100` | Number of RL episodes |
| `--k_value` | `7` | Generations per query |
| `--temperature` | `1.1` | Sampling temperature |
| `--eps_high` | `0.28` | PPO upper clip |
| `--eps_low` | `0.2` | PPO lower clip |
| `--beta` | `0.015` | KL coefficient |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation |
| `--max_grad_norm` | `0.9` | Max gradient norm |
| `--stage_weights` | `None` | Comma-separated reward weights for composed stages |

### SFT

| Flag | Default | Description |
|------|---------|-------------|
| `--sft_lr` | `2e-5` | Learning rate |
| `--sft_epochs` | `3` | Training epochs |
| `--confidence_threshold` | `0.7` | Min ensemble confidence for SFT examples |
