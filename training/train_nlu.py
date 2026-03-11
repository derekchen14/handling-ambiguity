"""Entry point for NLU pipeline training (SFT + RL).

Mirrors ``rl/train/train.py`` with NLU-specific stages and rollout logic.

Usage examples:

    # RL: train flow detection
    python -m training.train_nlu --mode rl --stages flow --domain hugo \
        --model_name Qwen/Qwen3-0.6B --data_path datasets/hugo/eval_set.json

    # RL: train tool selection + param extraction as one stage
    python -m training.train_nlu --mode rl \
        --stages tool_selection+param_extraction --stage_weights 0.6,0.4 \
        --domain hugo --model_name Qwen/Qwen3-0.6B \
        --data_path datasets/hugo/eval_set.json \
        --tool_manifest_path tools/tool_manifest_hugo.json

    # SFT: distil from gold labels
    python -m training.train_nlu --mode sft --stages flow --domain hugo \
        --model_name Qwen/Qwen3-0.6B --data_path datasets/hugo/eval_set.json

    # SFT: distil from ensemble results
    python -m training.train_nlu --mode sft --stages flow --domain hugo \
        --model_name Qwen/Qwen3-0.6B --data_path datasets/hugo/eval_set.json \
        --ensemble_results_path results/exp1b/hugo_3v-9_seed1.jsonl \
        --confidence_threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer
from accelerate import Accelerator

from training.stages import (
    STAGE_REGISTRY,
    ComposedStage,
    PipelineStage,
)
from training.rollouts import build_turn_examples, do_nlu_rollout
from training.utils import run_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train NLU pipeline stages with SFT or RL',
    )

    # Mode and stage
    parser.add_argument('--mode', type=str, choices=['sft', 'rl'], required=True,
                        help='Training mode')
    parser.add_argument('--stages', type=str, required=True,
                        help='Stage(s) to train, e.g. "flow" or "tool_selection+param_extraction"')
    parser.add_argument('--stage_weights', type=str, default=None,
                        help='Comma-separated reward weights for composed stages, e.g. "0.6,0.4"')

    # Data
    parser.add_argument('--domain', type=str, choices=['hugo', 'dana'], required=True,
                        help='Domain')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset JSON (used for both training and validation)')
    parser.add_argument('--tool_manifest_path', type=str, default=None,
                        help='Path to tool manifest JSON (for tool stages)')
    parser.add_argument('--ensemble_results_path', type=str, default=None,
                        help='Path to ensemble result JSONL (for SFT from ensemble)')

    # Model
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-0.6B',
                        help='Model name or path')
    parser.add_argument('--reference_model_name', type=str, default=None,
                        help='Reference model for KL divergence (defaults to model_name)')

    # RL hyperparams
    parser.add_argument('--temperature', type=float, default=1.1,
                        help='Sampling temperature for rollouts')
    parser.add_argument('--max_tokens', type=int, default=2048,
                        help='Max tokens for generation and training')
    parser.add_argument('--k_value', type=int, default=7,
                        help='Number of generations per query')
    parser.add_argument('--eps_high', type=float, default=0.28,
                        help='PPO upper clip')
    parser.add_argument('--eps_low', type=float, default=0.2,
                        help='PPO lower clip')
    parser.add_argument('--num_rollouts', type=int, default=100,
                        help='Number of RL episodes')
    parser.add_argument('--beta', type=float, default=0.015,
                        help='KL coefficient')
    parser.add_argument('--lr', type=float, default=2e-6,
                        help='RL learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=float, default=0.9)
    parser.add_argument('--compute_reference_probs', action='store_true')
    parser.add_argument('--use_kl_div', action='store_true')
    parser.add_argument('--use_tok_pg', action='store_true')
    parser.add_argument('--entropy_coef', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--use_value_head', action='store_true')
    parser.add_argument('--value_head_path', type=str, default='./value_head.pth')

    # SFT hyperparams
    parser.add_argument('--sft_epochs', type=int, default=3,
                        help='Number of SFT training epochs')
    parser.add_argument('--sft_lr', type=float, default=2e-5,
                        help='SFT learning rate')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                        help='Minimum ensemble confidence for SFT examples')

    # LoRA
    parser.add_argument('--use_lora', action='store_true',
                        help='Use LoRA for SFT instead of full fine-tuning')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA rank (r)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha scaling factor')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout rate')
    parser.add_argument('--lora_target_modules', type=str, default=None,
                        help='Comma-separated target modules for LoRA (default: peft auto-detect)')

    # Eval (SFT only)
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Run evaluation every N epochs (SFT only). 0 to disable.')
    parser.add_argument('--eval_temperature', type=float, default=1.0,
                        help='Sampling temperature for evaluation.')

    # Output
    parser.add_argument('--model_save_path', type=str, default='./nlu_model',
                        help='Path to save model checkpoints')
    parser.add_argument('--opt_save_path', type=str, default='./nlu_optimizer.pth',
                        help='Path to save optimizer state')
    parser.add_argument('--wandb_project', type=str, default='NLU-Training',
                        help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Wandb run name (auto-generated if not set)')

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Fraction of data held out for validation')

    args = parser.parse_args()

    if args.reference_model_name is None:
        args.reference_model_name = args.model_name

    if args.wandb_name is None:
        args.wandb_name = f'{args.mode}_{args.stages}_{args.domain}'

    return args


def build_stage(stage_spec: str, weights_spec: str | None = None) -> PipelineStage:
    """Parse a stage specification string into a PipelineStage.

    Examples:
        "flow"                           -> FlowStage()
        "tool_selection+param_extraction" -> ComposedStage([ToolSelectionStage(), ParamExtractionStage()])
    """
    parts = [p.strip() for p in stage_spec.split('+')]

    if len(parts) == 1:
        name = parts[0]
        if name not in STAGE_REGISTRY:
            raise ValueError(f'Unknown stage: {name}. Available: {list(STAGE_REGISTRY.keys())}')
        return STAGE_REGISTRY[name]()

    # Composed stage
    stages = []
    for name in parts:
        if name not in STAGE_REGISTRY:
            raise ValueError(f'Unknown stage: {name}. Available: {list(STAGE_REGISTRY.keys())}')
        stages.append(STAGE_REGISTRY[name]())

    weights = None
    if weights_spec:
        weights = [float(w.strip()) for w in weights_spec.split(',')]
        if len(weights) != len(stages):
            raise ValueError(
                f'weights count ({len(weights)}) must match stages count ({len(stages)})'
            )

    return ComposedStage(stages, weights)


def load_data(args) -> tuple[list[dict], list[dict] | None]:
    """Load eval set and optional tool manifest."""
    with open(args.data_path) as f:
        eval_set = json.load(f)

    tools = None
    if args.tool_manifest_path:
        with open(args.tool_manifest_path) as f:
            tools = json.load(f)

    return eval_set, tools


def run_rl(
    args: argparse.Namespace,
    stage: PipelineStage,
    eval_set: list[dict],
    tools: list[dict] | None = None,
) -> None:
    """RL training loop, structurally identical to rl/train/train.py:main()."""
    from training.utils import RLTrainingDataset, PPOTrainer, free_memory

    if args.use_kl_div and not args.compute_reference_probs:
        raise ValueError(
            'Cannot use KL divergence without reference probabilities. '
            'Set --compute_reference_probs.'
        )

    accelerator = Accelerator(mixed_precision='fp16')

    # Build stage kwargs
    stage_kwargs: dict = {}
    if tools:
        stage_kwargs['tools'] = tools
        from helpers.schema_utils import build_param_schema_index
        stage_kwargs['param_schema_index'] = build_param_schema_index(tools)

    # Build turn examples from eval set
    all_examples = build_turn_examples(eval_set, stage, args.domain, tools, **stage_kwargs)
    random.seed(args.seed)
    random.shuffle(all_examples)

    split_idx = int(len(all_examples) * (1 - args.val_ratio))
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    print(f'Train examples: {len(train_examples)}, Val examples: {len(val_examples)}')

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(args.model_save_path)
    if not os.environ.get('TOKENIZERS_PARALLELISM'):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Tool specs for RLTrainingDataset
    tool_specs = None
    if tools:
        from prompts.tool_calling import strip_tool_metadata
        tool_specs = strip_tool_metadata(tools)

    # Wandb
    if accelerator.is_main_process:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config={
                    'mode': 'rl',
                    'stage': stage.name,
                    'domain': args.domain,
                    'model': args.model_name,
                    'reference_model': args.reference_model_name,
                    'k': args.k_value,
                    'temperature': args.temperature,
                    'max_tokens': args.max_tokens,
                    'beta': args.beta,
                    'lr': args.lr,
                },
            )
            wandb.define_metric('episode')
            wandb.define_metric('Avg Reward', step_metric='episode')
            wandb.define_metric('Reward Std', step_metric='episode')
            wandb.define_metric('Avg Val Reward', step_metric='episode')
            wandb.define_metric('Loss', step_metric='episode')
            wandb.define_metric('KL Divergence', step_metric='episode')
            wandb.define_metric('Entropy', step_metric='episode')
        except ImportError:
            pass

    # Training loop
    model_name = args.model_name
    optimizer_path = None
    value_head_path = None

    for episode in range(args.num_rollouts):
        print(f'\n\nEpisode {episode + 1}')

        # Rollouts
        train_trajectories, train_queries, val_trajectories, val_queries = do_nlu_rollout(
            args, model_name, accelerator,
            train_examples, val_examples,
            stage, args.domain,
        )

        # Build dataset
        train_dataset = RLTrainingDataset(
            train_trajectories, tool_specs, tokenizer,
            max_length=args.max_tokens, actor_critic=args.use_value_head,
        )

        # Trainer
        trainer = PPOTrainer(
            model_name,
            args.reference_model_name,
            args.use_value_head,
            value_head_path,
            accelerator,
            train_dataset,
            batch_size=args.batch_size,
            lr=args.lr,
            eps_high=args.eps_high,
            eps_low=args.eps_low,
            beta=args.beta,
            k_value=args.k_value,
            optimizer_path=optimizer_path,
            use_kl_div=args.use_kl_div,
            use_tok_pg=args.use_tok_pg,
            entropy_coef=args.entropy_coef,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            group_size=args.k_value,
            max_grad_norm=args.max_grad_norm,
        )

        if args.compute_reference_probs:
            trainer.compute_reference_probs()

        trainer.compute_old_policy_probs()

        # Train
        total_loss, total_kl_div, total_entropy = trainer.train_epoch()

        # Save
        model_name, optimizer_path, value_head_path = trainer.save(
            args.model_save_path, args.opt_save_path, args.value_head_path,
        )

        # Log
        trainer.log_metrics(
            episode,
            train_trajectories,
            train_queries,
            total_loss,
            total_kl_div,
            total_entropy,
            advantages=train_dataset.advantages,
            val_trajectories=val_trajectories,
            val_queries=val_queries,
        )

        # Cleanup
        del trainer, train_dataset
        free_memory(accelerator)

    if accelerator.is_main_process:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    args = parse_args()

    # Load data
    eval_set, tools = load_data(args)
    print(f'Loaded {len(eval_set)} conversations from {args.data_path}')
    if tools:
        print(f'Loaded {len(tools)} tools from {args.tool_manifest_path}')

    # Build stage
    stage = build_stage(args.stages, args.stage_weights)
    print(f'Stage: {stage.name}')

    # Dispatch
    if args.mode == 'sft':
        run_sft(args, stage, eval_set, tools)
    elif args.mode == 'rl':
        run_rl(args, stage, eval_set, tools)


if __name__ == '__main__':
    main()
