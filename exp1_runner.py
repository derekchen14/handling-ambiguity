#!/usr/bin/env python3
"""Unified CLI for Experiment 1 — single-model flow detection.

Handles both deterministic (temp=0) and stochastic (temp>0) runs.
Self-consistency ensembles are just multiple seeds at temp>0.

Examples:
    # Exp 1A: deterministic run
    python3 exp1_runner.py --domain dana --config 1a_004 --seeds 1

    # Exp 1B: self-consistency (same model, temp>0, multiple seeds)
    python3 exp1_runner.py --domain hugo --config 1a_010 --temperature 0.3 \\
        --seeds 1-3 --output-dir exp1b --label 3v-1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add project root for imports
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers.client import UnifiedLLMClient
from helpers.harness import ExperimentRunner

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CONFIGS_PATH = BASE_DIR / 'helpers' / 'configs' / 'exp1a_configs.json'
RESULTS_DIR = BASE_DIR / 'results'

# Approximate pricing per 1M tokens (input, output) in USD
PRICING = {
    'claude-haiku-4-5-20251001':       (0.80,   4.00),
    'claude-sonnet-4-6':               (3.00,  15.00),
    'claude-opus-4-6':                (15.00,  75.00),
    'gemini-3-flash-preview':          (0.15,   0.60),
    'gemini-3-pro-preview':            (1.25,  10.00),
    'gpt-5-nano':                      (0.10,   0.40),
    'gpt-5-mini':                      (0.40,   1.60),
    'gpt-5.2':                         (2.50,  10.00),
    'deepseek-chat':                   (0.27,   1.10),
    'deepseek-reasoner':               (0.55,   2.19),
    'Qwen/Qwen2.5-7B-Instruct-Turbo': (0.18,   0.18),
    'Qwen/Qwen3-Next-80B-A3B-Instruct': (0.50, 0.50),
    'Qwen/Qwen3-235B-A22B-Thinking-2507': (3.50, 3.50),
    'gemma-3-27b-it':                  (0.15,   0.60),
    'claude-sonnet-4-20250514':        (3.00,  15.00),
    'qwen/qwen3.5-397b-a17b':           (0.55,  3.50),
    'google/gemini-3.1-pro-preview':   (2.00, 12.00)
}


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD from token counts."""
    in_rate, out_rate = PRICING.get(model_id, (1.0, 1.0))
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def load_configs() -> dict[str, dict]:
    """Load all 1A configs keyed by config_id."""
    with open(CONFIGS_PATH) as f:
        configs = json.load(f)
    return {c['config_id']: c for c in configs}


def load_eval_set(domain: str, override_path: Path | None = None) -> list[dict]:
    """Load the eval set for a domain."""
    eval_path = override_path or (BASE_DIR / 'datasets' / domain / 'eval_set.json')
    if not eval_path.exists():
        log.error('Eval set not found: %s', eval_path)
        sys.exit(1)
    with open(eval_path) as f:
        return json.load(f)


def parse_seeds(seed_str: str) -> list[int]:
    """Parse seed specification: '1', '1-5', '1,3,5'."""
    if '-' in seed_str:
        lo, hi = seed_str.split('-')
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in seed_str.split(',')]


def main():
    parser = argparse.ArgumentParser(description='Run Experiment 1 — flow detection')
    parser.add_argument('--domain', required=True, choices=['hugo', 'dana'])
    parser.add_argument('--config', help='Single config_id to run')
    parser.add_argument('--tier', choices=['low', 'medium', 'high'],
                        help='Run all configs at a given model_level')
    parser.add_argument('--all', action='store_true', help='Run all configs')
    parser.add_argument('--seeds', default='1', help='Seed(s): 1, 1-5, or 1,3,5')
    parser.add_argument('--workers', type=int, default=4, help='Max parallel conversations')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (default: 0.0)')
    parser.add_argument('--output-dir', default='exp1a',
                        help='Results subdirectory (default: exp1a)')
    parser.add_argument('--label',
                        help='Override config_id in output filenames (e.g., 3v-1)')
    parser.add_argument('--eval-path', type=Path, default=None,
                        help='Override eval set path (default: datasets/{domain}/eval_set.json)')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    # Quiet noisy third-party loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('google_genai').setLevel(logging.WARNING)
    logging.getLogger('google.genai').setLevel(logging.WARNING)

    configs = load_configs()
    seeds = parse_seeds(args.seeds)
    eval_set = load_eval_set(args.domain, args.eval_path)

    if args.all:
        config_ids = list(configs.keys())
    elif args.tier:
        config_ids = [cid for cid, c in configs.items() if c.get('model_level') == args.tier]
        if not config_ids:
            log.error('No configs found for tier=%s', args.tier)
            sys.exit(1)
    elif args.config:
        if args.config not in configs:
            log.error('Unknown config: %s', args.config)
            sys.exit(1)
        config_ids = [args.config]
    else:
        log.error('Specify --config, --tier, or --all')
        sys.exit(1)

    client = UnifiedLLMClient()
    runner = ExperimentRunner(client, RESULTS_DIR, max_workers=args.workers)

    session_start = time.time()
    total_cost = 0.0
    run_count = 0

    for config_id in config_ids:
        config = dict(configs[config_id])  # copy so we don't mutate
        model_id = config.get('model_id', '')

        # Apply temperature override
        if args.temperature > 0:
            config['temperature'] = args.temperature

        file_label = args.label or config.get('label') or config_id

        for seed in seeds:
            run_start = time.time()
            log.info('Starting %s [%s] domain=%s seed=%d temp=%.1f',
                     file_label, model_id, args.domain, seed, config.get('temperature', 0.0))

            result = runner.run_exp1a(
                args.domain, config, eval_set, seed,
                output_dir=args.output_dir, file_label=file_label,
            )

            # Extract stats
            summary = result.summary.get('summary', {})
            accuracy = summary.get('accuracy_top1', 0)
            in_tok = summary.get('input_tokens_total', 0)
            out_tok = summary.get('output_tokens_total', 0)
            cost = estimate_cost(model_id, in_tok, out_tok)
            wall = time.time() - run_start
            total_cost += cost
            run_count += 1

            # Save summary
            summary_path = (
                RESULTS_DIR / args.output_dir
                / f'{args.domain}_{file_label}_seed{seed}_summary.json'
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(result.summary, f, indent=2)

            log.info(
                'Done %s seed=%d | acc=%.1f%% | %ds wall | %dK in + %dK out | ~$%.3f',
                file_label, seed,
                accuracy * 100,
                int(wall),
                in_tok // 1000, out_tok // 1000,
                cost,
            )

    elapsed = time.time() - session_start
    log.info(
        'Session complete: %d runs | %.0fs total | ~$%.3f estimated cost',
        run_count, elapsed, total_cost,
    )


if __name__ == '__main__':
    main()
