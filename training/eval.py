"""Periodic evaluation for NLU training.

Spins up an sglang server with the current checkpoint, runs single-sample
generation on a held-out validation set, computes stage rewards, and returns
aggregate metrics.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import numpy as np

from training.stages import PipelineStage
from training.rollouts import NLUTurnExample, _do_generations_parallel, _clean_trajectories


def _compute_breakdown_metrics(
    val_examples: list[NLUTurnExample],
    trajectories: list[list[dict]],
    stage: PipelineStage,
    seed: int,
) -> dict[str, float]:
    """Compute per-category, per-domain, and ambiguity breakdown metrics.

    Aligned with ``results/build_report_1b.py`` ``_baseline_ambiguity_detail``
    for single-model scoring:
    - Recognition: ``len(detected_flows) >= 2``
    - Selection: ``random.choice(detected_flows) in candidate_flows``
    - Selection only computed when recognition passed
    """
    # Per-category and per-domain reward accumulators
    by_category: dict[str, list[float]] = defaultdict(list)
    by_domain: dict[str, list[float]] = defaultdict(list)

    # Ambiguity breakdown accumulators
    amb_recognized: list[bool] = []
    amb_selected: list[bool] = []

    rng = random.Random(seed)

    for example, traj_group in zip(val_examples, trajectories):
        if not traj_group:
            continue
        reward = traj_group[0].get('reward')
        if reward is None:
            continue

        category = example.convo.get('category', 'unknown')
        ex_domain = example.domain

        by_category[category].append(reward)
        if ex_domain:
            by_domain[ex_domain].append(reward)

        # Ambiguity breakdown: only for turns with candidate_flows
        candidate_flows = example.turn.get('candidate_flows')
        if candidate_flows:
            # Re-parse to get detected flows
            raw_response = traj_group[0].get('messages', [{}])[-1].get('content', '')
            parsed_output = stage.parse_response(raw_response, example.turn)
            detected_flows = parsed_output.parsed.get('flows', [])

            recognized = len(detected_flows) >= 2
            amb_recognized.append(recognized)

            if recognized:
                predicted = rng.choice(detected_flows)
                amb_selected.append(predicted in set(candidate_flows))

    metrics: dict[str, float] = {}

    # Per-category accuracy
    for cat, rewards in by_category.items():
        metrics[f'eval/{cat}/accuracy'] = float(np.mean(rewards))

    # Ambiguous overall (combined ambiguous_first + ambiguous_second)
    amb_rewards = by_category.get('ambiguous_first', []) + by_category.get('ambiguous_second', [])
    if amb_rewards:
        metrics['eval/ambiguous_overall'] = float(np.mean(amb_rewards))

    # Per-domain accuracy
    for dom, rewards in by_domain.items():
        metrics[f'eval/{dom}/accuracy'] = float(np.mean(rewards))

    # Ambiguity recognition/selection
    if amb_recognized:
        metrics['eval/ambiguous_recognized'] = sum(amb_recognized) / len(amb_recognized)
    if amb_selected:
        metrics['eval/ambiguous_selected'] = sum(amb_selected) / len(amb_selected)

    return metrics


def evaluate(
    model_path: str,
    val_examples: list[NLUTurnExample],
    stage: PipelineStage,
    domain: str,
    accelerator: Any,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    seed: int = 42,
) -> dict[str, float]:
    """Spin up sglang, run eval on val set, return metrics dict.

    Only the main process launches the server and runs generation.
    All processes synchronise via ``accelerator.wait_for_everyone()``.
    """
    from training.utils import setup_server_and_client, cleanup_server

    metrics: dict[str, float] = {}

    if accelerator.is_main_process:
        server_process, port, client = setup_server_and_client(
            model_path, num_gpus=1, seed=seed,
        )

        trajectories, queries = _do_generations_parallel(
            val_examples,
            k=1,
            client=client,
            model_name=model_path,
            stage=stage,
            domain=domain,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        cleanup_server(server_process)
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

        trajectories = _clean_trajectories(trajectories)

        # Aggregate rewards
        all_rewards = []
        for group in trajectories:
            for traj in group:
                if traj.get('reward') is not None:
                    all_rewards.append(traj['reward'])

        if all_rewards:
            metrics['eval/accuracy'] = float(np.mean(all_rewards))
            metrics['eval/reward_std'] = float(np.std(all_rewards))
        else:
            metrics['eval/accuracy'] = 0.0
            metrics['eval/reward_std'] = 0.0

        # Breakdown metrics (per-category, per-domain, ambiguity)
        breakdown = _compute_breakdown_metrics(
            val_examples, trajectories, stage, seed,
        )
        metrics.update(breakdown)

        print(
            f"Eval: accuracy={metrics['eval/accuracy']:.4f}, "
            f"reward_std={metrics['eval/reward_std']:.4f} "
            f"({len(all_rewards)}/{len(val_examples)} examples)"
        )
        # Log breakdown summary
        for key in sorted(breakdown):
            print(f"  {key}={breakdown[key]:.4f}")

    accelerator.wait_for_everyone()
    return metrics
