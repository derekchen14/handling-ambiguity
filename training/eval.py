"""Periodic evaluation for NLU training.

Spins up an sglang server with the current checkpoint, runs single-sample
generation on a held-out validation set, computes stage rewards, and returns
aggregate metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from training.stages import PipelineStage
from training.rollouts import NLUTurnExample, _do_generations_parallel, _clean_trajectories


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
            

        print(
            f"Eval: accuracy={metrics['eval/accuracy']:.4f}, "
            f"reward_std={metrics['eval/reward_std']:.4f} "
            f"({len(all_rewards)}/{len(val_examples)} examples)"
        )

    accelerator.wait_for_everyone()
    return metrics
