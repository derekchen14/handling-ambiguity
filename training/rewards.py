"""Bridge between pipeline stage scoring and RL reward callbacks.

Provides factory functions that create reward callbacks compatible with
the rollout generation pipeline (``do_nlu_rollout``).  Rewards are derived
directly from ``helpers/scoring.py`` via the stage abstraction — never
duplicated.
"""

from __future__ import annotations

from typing import Any, Callable

from training.stages import PipelineStage, StageOutput, StageReward


def make_trajectory_eval_callback(
    stage: PipelineStage,
    domain: str,
    convo: dict,
    turn: dict,
    **kwargs: Any,
) -> Callable[[str], float]:
    """Return a ``callable(raw_response) -> float`` compatible with rollout generation.

    The callback parses the raw model response using the stage's parser,
    then scores it using the stage's reward function (which delegates to
    ``helpers/scoring.py``).

    Args:
        stage: The pipeline stage (or ComposedStage) to use for parsing + scoring.
        domain: Domain name ('hugo' or 'dana').
        convo: The full conversation dict from eval_set.
        turn: The specific user turn dict being evaluated.
        **kwargs: Extra kwargs forwarded to parse_response / compute_reward
                  (e.g. tool_lookup, param_schema_index, fuzzy_evaluator).

    Returns:
        A function that takes a raw model response string and returns a
        scalar reward in [0, 1].
    """

    # Ensure domain isn't duplicated in kwargs
    fwd_kwargs = {k: v for k, v in kwargs.items() if k != 'domain'}

    def callback(raw_response: str) -> float:
        output: StageOutput = stage.parse_response(raw_response, turn, **fwd_kwargs)
        reward: StageReward = stage.compute_reward(
            output, turn, convo, domain=domain, **fwd_kwargs
        )
        return reward.reward

    return callback


def make_batch_reward_fn(
    stage: PipelineStage,
    domain: str,
    examples: list[dict],
    **kwargs: Any,
) -> Callable[[int, str], float]:
    """Return a callback that scores by example index.

    Each entry in *examples* should have ``'turn'`` and ``'convo'`` keys.
    The returned callable takes ``(example_idx, raw_response) -> float``.
    """

    fwd_kwargs = {k: v for k, v in kwargs.items() if k != 'domain'}

    def callback(idx: int, raw_response: str) -> float:
        ex = examples[idx]
        output = stage.parse_response(raw_response, ex['turn'], **fwd_kwargs)
        reward = stage.compute_reward(
            output, ex['turn'], ex['convo'], domain=domain, **fwd_kwargs
        )
        return reward.reward

    return callback
