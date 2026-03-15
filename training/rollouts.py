"""NLU-adapted rollout generation via sglang.

Generates trajectories for RL training by:
1. Walking each conversation to build per-turn rollout examples.
2. Generating K trajectories per turn using sglang's OpenAI-compatible API.
3. Scoring each trajectory using the stage's compute_reward.

Reuses server lifecycle utilities from the existing RL pipeline.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from tqdm import tqdm

from training.stages import PipelineStage
from training.rewards import make_trajectory_eval_callback


@dataclass
class NLUTurnExample:
    """One rollout input = one user turn within a conversation."""

    system_prompt: str
    message_history: list[dict[str, str]]
    turn: dict[str, Any]
    convo: dict[str, Any]
    tool_specs: list[dict] | None = None
    stage_kwargs: dict[str, Any] = field(default_factory=dict)
    domain: str = ''

    @property
    def query_key(self) -> str:
        """Unique identifier for this turn (for logging)."""
        cid = self.convo.get('convo_id', '?')
        tnum = self.turn.get('turn_num', '?')
        return f'{cid}_t{tnum}'


def build_turn_examples(
    eval_set: list[dict],
    stage: PipelineStage,
    domain: str,
    tools: list[dict] | None = None,
    **stage_kwargs: Any,
) -> list[NLUTurnExample]:
    """Build one NLUTurnExample per user turn across all conversations.

    Walks each conversation, building message history incrementally.
    Each user turn becomes one rollout example that gets K trajectories.
    """
    examples: list[NLUTurnExample] = []

    for convo in eval_set:
        message_history: list[dict[str, str]] = []

        for turn in convo.get('turns', []):
            if turn.get('speaker') == 'user':
                utterance = turn['utterance']
                message_history.append({'role': 'user', 'content': utterance})

                system_prompt = stage.build_prompt(
                    domain, turn, convo, **stage_kwargs
                )
                tool_specs = stage.get_tool_specs(
                    domain, turn, convo, tools=tools, **stage_kwargs
                )

                examples.append(NLUTurnExample(
                    system_prompt=system_prompt,
                    message_history=list(message_history),
                    turn=turn,
                    convo=convo,
                    tool_specs=tool_specs,
                    stage_kwargs=stage_kwargs,
                    domain=domain,
                ))

                # Synthetic assistant response for context continuity
                message_history.append({
                    'role': 'assistant',
                    'content': '[response]',
                })
            else:
                # Agent turn
                message_history.append({
                    'role': 'assistant',
                    'content': turn.get('utterance', '[OK]'),
                })

    return examples


def generate_single_nlu_trajectory(
    example: NLUTurnExample,
    client: Any,
    model_name: str,
    stage: PipelineStage,
    domain: str,
    temperature: float = 1.1,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Generate one trajectory for a turn example.

    Calls sglang's OpenAI-compatible chat completions API, parses the
    response, computes the reward using the stage, and returns a trajectory
    dict compatible with ``RLTrainingDataset``.

    Returns:
        {"messages": list[dict], "reward": float} or {"messages": None, "reward": None}
    """
    messages = (
        [{'role': 'system', 'content': example.system_prompt}]
        + example.message_history
    )

    try:
        kwargs: dict[str, Any] = {
            'model': model_name,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        if example.tool_specs:
            kwargs['tools'] = [
                {'type': 'function', 'function': spec}
                for spec in example.tool_specs
            ]

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # Diagnostic logging for empty responses
        if not choice.message.content and not choice.message.tool_calls:
            reasoning = getattr(choice.message, 'reasoning_content', None)
            if reasoning:
                print(f'[DEBUG] Empty content for {example.query_key}: '
                      f'reasoning_content={reasoning!r:.200}')
            else:
                print(f'[DEBUG] Empty response for {example.query_key}')

        # Extract raw response
        if choice.message.tool_calls:
            # Tool-calling response: serialize tool calls to JSON
            tool_calls = []
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    'name': tc.function.name,
                    'args': json.loads(tc.function.arguments) if tc.function.arguments else {},
                })
            raw_response = json.dumps(tool_calls)
            assistant_content = raw_response
        else:
            raw_response = choice.message.content or ''
            assistant_content = raw_response

        if not raw_response:
            # Fallback: check reasoning_content (sglang reasoning parsers)
            reasoning = getattr(choice.message, 'reasoning_content', None) or ''
            if reasoning:
                raw_response = reasoning
                assistant_content = reasoning
            else:
                return {'messages': None, 'reward': None}

        # Compute reward
        effective_domain = example.domain or domain
        callback = make_trajectory_eval_callback(
            stage, effective_domain, example.convo, example.turn, **example.stage_kwargs
        )
        reward = callback(raw_response)

        # Build trajectory messages (full conversation + assistant response)
        traj_messages = messages + [{'role': 'assistant', 'content': assistant_content}]

        return {'messages': traj_messages, 'reward': reward}

    except Exception as e:
        print(f'Trajectory generation error for {example.query_key}: {e}')
        return {'messages': None, 'reward': None}


def _do_generations_parallel(
    examples: list[NLUTurnExample],
    k: int,
    client: Any,
    model_name: str,
    stage: PipelineStage,
    domain: str,
    temperature: float = 1.1,
    max_tokens: int = 2048,
    max_workers: int = 30,
) -> tuple[list[list[dict]], list[str]]:
    """Generate K trajectories per example in parallel.

    Returns:
        (trajectories, queries) where trajectories[i] is a list of
        k trajectory dicts for example i.
    """
    num_examples = len(examples)
    trajectories: list[list[dict]] = [[] for _ in range(num_examples)]
    queries: list[str] = [ex.query_key for ex in examples]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, example in enumerate(examples):
            for _ in range(k):
                future = executor.submit(
                    generate_single_nlu_trajectory,
                    example, client, model_name, stage, domain,
                    temperature, max_tokens,
                )
                futures.append((i, future))

        for i, future in tqdm(futures, total=len(futures), desc='Generating trajectories'):
            traj = future.result()
            if traj.get('messages') is not None:
                trajectories[i].append(traj)

    return trajectories, queries


def _clean_trajectories(trajectories: list[list[dict]]) -> list[list[dict]]:
    """Remove None content from trajectory messages."""
    cleaned = []
    for group in trajectories:
        clean_group = []
        for traj in group:
            if traj.get('messages') is not None:
                msgs = []
                for m in traj['messages']:
                    content = m.get('content')
                    if content is None:
                        content = ''
                    msgs.append({'role': m['role'], 'content': content})
                clean_group.append({'messages': msgs, 'reward': traj['reward']})
        cleaned.append(clean_group)
    return cleaned


def do_nlu_rollout(
    args: Any,
    model_name: str,
    accelerator: Any,
    train_examples: list[NLUTurnExample],
    val_examples: list[NLUTurnExample],
    stage: PipelineStage,
    domain: str,
) -> tuple[list[list[dict]], list[str], list[list[dict]], list[str]]:
    """Full rollout cycle: launch sglang, generate trajectories, cleanup.

    Mirrors ``do_rollout`` from ``rl/train/data/generation.py``.

    Returns:
        (train_trajectories, train_queries, val_trajectories, val_queries)
        in the format expected by ``RLTrainingDataset``.
    """
    from training.utils import (
        cleanup_server,
        setup_server_and_client,
        save_json,
        load_json,
    )

    if accelerator.is_main_process:
        server_process, port, client = setup_server_and_client(
            model_name, accelerator.num_processes, seed=args.seed,
        )

        train_trajectories, train_queries = _do_generations_parallel(
            train_examples, args.k_value, client, model_name,
            stage, domain, args.temperature, args.max_tokens,
        )
        val_trajectories, val_queries = _do_generations_parallel(
            val_examples, args.k_value, client, model_name,
            stage, domain, args.temperature, args.max_tokens,
        )

        cleanup_server(server_process)

        train_trajectories = _clean_trajectories(train_trajectories)
        val_trajectories = _clean_trajectories(val_trajectories)

        save_json(train_trajectories, 'train_trajectories.json')
        save_json(train_queries, 'all_queries.json')
        save_json(val_trajectories, 'val_trajectories.json')
        save_json(val_queries, 'val_queries.json')

    accelerator.wait_for_everyone()
    train_trajectories = load_json('train_trajectories.json')
    train_queries = load_json('all_queries.json')
    val_trajectories = load_json('val_trajectories.json')
    val_queries = load_json('val_queries.json')

    return train_trajectories, train_queries, val_trajectories, val_queries
