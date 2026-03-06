"""Experiment orchestrator — runs configs across eval sets, multi-turn aware."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

import numpy as np

from helpers.client import UnifiedLLMClient
from helpers.scoring import (
    score_turn, score_turn_ensemble, tally_votes_multi,
    score_intent,
    build_tool_flow_map, score_nlu_staged_funnel,
    score_tool_turn, build_fuzzy_evaluator,
)
from prompts.flow_detection import build_flow_detection_prompt
from prompts.intent_classification import build_intent_classification_prompt
from prompts.slot_filling import build_slot_filling_prompt, get_flow_slot_schema
from prompts.tool_calling import build_tool_calling_prompt, strip_tool_metadata

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Summary of a single experiment run."""
    run_id: str
    experiment: str
    domain: str
    config_id: str
    seed: int
    timestamp: str
    conversations: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


class ExperimentRunner:
    """Orchestrates running experiments across configs, domains, and seeds."""

    def __init__(
        self,
        client: UnifiedLLMClient,
        results_dir: Path,
        max_workers: int = 4,
    ):
        self.client = client
        self.results_dir = results_dir
        self.max_workers = max_workers

    # ── Experiment 1A: Single-model flow detection ────────────────

    def run_exp1a(
        self,
        domain: str,
        config: dict,
        eval_set: list[dict],
        seed: int,
        output_dir: str = 'exp1a',
        file_label: str | None = None,
    ) -> RunResult:
        """Run single-model flow detection on all eval conversations."""
        config_id = config['config_id']
        label = file_label or config_id
        run_id = f'{output_dir}_{domain}_{label}_seed{seed}'

        output_path = self.results_dir / output_dir / f'{domain}_{label}_seed{seed}.jsonl'
        completed = self._load_completed(output_path)

        log.info('Run %s: %d conversations, %d already done',
                 run_id, len(eval_set), len(completed))

        remaining = [c for c in eval_set if c['convo_id'] not in completed]

        results = list(completed.values())
        results.extend(
            self._run_conversations_parallel(
                remaining, config, domain, output_path,
            )
        )

        summary = self._compute_summary(results, run_id, output_dir, domain, label, seed)
        return RunResult(
            run_id=run_id, experiment=output_dir, domain=domain,
            config_id=label, seed=seed,
            timestamp=summary['timestamp'],
            conversations=results, summary=summary,
        )

    # ── Experiment 1B: Self-consistency ensemble ──────────────────

    def run_exp1b_self_consistency(
        self,
        domain: str,
        ensemble_config: dict,
        system_prompt: str,
        eval_set: list[dict],
        seed: int,
    ) -> RunResult:
        """Run self-consistency ensemble (same model × N at temp>0)."""
        ensemble_id = ensemble_config['ensemble_id']
        run_id = f'exp1b_{domain}_{ensemble_id}_seed{seed}'

        output_path = self.results_dir / 'exp1b' / f'{domain}_{ensemble_id}_seed{seed}.jsonl'
        completed = self._load_completed(output_path)

        temperature = ensemble_config.get('temperature', 0.3)

        remaining = [c for c in eval_set if c['convo_id'] not in completed]
        results = list(completed.values())

        for convo in remaining:
            convo_result = self._run_self_consistency_convo(
                convo, ensemble_config, system_prompt, temperature,
            )
            results.append(convo_result)
            self._append_jsonl(output_path, convo_result)

        summary = self._compute_summary(results, run_id, 'exp1b', domain, ensemble_id, seed)
        return RunResult(
            run_id=run_id, experiment='1B', domain=domain,
            config_id=ensemble_id, seed=seed,
            timestamp=summary['timestamp'],
            conversations=results, summary=summary,
        )

    # ── Experiment 2: Tool-calling flow detection ─────────────────

    def run_exp2(
        self,
        domain: str,
        config: dict,
        system_prompt: str | None,
        eval_set: list[dict],
        tools: list[dict],
        seed: int,
    ) -> RunResult:
        """Run tool-calling flow detection on all eval conversations.

        If system_prompt is None, a per-conversation prompt is built
        with context metadata injected from the eval data.
        """
        config_id = config['config_id']
        run_id = f'exp2_{domain}_{config_id}_seed{seed}'

        output_path = self.results_dir / 'exp2b' / f'{domain}_{config_id}_seed{seed}.jsonl'
        completed = self._load_completed(output_path)

        # Build flow map from full manifest (with metadata), then strip for LLM
        tool_flow_map = build_tool_flow_map(tools)
        client_tools = strip_tool_metadata(tools)

        remaining = [c for c in eval_set if c['convo_id'] not in completed]
        results = list(completed.values())

        fuzzy_eval = build_fuzzy_evaluator(self.client)

        for convo in remaining:
            # Build per-convo prompt with context if no fixed prompt provided
            prompt = system_prompt or build_tool_calling_prompt(
                domain, convo.get('context'),
            )
            convo_result = self._run_tool_convo(
                convo, config, prompt, client_tools, tool_flow_map,
                domain=domain, fuzzy_evaluator=fuzzy_eval,
            )
            results.append(convo_result)
            self._append_jsonl(output_path, convo_result)

        summary = self._compute_summary(results, run_id, 'exp2b', domain, config_id, seed)
        return RunResult(
            run_id=run_id, experiment='2', domain=domain,
            config_id=config_id, seed=seed,
            timestamp=summary['timestamp'],
            conversations=results, summary=summary,
        )

    # ── Experiment 2C: Tool-calling with ambiguity hint ─────────────

    def run_exp2c(
        self,
        domain: str,
        config: dict,
        system_prompt: str | None,
        eval_set: list[dict],
        tools: list[dict],
        seed: int,
    ) -> RunResult:
        """Run flat tool-calling with ambiguity hint (Exp 2C).

        Same as run_exp2 but passes mode='hint' to build_tool_calling_prompt
        so the system prompt includes an explicit ambiguity-handling paragraph.
        """
        config_id = config['config_id']
        run_id = f'exp2c_{domain}_{config_id}_seed{seed}'

        output_path = self.results_dir / 'exp2c' / f'{domain}_{config_id}_seed{seed}.jsonl'
        completed = self._load_completed(output_path)

        # Build flow map from full manifest (with metadata), then strip for LLM
        tool_flow_map = build_tool_flow_map(tools)
        client_tools = strip_tool_metadata(tools)

        remaining = [c for c in eval_set if c['convo_id'] not in completed]
        results = list(completed.values())

        fuzzy_eval = build_fuzzy_evaluator(self.client)

        for convo in remaining:
            # Build per-convo prompt with context and hint mode
            prompt = system_prompt or build_tool_calling_prompt(
                domain, convo.get('context'), mode='hint',
            )
            convo_result = self._run_tool_convo(
                convo, config, prompt, client_tools, tool_flow_map,
                domain=domain, fuzzy_evaluator=fuzzy_eval,
            )
            results.append(convo_result)
            self._append_jsonl(output_path, convo_result)

        summary = self._compute_summary(results, run_id, 'exp2c', domain, config_id, seed)
        return RunResult(
            run_id=run_id, experiment='exp2c', domain=domain,
            config_id=config_id, seed=seed,
            timestamp=summary['timestamp'],
            conversations=results, summary=summary,
        )

    # ── Experiment 2: Intent classification ─────────────────────────

    def run_exp2_intent(
        self,
        domain: str,
        config: dict,
        eval_set: list[dict],
        seed: int,
    ) -> RunResult:
        """Run intent classification on all eval conversations."""
        config_id = config['config_id']
        run_id = f'exp2a_intent_{domain}_{config_id}_seed{seed}'

        output_path = self.results_dir / 'exp2a' / 'intents' / f'{domain}_{config_id}_seed{seed}.jsonl'
        completed = self._load_completed(output_path)

        system_prompt = build_intent_classification_prompt(domain)

        remaining = [c for c in eval_set if c['convo_id'] not in completed]
        results = list(completed.values())

        for convo in remaining:
            convo_result = self._run_intent_convo(convo, config, system_prompt)
            results.append(convo_result)
            self._append_jsonl(output_path, convo_result)

        summary = self._compute_summary(results, run_id, 'exp2a_intent', domain, config_id, seed)
        return RunResult(
            run_id=run_id, experiment='2a_intent', domain=domain,
            config_id=config_id, seed=seed,
            timestamp=summary['timestamp'],
            conversations=results, summary=summary,
        )

    # ── Experiment 2: Slot-filling ───────────────────────────────────

    def run_exp2_slots(
        self,
        domain: str,
        config: dict,
        eval_set: list[dict],
        seed: int,
    ) -> RunResult:
        """Run slot-filling on all eval conversations (using gold flow)."""
        config_id = config['config_id']
        run_id = f'exp2a_slot_{domain}_{config_id}_seed{seed}'

        output_path = self.results_dir / 'exp2a' / 'slots' / f'{domain}_{config_id}_seed{seed}.jsonl'
        completed = self._load_completed(output_path)

        remaining = [c for c in eval_set if c['convo_id'] not in completed]
        results = list(completed.values())

        for convo in remaining:
            convo_result = self._run_slot_convo(convo, config, domain)
            results.append(convo_result)
            self._append_jsonl(output_path, convo_result)

        summary = self._compute_summary(results, run_id, 'exp2a_slot', domain, config_id, seed)
        return RunResult(
            run_id=run_id, experiment='2a_slot', domain=domain,
            config_id=config_id, seed=seed,
            timestamp=summary['timestamp'],
            conversations=results, summary=summary,
        )

    # ── Experiment 2A: Scoped tool selection ──────────────────────

    @staticmethod
    def _filter_tools_by_flow(
        tools: list[dict], flow_name: str,
    ) -> list[dict]:
        """Return domain tools scoped to flow_name.

        Excludes internal component tools (handle_ambiguity, coordinate_context,
        etc.) because the upstream NLU pipeline has already handled ambiguity
        resolution and context gathering.
        """
        filtered = []
        for tool in tools:
            if tool.get('internal_component'):
                continue
            flows = tool.get('_flows', [])
            if flow_name in flows:
                filtered.append(tool)
        return filtered

    def run_exp2_scoped_tool(
        self,
        domain: str,
        config: dict,
        eval_set: list[dict],
        tools: list[dict],
        seed: int,
    ) -> RunResult:
        """Run scoped tool-calling: per-turn tools filtered by gold flow.

        Builds a per-conversation system prompt with context injection
        (post/dataset metadata), matching how flat mode operates.
        """
        config_id = config['config_id']
        run_id = f'exp2a_tool_{domain}_{config_id}_seed{seed}'

        output_path = self.results_dir / 'exp2a' / 'tools' / f'{domain}_{config_id}_seed{seed}.jsonl'
        completed = self._load_completed(output_path)

        remaining = [c for c in eval_set if c['convo_id'] not in completed]
        results = list(completed.values())

        fuzzy_eval = build_fuzzy_evaluator(self.client)

        for convo in remaining:
            # Build per-convo prompt with entity context (same as flat mode)
            system_prompt = build_tool_calling_prompt(
                domain, convo.get('context'), mode='scoped_tool',
            )
            convo_result = self._run_scoped_tool_convo(
                convo, config, system_prompt, tools, domain=domain,
                fuzzy_evaluator=fuzzy_eval,
            )
            results.append(convo_result)
            self._append_jsonl(output_path, convo_result)

        summary = self._compute_summary(results, run_id, 'exp2a_tool', domain, config_id, seed)
        return RunResult(
            run_id=run_id, experiment='2a_tool', domain=domain,
            config_id=config_id, seed=seed,
            timestamp=summary['timestamp'],
            conversations=results, summary=summary,
        )

    # ── Internal: conversation runners ────────────────────────────

    @staticmethod
    def _get_turn_intents(turn: dict) -> list[str]:
        """Determine which intent(s) to scope the candidate set.

        For ambiguous turns, uses candidate_intents (may span two intents).
        For clear turns, uses the gold intent label.
        """
        if turn.get('candidate_intents'):
            return list(dict.fromkeys(turn['candidate_intents']))  # dedupe, preserve order
        if turn.get('intent'):
            return [turn['intent']]
        return []

    @staticmethod
    def _score_turn(turn: dict, category: str, detected_flows: list[str]) -> bool:
        """Score a turn using shared scoring rules.

        Returns True/False.  See runner/scoring.py for full documentation.
        """
        return score_turn(
            category,
            detected_flows,
            turn.get('flow', ''),
            turn.get('candidate_flows'),
        )

    def _run_single_convo(
        self, convo: dict, config: dict, domain: str,
    ) -> dict:
        """Run flow detection on a single multi-turn conversation."""
        convo_id = convo['convo_id']
        category = convo.get('category', 'unknown')
        turns_spec = convo.get('turns', [])

        message_history: list[dict] = []
        turn_results = []

        for turn in turns_spec:
            if turn.get('speaker') == 'user':
                # Build per-turn system prompt scoped to this turn's intent(s)
                intents = self._get_turn_intents(turn)
                system_prompt = build_flow_detection_prompt(domain, intents)

                utterance = turn['utterance']
                message_history.append({'role': 'user', 'content': utterance})

                result = self.client.call_flow_detection(
                    config, system_prompt, list(message_history),
                )
                detected_flows = result.get('detected_flows', [])
                reasoning = result.get('reasoning', '')

                correct = self._score_turn(turn, category, detected_flows)

                turn_results.append({
                    'turn_num': turn['turn_num'],
                    'utterance': utterance,
                    'flow': turn.get('flow'),
                    'candidate_flows': turn.get('candidate_flows'),
                    'intent': turn.get('intent'),
                    'detected_flows': detected_flows,
                    'reasoning': reasoning,
                    'correct': correct,
                    'latency_ms': result.get('latency_ms', 0),
                    'input_tokens': result.get('input_tokens', 0),
                    'output_tokens': result.get('output_tokens', 0),
                })

                # Add a synthetic assistant response for context
                message_history.append({
                    'role': 'assistant',
                    'content': f'[Detected flow: {", ".join(detected_flows)}]',
                })
            else:
                # Agent turn — just add context
                message_history.append({
                    'role': 'assistant',
                    'content': turn.get('utterance', '[OK]'),
                })

        return {
            'convo_id': convo_id,
            'category': category,
            'turns': turn_results,
        }

    def _run_self_consistency_convo(
        self,
        convo: dict,
        ensemble_config: dict,
        system_prompt: str,
        temperature: float,
    ) -> dict:
        """Run self-consistency for a single conversation."""
        convo_id = convo['convo_id']
        category = convo.get('category', 'unknown')
        turns_spec = convo.get('turns', [])
        voter_configs = ensemble_config.get('composition', [])

        message_history: list[dict] = []
        turn_results = []

        for turn in turns_spec:
            if turn.get('speaker') == 'user':
                # User turn — add user message then call voters
                utterance = turn['utterance']
                message_history.append({'role': 'user', 'content': utterance})

                votes = []
                total_latency = 0

                for vi, voter_id in enumerate(voter_configs):
                    voter_temp = temperature[vi] if isinstance(temperature, list) else temperature
                    voter_config = {
                        'config_id': voter_id,
                        'provider': ensemble_config.get('provider', 'anthropic'),
                        'model_id': ensemble_config.get('model_id', ''),
                        'temperature': voter_temp,
                    }
                    result = self.client.call_flow_detection(
                        voter_config, system_prompt, list(message_history),
                    )
                    votes.append(result.get('detected_flows', []))
                    total_latency += result.get('latency_ms', 0)

                # Tally votes and score (consistent with bootstrap)
                detected, confidence = tally_votes_multi(votes)
                correct = score_turn_ensemble(
                    category, votes,
                    turn.get('flow', ''),
                    turn.get('candidate_flows'),
                )

                turn_results.append({
                    'turn_num': turn['turn_num'],
                    'utterance': utterance,
                    'flow': turn.get('flow'),
                    'candidate_flows': turn.get('candidate_flows'),
                    'intent': turn.get('intent'),
                    'detected_flows': detected,
                    'correct': correct,
                    'confidence': confidence,
                    'votes': votes,
                    'latency_ms': total_latency,
                })

                message_history.append({
                    'role': 'assistant',
                    'content': f'[Detected flow: {", ".join(detected) if detected else "none"}]',
                })
            else:
                # Agent turn — just add context
                message_history.append({
                    'role': 'assistant',
                    'content': turn.get('utterance', '[OK]'),
                })

        return {
            'convo_id': convo_id,
            'category': category,
            'turns': turn_results,
        }

    def _run_tool_convo(
        self,
        convo: dict,
        config: dict,
        system_prompt: str,
        tools: list[dict],
        tool_flow_map: dict[str, list[str]],
        domain: str = 'hugo',
        fuzzy_evaluator=None,
    ) -> dict:
        """Run tool-calling flow detection on a single conversation.

        Uses v2 scoring (precision/recall with freebie tools). Gold tools
        come from ``target_tools`` keys in the eval data. Turns with
        ``exclude: true`` are passed through but marked as excluded from
        accuracy computation.

        Args:
            tools: Cleaned tool defs (metadata stripped) for the LLM client.
            tool_flow_map: Full tool_name → _flows mapping (kept for compat).
        """
        convo_id = convo['convo_id']
        category = convo.get('category', 'unknown')
        turns_spec = convo.get('turns', [])

        message_history: list[dict] = []
        turn_results = []

        for turn in turns_spec:
            if turn.get('speaker') == 'user':
                utterance = turn['utterance']
                message_history.append({'role': 'user', 'content': utterance})

                result = self.client.call_tool_use(
                    config, system_prompt, list(message_history), tools,
                )

                tool_called = result.get('tool_called') or None
                tool_args = result.get('tool_args') or None
                all_tools = result.get('tools_called') or []

                # Extract predicted tool names from all tool-use blocks
                predicted_tools = (
                    [t['name'] for t in all_tools]
                    if all_tools
                    else ([tool_called] if tool_called else [])
                )

                # Gold tools from target_tools keys
                gold_tools = list(turn.get('target_tools', {}).keys())
                excluded = turn.get('exclude', False)

                # Score with v2 precision/recall framework + param scoring
                gold_target_tools = turn.get('target_tools', {})
                score = score_tool_turn(
                    predicted_tools=predicted_tools,
                    gold_tools=gold_tools,
                    candidate_flows=turn.get('candidate_flows'),
                    domain=domain,
                    predicted_tools_with_args=all_tools,
                    gold_target_tools=gold_target_tools,
                    fuzzy_evaluator=fuzzy_evaluator,
                )

                turn_result = {
                    'turn_num': turn['turn_num'],
                    'utterance': utterance,
                    'flow': turn.get('flow'),
                    'candidate_flows': turn.get('candidate_flows'),
                    'intent': turn.get('intent'),
                    'predicted_tools': score['predicted'],
                    'gold_tools': score['gold'],
                    'all_tools_called': all_tools,
                    'correct': score['correct'],
                    'precision': score['precision'],
                    'recall': score['recall'],
                    'hits': score['hits'],
                    'min_hits': score['min_hits'],
                    'freebies_called': score['freebies_called'],
                    'ambiguity_flagged': score['ambiguity_flagged'],
                    'null_call': score['null_call'],
                    'excluded': excluded,
                    'latency_ms': result.get('latency_ms', 0),
                    'input_tokens': result.get('input_tokens', 0),
                    'output_tokens': result.get('output_tokens', 0),
                }
                # Add param scoring fields if present
                for key in ('param_accuracy', 'matched_params', 'total_scored_params',
                            'param_details', 'correct_with_params'):
                    if key in score:
                        turn_result[key] = score[key]
                turn_results.append(turn_result)

                # Summarise tool calls for conversation history
                called_str = ', '.join(predicted_tools) if predicted_tools else 'none'
                message_history.append({
                    'role': 'assistant',
                    'content': f'[Called tools: {called_str}]',
                })
            else:
                message_history.append({
                    'role': 'assistant',
                    'content': turn.get('utterance', '[OK]'),
                })

        return {
            'convo_id': convo_id,
            'category': category,
            'turns': turn_results,
        }

    @staticmethod
    def _trivial_result(
        turn: dict, utterance: str, skip_reason: str,
        predicted_tool: str | None = None,
    ) -> dict:
        """Build result dict for turns that skip the API call.

        Used by all three skip cases: ambiguous, zero_tool, single_tool.
        """
        gold_tools = list(turn.get('target_tools', {}).keys())
        predicted = [predicted_tool] if predicted_tool else []
        correct = (predicted_tool in gold_tools) if predicted_tool and gold_tools else True
        return {
            'turn_num': turn['turn_num'],
            'utterance': utterance,
            'flow': turn.get('flow'),
            'candidate_flows': turn.get('candidate_flows'),
            'intent': turn.get('intent'),
            'predicted_tools': predicted,
            'gold_tools': gold_tools,
            'correct': correct,
            'skip_reason': skip_reason,
            'tools_offered': 1 if predicted_tool else 0,
            'param_accuracy': None,
            'correct_with_params': None,
            'latency_ms': 0,
            'input_tokens': 0,
            'output_tokens': 0,
        }

    def _run_scoped_tool_convo(
        self,
        convo: dict,
        config: dict,
        system_prompt: str,
        all_tools: list[dict],
        domain: str = 'hugo',
        fuzzy_evaluator=None,
    ) -> dict:
        """Run scoped tool-calling: tools filtered per-turn by gold flow.

        For each user turn, only domain tools scoped to the gold flow are
        presented. Internal component tools are excluded (handled upstream).

        Three ordered skip checks (no API call needed):
        1. **Ambiguous** — turn has candidate_flows → resolved upstream by NLU.
        2. **Zero-tool** — flow maps to 0 domain tools (e.g., chat) → trivially correct.
        3. **Single-tool** — flow maps to 1 domain tool → trivially correct.

        Everything else (2+ tools) calls the API.
        """
        convo_id = convo['convo_id']
        category = convo.get('category', 'unknown')
        turns_spec = convo.get('turns', [])

        message_history: list[dict] = []
        turn_results = []

        for turn in turns_spec:
            if turn.get('speaker') == 'user':
                utterance = turn['utterance']
                gold_flow = turn.get('flow', '')
                message_history.append({'role': 'user', 'content': utterance})

                # 1. Ambiguous turns — skip entirely, handled upstream
                if turn.get('candidate_flows'):
                    turn_results.append(self._trivial_result(
                        turn, utterance, skip_reason='ambiguous',
                    ))
                    message_history.append({
                        'role': 'assistant',
                        'content': '[Ambiguous — resolved upstream]',
                    })
                    continue

                # 2. Scope tools to gold flow
                scoped = self._filter_tools_by_flow(all_tools, gold_flow)
                client_tools = strip_tool_metadata(scoped)

                # 3. Zero-tool flow (e.g., chat) — trivially correct
                if not client_tools:
                    turn_results.append(self._trivial_result(
                        turn, utterance, skip_reason='zero_tool',
                    ))
                    message_history.append({
                        'role': 'assistant',
                        'content': '[Zero tools — no API call needed]',
                    })
                    continue

                # 4. Single-tool flow — trivially correct
                if len(client_tools) == 1:
                    only_tool = client_tools[0]['name']
                    turn_results.append(self._trivial_result(
                        turn, utterance, skip_reason='single_tool',
                        predicted_tool=only_tool,
                    ))
                    message_history.append({
                        'role': 'assistant',
                        'content': f'[Called tool: {only_tool}]',
                    })
                    continue

                result = self.client.call_tool_use(
                    config, system_prompt, list(message_history), client_tools,
                )

                tool_called = result.get('tool_called') or None
                tool_args = result.get('tool_args') or None
                all_tools_called = result.get('tools_called') or []

                predicted_tools = (
                    [t['name'] for t in all_tools_called]
                    if all_tools_called
                    else ([tool_called] if tool_called else [])
                )

                gold_target_tools = turn.get('target_tools', {})
                gold_tools = list(gold_target_tools.keys())

                score = score_tool_turn(
                    predicted_tools=predicted_tools,
                    gold_tools=gold_tools,
                    candidate_flows=turn.get('candidate_flows'),
                    domain=domain,
                    predicted_tools_with_args=all_tools_called,
                    gold_target_tools=gold_target_tools,
                    fuzzy_evaluator=fuzzy_evaluator,
                )

                turn_result = {
                    'turn_num': turn['turn_num'],
                    'utterance': utterance,
                    'flow': turn.get('flow'),
                    'candidate_flows': turn.get('candidate_flows'),
                    'intent': turn.get('intent'),
                    'predicted_tools': score['predicted'],
                    'gold_tools': score['gold'],
                    'all_tools_called': all_tools_called,
                    'correct': score['correct'],
                    'precision': score['precision'],
                    'recall': score['recall'],
                    'hits': score['hits'],
                    'min_hits': score['min_hits'],
                    'freebies_called': score['freebies_called'],
                    'ambiguity_flagged': score['ambiguity_flagged'],
                    'null_call': score['null_call'],
                    'tools_offered': len(client_tools),
                    'latency_ms': result.get('latency_ms', 0),
                    'input_tokens': result.get('input_tokens', 0),
                    'output_tokens': result.get('output_tokens', 0),
                }
                for key in ('param_accuracy', 'matched_params', 'total_scored_params',
                            'param_details', 'correct_with_params'):
                    if key in score:
                        turn_result[key] = score[key]
                turn_results.append(turn_result)

                message_history.append({
                    'role': 'assistant',
                    'content': f'[Called tool: {tool_called}]',
                })
            else:
                message_history.append({
                    'role': 'assistant',
                    'content': turn.get('utterance', '[OK]'),
                })

        return {
            'convo_id': convo_id,
            'category': category,
            'turns': turn_results,
        }

    def _run_intent_convo(
        self, convo: dict, config: dict, system_prompt: str,
    ) -> dict:
        """Run intent classification on a single conversation."""
        convo_id = convo['convo_id']
        category = convo.get('category', 'unknown')
        turns_spec = convo.get('turns', [])

        message_history: list[dict] = []
        turn_results = []

        for turn in turns_spec:
            if turn.get('speaker') == 'user':
                utterance = turn['utterance']
                message_history.append({'role': 'user', 'content': utterance})

                result = self.client.call_flow_detection(
                    config, system_prompt, list(message_history),
                )

                # Parse intent from JSON response
                detected_intent = self._parse_intent(result.get('raw_response', ''))
                expected_intent = turn.get('intent', '')
                candidate_intents = turn.get('candidate_intents')

                correct = score_intent(
                    detected_intent, expected_intent, candidate_intents,
                )

                turn_results.append({
                    'turn_num': turn['turn_num'],
                    'utterance': utterance,
                    'flow': turn.get('flow'),
                    'intent': expected_intent,
                    'candidate_intents': candidate_intents,
                    'detected_intent': detected_intent,
                    'correct': correct,
                    'latency_ms': result.get('latency_ms', 0),
                    'input_tokens': result.get('input_tokens', 0),
                    'output_tokens': result.get('output_tokens', 0),
                })

                message_history.append({
                    'role': 'assistant',
                    'content': f'[Detected intent: {detected_intent}]',
                })
            else:
                message_history.append({
                    'role': 'assistant',
                    'content': turn.get('utterance', '[OK]'),
                })

        return {
            'convo_id': convo_id,
            'category': category,
            'turns': turn_results,
        }

    @staticmethod
    def _parse_intent(raw_response: str) -> str | None:
        """Extract intent field from JSON response."""
        import re
        try:
            cleaned = raw_response.strip()
            cleaned = re.sub(r'^```json\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            obj = json.loads(cleaned)
            intent = obj.get('intent')
            if isinstance(intent, str):
                return intent.strip()
        except (json.JSONDecodeError, AttributeError):
            pass
        # Fallback: try to find JSON object with "intent" key in the response
        match = re.search(r'\{[^{}]*"intent"\s*:\s*"([^"]+)"[^{}]*\}', raw_response)
        if match:
            return match.group(1).strip()
        return None

    def _run_slot_convo(
        self, convo: dict, config: dict, domain: str,
    ) -> dict:
        """Run slot-filling on a single conversation using gold flow."""
        convo_id = convo['convo_id']
        category = convo.get('category', 'unknown')
        turns_spec = convo.get('turns', [])

        message_history: list[dict] = []
        turn_results = []

        for turn in turns_spec:
            if turn.get('speaker') == 'user':
                utterance = turn['utterance']
                message_history.append({'role': 'user', 'content': utterance})

                gold_flow = turn.get('flow', '')
                system_prompt = build_slot_filling_prompt(domain, gold_flow)
                flow_schema = get_flow_slot_schema(domain, gold_flow)

                result = self.client.call_flow_detection(
                    config, system_prompt, list(message_history),
                )

                # Parse slots from JSON response
                detected_slots = self._parse_slots(result.get('raw_response', ''))

                # Score structurally using the funnel scorer (slot stage only)
                score = score_nlu_staged_funnel(
                    detected_intent=turn.get('intent'),  # gold (not scored)
                    detected_flow=gold_flow,              # gold (not scored)
                    detected_slots=detected_slots,
                    expected_intent=turn.get('intent', ''),
                    expected_flow=gold_flow,
                    expected_slots=None,  # no gold slot values yet
                    flow_slot_schema=flow_schema,
                )

                turn_results.append({
                    'turn_num': turn['turn_num'],
                    'utterance': utterance,
                    'flow': gold_flow,
                    'intent': turn.get('intent'),
                    'detected_slots': detected_slots,
                    'slot_precision': score['slots_score']['slot_precision'],
                    'slot_recall': score['slots_score']['slot_recall'],
                    'slot_f1': score['slots_score']['slot_f1'],
                    'valid_keys': score['slots_score'].get('valid_keys', []),
                    'hallucinated_keys': score['slots_score'].get('hallucinated_keys', []),
                    'missing_required': score['slots_score'].get('missing_required', []),
                    'correct': score['slots_score']['slot_f1'] >= 0.5,
                    'latency_ms': result.get('latency_ms', 0),
                    'input_tokens': result.get('input_tokens', 0),
                    'output_tokens': result.get('output_tokens', 0),
                })

                message_history.append({
                    'role': 'assistant',
                    'content': f'[Slots: {detected_slots}]',
                })
            else:
                message_history.append({
                    'role': 'assistant',
                    'content': turn.get('utterance', '[OK]'),
                })

        return {
            'convo_id': convo_id,
            'category': category,
            'turns': turn_results,
        }

    @staticmethod
    def _parse_slots(raw_response: str) -> dict | None:
        """Extract slots dict from JSON response."""
        import re
        try:
            cleaned = raw_response.strip()
            cleaned = re.sub(r'^```json\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            obj = json.loads(cleaned)
            slots = obj.get('slots')
            if isinstance(slots, dict):
                return slots
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    def _run_conversations_parallel(
        self,
        conversations: list[dict],
        config: dict,
        domain: str,
        output_path: Path,
    ) -> list[dict]:
        """Run conversations with ThreadPoolExecutor."""
        if not conversations:
            return []

        results = []
        pbar = tqdm(total=len(conversations), desc='Conversations', unit='convo')
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_convo = {
                executor.submit(
                    self._run_single_convo, convo, config, domain,
                ): convo['convo_id']
                for convo in conversations
            }

            for future in as_completed(future_to_convo):
                convo_id = future_to_convo[future]
                try:
                    result = future.result()
                    results.append(result)
                    self._append_jsonl(output_path, result)
                    log.debug('Completed conversation %s', convo_id)
                except Exception:
                    log.exception('Failed conversation %s', convo_id)
                    results.append({
                        'convo_id': convo_id,
                        'category': 'error',
                        'turns': [],
                        'error': True,
                    })
                pbar.update(1)
        pbar.close()

        return results

    # ── Resume support ────────────────────────────────────────────

    @staticmethod
    def _load_completed(output_path: Path) -> dict[str, dict]:
        """Load previously completed conversations from a JSONL file."""
        completed = {}
        if output_path.exists():
            with open(output_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        completed[record['convo_id']] = record
                    except (json.JSONDecodeError, KeyError):
                        continue
        return completed

    @staticmethod
    def _append_jsonl(output_path: Path, record: dict) -> None:
        """Append a single record to a JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')

    # ── Summary computation ───────────────────────────────────────

    @staticmethod
    def _compute_summary(
        conversations: list[dict],
        run_id: str,
        experiment: str,
        domain: str,
        config_id: str,
        seed: int,
    ) -> dict:
        """Compute run-level summary from conversation results."""
        all_turns = []
        for convo in conversations:
            for turn in convo.get('turns', []):
                if turn.get('excluded', False):
                    continue
                turn_with_meta = {**turn, 'category': convo.get('category', 'unknown')}
                all_turns.append(turn_with_meta)

        if not all_turns:
            return {
                'run_id': run_id,
                'experiment': experiment,
                'domain': domain,
                'config_id': config_id,
                'seed': seed,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'summary': {},
            }

        correct = [t for t in all_turns if t.get('correct')]
        accuracy = len(correct) / len(all_turns) if all_turns else 0.0

        # Accuracy by category
        by_cat: dict[str, list[bool]] = {}
        for t in all_turns:
            cat = t.get('category', 'unknown')
            by_cat.setdefault(cat, []).append(bool(t.get('correct')))
        acc_by_cat = {k: sum(v) / len(v) for k, v in by_cat.items() if v}

        # Accuracy by turn number
        by_turn: dict[int, list[bool]] = {}
        for t in all_turns:
            tn = t.get('turn_num', 1)
            by_turn.setdefault(tn, []).append(bool(t.get('correct')))
        acc_by_turn = {
            f'turn_{k}': sum(v) / len(v)
            for k, v in sorted(by_turn.items()) if v
        }

        # Latency percentiles
        latencies = [t.get('latency_ms', 0) for t in all_turns if t.get('latency_ms')]
        latency_p50 = int(np.percentile(latencies, 50)) if latencies else 0
        latency_p95 = int(np.percentile(latencies, 95)) if latencies else 0

        # Token totals
        input_tokens = sum(t.get('input_tokens', 0) for t in all_turns)
        output_tokens = sum(t.get('output_tokens', 0) for t in all_turns)

        # Failure rate
        errors = sum(1 for c in conversations if c.get('error'))
        failure_rate = errors / len(conversations) if conversations else 0.0

        # Param accuracy (only for turns that have param scoring, excluding None/skipped)
        param_turns = [
            t for t in all_turns
            if t.get('param_accuracy') is not None
        ]
        param_accuracy_mean = (
            round(sum(t['param_accuracy'] for t in param_turns) / len(param_turns), 4)
            if param_turns else None
        )
        cwp_turns = [
            t for t in all_turns
            if t.get('correct_with_params') is not None
        ]
        correct_with_params_rate = (
            round(sum(1 for t in cwp_turns if t['correct_with_params']) / len(cwp_turns), 4)
            if cwp_turns else None
        )

        summary_dict = {
            'accuracy_top1': round(accuracy, 4),
            'accuracy_by_category': {k: round(v, 4) for k, v in acc_by_cat.items()},
            'accuracy_by_turn': {k: round(v, 4) for k, v in acc_by_turn.items()},
            'latency_p50_ms': latency_p50,
            'latency_p95_ms': latency_p95,
            'input_tokens_total': input_tokens,
            'output_tokens_total': output_tokens,
            'failure_rate': round(failure_rate, 4),
        }
        if param_accuracy_mean is not None:
            summary_dict['param_accuracy_mean'] = param_accuracy_mean
        if correct_with_params_rate is not None:
            summary_dict['correct_with_params_rate'] = correct_with_params_rate

        return {
            'run_id': run_id,
            'experiment': experiment,
            'domain': domain,
            'config_id': config_id,
            'seed': seed,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': summary_dict,
        }
