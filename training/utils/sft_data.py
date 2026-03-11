"""SFT data generation from ensemble results or gold labels.

Converts evaluation data into ``(messages, label)`` pairs suitable for
supervised fine-tuning with HuggingFace chat format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.stages import PipelineStage


@dataclass
class SFTExample:
    """One SFT training example."""

    messages: list[dict[str, str]]
    label: str  # The assistant's expected response
    stage: str
    domain: str
    convo_id: str
    turn_num: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_chat_format(self) -> dict:
        """Return HuggingFace chat-format dict with messages including the label."""
        return {
            'messages': self.messages + [{'role': 'assistant', 'content': self.label}],
            'stage': self.stage,
            'domain': self.domain,
            'convo_id': self.convo_id,
            'turn_num': self.turn_num,
            'metadata': self.metadata,
        }


class SFTDataGenerator:
    """Generate SFT examples from eval sets.

    Supports two modes:
    - ``generate_from_gold_labels``: Uses gold labels directly from the eval set.
    - ``generate_from_ensemble_results``: Uses ensemble outputs as labels,
      filtered by confidence threshold.
    """

    def __init__(self, stage: PipelineStage, domain: str, **kwargs: Any):
        self.stage = stage
        self.domain = domain
        self.kwargs = kwargs

    def generate_from_gold_labels(
        self,
        eval_set: list[dict],
    ) -> list[SFTExample]:
        """Generate SFT examples using gold labels from the eval set.

        For each user turn in each conversation, builds the message history
        and uses the gold label as the assistant response.
        """
        examples: list[SFTExample] = []

        for convo in eval_set:
            convo_id = convo.get('convo_id', '')
            message_history: list[dict[str, str]] = []

            for turn in convo.get('turns', []):
                if turn.get('speaker') == 'user':
                    utterance = turn['utterance']
                    message_history.append({'role': 'user', 'content': utterance})

                    label = self._build_gold_label(turn)
                    if label is None:
                        # No valid label for this stage/turn
                        message_history.append({
                            'role': 'assistant',
                            'content': '[skipped]',
                        })
                        continue

                    # Build system prompt for this turn
                    system_prompt = self.stage.build_prompt(
                        self.domain, turn, convo, **self.kwargs
                    )

                    messages = (
                        [{'role': 'system', 'content': system_prompt}]
                        + list(message_history)
                    )

                    examples.append(SFTExample(
                        messages=messages,
                        label=label,
                        stage=self.stage.name,
                        domain=self.domain,
                        convo_id=convo_id,
                        turn_num=turn.get('turn_num', 0),
                    ))

                    # Add synthetic assistant response for context continuity
                    message_history.append({
                        'role': 'assistant',
                        'content': label,
                    })
                else:
                    # Agent turn — preserve as context
                    message_history.append({
                        'role': 'assistant',
                        'content': turn.get('utterance', '[OK]'),
                    })

        return examples

    def generate_from_ensemble_results(
        self,
        ensemble_results: list[dict],
        eval_set: list[dict],
        confidence_threshold: float = 0.7,
    ) -> list[SFTExample]:
        """Generate SFT examples using ensemble outputs as labels.

        Only includes turns where the ensemble was correct and confident
        above the given threshold.

        Args:
            ensemble_results: List of per-conversation result dicts from
                ensemble runs (with ``turns[].detected_flows``, ``turns[].correct``,
                ``turns[].confidence``).
            eval_set: The original eval set (for building message histories).
            confidence_threshold: Minimum ensemble confidence to include.
        """
        # Index ensemble results by convo_id
        result_by_id: dict[str, dict] = {}
        for res in ensemble_results:
            cid = res.get('convo_id', '')
            if cid:
                result_by_id[cid] = res

        examples: list[SFTExample] = []

        for convo in eval_set:
            convo_id = convo.get('convo_id', '')
            ens_result = result_by_id.get(convo_id)
            if not ens_result:
                continue

            ens_turns = {t['turn_num']: t for t in ens_result.get('turns', [])}
            message_history: list[dict[str, str]] = []

            for turn in convo.get('turns', []):
                if turn.get('speaker') == 'user':
                    utterance = turn['utterance']
                    turn_num = turn.get('turn_num', 0)
                    message_history.append({'role': 'user', 'content': utterance})

                    ens_turn = ens_turns.get(turn_num, {})
                    correct = ens_turn.get('correct', False)
                    confidence = ens_turn.get('confidence', 0.0)

                    if not correct or confidence < confidence_threshold:
                        message_history.append({
                            'role': 'assistant',
                            'content': '[skipped]',
                        })
                        continue

                    label = self._build_ensemble_label(ens_turn)
                    if label is None:
                        message_history.append({
                            'role': 'assistant',
                            'content': '[skipped]',
                        })
                        continue

                    system_prompt = self.stage.build_prompt(
                        self.domain, turn, convo, **self.kwargs
                    )
                    messages = (
                        [{'role': 'system', 'content': system_prompt}]
                        + list(message_history)
                    )

                    examples.append(SFTExample(
                        messages=messages,
                        label=label,
                        stage=self.stage.name,
                        domain=self.domain,
                        convo_id=convo_id,
                        turn_num=turn_num,
                        metadata={
                            'confidence': confidence,
                            'ensemble_correct': correct,
                        },
                    ))

                    message_history.append({
                        'role': 'assistant',
                        'content': label,
                    })
                else:
                    message_history.append({
                        'role': 'assistant',
                        'content': turn.get('utterance', '[OK]'),
                    })

        return examples

    def _build_gold_label(self, turn: dict) -> str | None:
        """Build the expected assistant response from gold labels."""
        stage_name = self.stage.name

        if stage_name == 'intent':
            intent = turn.get('intent')
            if not intent:
                return None
            return json.dumps({'reasoning': '', 'intent': intent})

        if stage_name == 'flow':
            flow = turn.get('flow')
            candidate_flows = turn.get('candidate_flows')
            if candidate_flows:
                flows = candidate_flows
            elif flow:
                flows = [flow]
            else:
                return None
            return json.dumps({'reasoning': '', 'flows': flows})

        if stage_name == 'tool_selection':
            target_tools = turn.get('target_tools', {})
            if not target_tools:
                return None
            # Label is the list of tool names
            return json.dumps(list(target_tools.keys()))

        if stage_name == 'param_extraction':
            target_tools = turn.get('target_tools', {})
            if not target_tools:
                return None
            tools_list = []
            for tool_name, params in target_tools.items():
                if isinstance(params, dict):
                    tools_list.append({'name': tool_name, 'params': params})
            return json.dumps({'tools': tools_list})

        # For composed stages, use the last sub-stage's label format
        if '+' in stage_name:
            # Composed: try param_extraction label if available, else tool_selection
            target_tools = turn.get('target_tools', {})
            if not target_tools:
                return None
            tools_list = []
            for tool_name, params in target_tools.items():
                if isinstance(params, dict):
                    tools_list.append({'name': tool_name, 'params': params})
            return json.dumps({'tools': tools_list})

        return None

    def _build_ensemble_label(self, ens_turn: dict) -> str | None:
        """Build the assistant response from an ensemble turn result."""
        stage_name = self.stage.name

        if stage_name == 'flow':
            detected_flows = ens_turn.get('detected_flows', [])
            if not detected_flows:
                return None
            return json.dumps({'reasoning': '', 'flows': detected_flows})

        if stage_name == 'intent':
            detected_intent = ens_turn.get('detected_intent')
            if not detected_intent:
                return None
            return json.dumps({'reasoning': '', 'intent': detected_intent})

        if stage_name == 'tool_selection':
            predicted_tools = ens_turn.get('predicted_tools', [])
            if not predicted_tools:
                return None
            return json.dumps(predicted_tools)

        return None

    @staticmethod
    def save_jsonl(examples: list[SFTExample], path: str | Path) -> None:
        """Save SFT examples to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex.to_chat_format()) + '\n')

    @staticmethod
    def load_jsonl(path: str | Path) -> list[dict]:
        """Load SFT examples from a JSONL file."""
        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples
