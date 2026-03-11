"""Stage abstraction for the NLU training pipeline.

Each stage wraps an existing prompt builder and scoring function from the
handling-ambiguity codebase, providing a uniform interface for SFT data
generation and RL reward computation.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from helpers.scoring import (
    score_intent,
    score_tool_params,
    score_tool_turn,
    score_turn,
)
from prompts.flow_detection import build_flow_detection_prompt
from prompts.intent_classification import build_intent_classification_prompt
from prompts.param_extraction import (
    build_batch_param_extraction_prompt,
    build_param_extraction_prompt,
)
from prompts.tool_calling import build_tool_calling_prompt, strip_tool_metadata


# ── Data classes ────────────────────────────────────────────────


@dataclass
class StageOutput:
    """Structured output from parsing a model response."""

    raw: str
    parsed: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class StageReward:
    """Reward signal from scoring a stage output."""

    reward: float  # In [0, 1]
    details: dict[str, Any] = field(default_factory=dict)


# ── JSON parsing helpers ────────────────────────────────────────


def _clean_json(raw: str) -> str:
    """Strip markdown code fences from a JSON response."""
    cleaned = raw.strip()
    cleaned = re.sub(r'^```json\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned


def _parse_json_safe(raw: str) -> dict | None:
    """Parse JSON from a potentially fenced response string."""
    try:
        return json.loads(_clean_json(raw))
    except (json.JSONDecodeError, AttributeError):
        return None


# ── Abstract base class ────────────────────────────────────────


class PipelineStage(ABC):
    """Abstract base class for a single pipeline stage."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def build_prompt(self, domain: str, turn: dict, convo: dict, **kwargs) -> str:
        """Build the system prompt for this stage."""
        ...

    def build_messages(
        self,
        domain: str,
        turn: dict,
        convo: dict,
        message_history: list[dict],
        **kwargs,
    ) -> list[dict]:
        """Construct the full chat message list.

        Default implementation: system prompt + existing message_history.
        """
        system_prompt = self.build_prompt(domain, turn, convo, **kwargs)
        return [{'role': 'system', 'content': system_prompt}] + list(message_history)

    @abstractmethod
    def parse_response(self, raw_response: str, turn: dict, **kwargs) -> StageOutput:
        """Parse the raw model output into structured form."""
        ...

    @abstractmethod
    def compute_reward(
        self, output: StageOutput, turn: dict, convo: dict, **kwargs
    ) -> StageReward:
        """Score the output against gold labels. Returns reward in [0, 1]."""
        ...

    def get_tool_specs(
        self, domain: str, turn: dict, convo: dict, **kwargs
    ) -> list[dict] | None:
        """Return tool specs for stages that use native tool calling."""
        return None


# ── Concrete stages ─────────────────────────────────────────────


class IntentStage(PipelineStage):
    """Intent classification stage."""

    @property
    def name(self) -> str:
        return 'intent'

    def build_prompt(self, domain: str, turn: dict, convo: dict, **kwargs) -> str:
        return build_intent_classification_prompt(domain)

    def parse_response(self, raw_response: str, turn: dict, **kwargs) -> StageOutput:
        obj = _parse_json_safe(raw_response)
        if obj and isinstance(obj.get('intent'), str):
            return StageOutput(raw=raw_response, parsed={'intent': obj['intent'].strip()})
        # Fallback regex
        match = re.search(r'\{[^{}]*"intent"\s*:\s*"([^"]+)"[^{}]*\}', raw_response)
        if match:
            return StageOutput(raw=raw_response, parsed={'intent': match.group(1).strip()})
        return StageOutput(raw=raw_response, error='failed to parse intent')

    def compute_reward(
        self, output: StageOutput, turn: dict, convo: dict, **kwargs
    ) -> StageReward:
        detected = output.parsed.get('intent')
        expected = turn.get('intent')
        candidate_intents = turn.get('candidate_intents')
        correct = score_intent(detected, expected, candidate_intents)
        return StageReward(reward=1.0 if correct else 0.0, details={'correct': correct})


class FlowStage(PipelineStage):
    """Flow detection stage."""

    @property
    def name(self) -> str:
        return 'flow'

    @staticmethod
    def _get_turn_intents(turn: dict) -> list[str]:
        """Determine intent(s) for scoping candidate flows."""
        if turn.get('candidate_intents'):
            return list(dict.fromkeys(turn['candidate_intents']))
        if turn.get('intent'):
            return [turn['intent']]
        return []

    def build_prompt(self, domain: str, turn: dict, convo: dict, **kwargs) -> str:
        intents = self._get_turn_intents(turn)
        return build_flow_detection_prompt(domain, intents or None)

    def parse_response(self, raw_response: str, turn: dict, **kwargs) -> StageOutput:
        obj = _parse_json_safe(raw_response)
        if obj and isinstance(obj.get('flows'), list):
            flows = [f.strip() for f in obj['flows'] if isinstance(f, str)]
            return StageOutput(
                raw=raw_response,
                parsed={'flows': flows, 'reasoning': obj.get('reasoning', '')},
            )
        return StageOutput(raw=raw_response, parsed={'flows': []}, error='failed to parse flows')

    def compute_reward(
        self, output: StageOutput, turn: dict, convo: dict, **kwargs
    ) -> StageReward:
        detected_flows = output.parsed.get('flows', [])
        category = convo.get('category', 'unknown')
        expected_flow = turn.get('flow', '')
        candidate_flows = turn.get('candidate_flows')
        correct = score_turn(category, detected_flows, expected_flow, candidate_flows)
        return StageReward(reward=1.0 if correct else 0.0, details={'correct': correct})


class ToolSelectionStage(PipelineStage):
    """Tool selection stage (native tool calling)."""

    @property
    def name(self) -> str:
        return 'tool_selection'

    def build_prompt(self, domain: str, turn: dict, convo: dict, **kwargs) -> str:
        context = convo.get('context')
        mode = kwargs.get('mode', 'tool')
        return build_tool_calling_prompt(domain, context, mode)

    def build_messages(
        self,
        domain: str,
        turn: dict,
        convo: dict,
        message_history: list[dict],
        **kwargs,
    ) -> list[dict]:
        system_prompt = self.build_prompt(domain, turn, convo, **kwargs)
        return [{'role': 'system', 'content': system_prompt}] + list(message_history)

    def get_tool_specs(
        self, domain: str, turn: dict, convo: dict, **kwargs
    ) -> list[dict] | None:
        tools = kwargs.get('tools')
        if tools is not None:
            return strip_tool_metadata(tools)
        return None

    def parse_response(self, raw_response: str, turn: dict, **kwargs) -> StageOutput:
        """Parse tool calls from the response.

        For native tool calling, raw_response may be a JSON string of tool calls
        or a pre-parsed list. We handle both cases.
        """
        # If raw_response is already a list of tool calls (from sglang tool_use)
        if isinstance(raw_response, list):
            tool_names = [t.get('name', '') for t in raw_response if isinstance(t, dict)]
            return StageOutput(
                raw=str(raw_response),
                parsed={'predicted_tools': tool_names, 'all_tools_called': raw_response},
            )

        # Try to parse as JSON array of tool calls
        obj = _parse_json_safe(raw_response)
        if isinstance(obj, list):
            tool_names = [t.get('name', '') for t in obj if isinstance(t, dict)]
            return StageOutput(
                raw=raw_response,
                parsed={'predicted_tools': tool_names, 'all_tools_called': obj},
            )
        if isinstance(obj, dict):
            # Single tool call
            name = obj.get('name', '')
            return StageOutput(
                raw=raw_response,
                parsed={'predicted_tools': [name] if name else [], 'all_tools_called': [obj]},
            )

        return StageOutput(
            raw=raw_response, parsed={'predicted_tools': [], 'all_tools_called': []},
            error='failed to parse tool calls',
        )

    def compute_reward(
        self, output: StageOutput, turn: dict, convo: dict, **kwargs
    ) -> StageReward:
        predicted_tools = output.parsed.get('predicted_tools', [])
        gold_tools = list((turn.get('target_tools') or {}).keys())
        candidate_flows = turn.get('candidate_flows')
        domain = kwargs.get('domain', 'hugo')
        result = score_tool_turn(predicted_tools, gold_tools, candidate_flows, domain)
        return StageReward(
            reward=1.0 if result['correct'] else 0.0,
            details=result,
        )


class ParamExtractionStage(PipelineStage):
    """Parameter extraction stage."""

    @property
    def name(self) -> str:
        return 'param_extraction'

    def build_prompt(self, domain: str, turn: dict, convo: dict, **kwargs) -> str:
        """Build param extraction prompt.

        When tool_name and tool_schema are provided in kwargs, builds a
        single-tool prompt. Otherwise builds a batch prompt from target_tools.
        """
        context = convo.get('context')
        tool_name = kwargs.get('tool_name')
        tool_schema = kwargs.get('tool_schema')

        if tool_name and tool_schema:
            return build_param_extraction_prompt(domain, tool_name, tool_schema, context)

        # Batch mode: build from turn's target_tools
        tool_lookup = kwargs.get('tool_lookup', {})
        target_tools = turn.get('target_tools', {})
        tools_with_schemas = []
        for tname in target_tools:
            tdef = tool_lookup.get(tname, {})
            tools_with_schemas.append({'name': tname, 'schema': tdef})
        return build_batch_param_extraction_prompt(domain, tools_with_schemas, context)

    def parse_response(self, raw_response: str, turn: dict, **kwargs) -> StageOutput:
        obj = _parse_json_safe(raw_response)
        if obj is None:
            return StageOutput(raw=raw_response, error='failed to parse params JSON')

        # List of tool calls: [{"name": ..., "args": {...}}, ...]
        if isinstance(obj, list):
            predicted = []
            for entry in obj:
                if isinstance(entry, dict):
                    predicted.append({
                        'name': entry.get('name', ''),
                        'args': entry.get('args') or entry.get('params') or {},
                    })
            return StageOutput(
                raw=raw_response, parsed={'predicted_tools_with_args': predicted}
            )

        # Single-tool format: {"reasoning": ..., "params": {...}}
        if isinstance(obj.get('params'), dict):
            tool_name = kwargs.get('tool_name', '')
            return StageOutput(
                raw=raw_response,
                parsed={
                    'predicted_tools_with_args': [
                        {'name': tool_name, 'args': obj['params']}
                    ],
                },
            )

        # Batch format: {"tools": [{"name": ..., "params": {...}}, ...]}
        if isinstance(obj.get('tools'), list):
            predicted = []
            for entry in obj['tools']:
                if isinstance(entry, dict):
                    predicted.append({
                        'name': entry.get('name', ''),
                        'args': entry.get('params') or {},
                    })
            return StageOutput(
                raw=raw_response, parsed={'predicted_tools_with_args': predicted}
            )

        return StageOutput(raw=raw_response, error='unrecognised param format')

    def compute_reward(
        self, output: StageOutput, turn: dict, convo: dict, **kwargs
    ) -> StageReward:
        predicted = output.parsed.get('predicted_tools_with_args', [])
        gold_target_tools = turn.get('target_tools', {})
        fuzzy_evaluator = kwargs.get('fuzzy_evaluator')
        param_schema_index = kwargs.get('param_schema_index')
        result = score_tool_params(predicted, gold_target_tools, fuzzy_evaluator, param_schema_index)
        return StageReward(
            reward=result['param_accuracy'],
            details=result,
        )


# ── Composed stage ──────────────────────────────────────────────


class ComposedStage(PipelineStage):
    """Wraps multiple stages into a single trainable unit.

    The composed stage uses the *last* sub-stage's prompt (most downstream)
    and combines reward signals with configurable weights.
    """

    def __init__(
        self, stages: list[PipelineStage], weights: list[float] | None = None
    ):
        self.stages = stages
        if weights is None:
            weights = [1.0 / len(stages)] * len(stages)
        if len(weights) != len(stages):
            raise ValueError(
                f'weights length ({len(weights)}) must match stages length ({len(stages)})'
            )
        self.weights = weights

    @property
    def name(self) -> str:
        return '+'.join(s.name for s in self.stages)

    def build_prompt(self, domain: str, turn: dict, convo: dict, **kwargs) -> str:
        # Use the last (most downstream) stage's prompt
        return self.stages[-1].build_prompt(domain, turn, convo, **kwargs)

    def build_messages(
        self,
        domain: str,
        turn: dict,
        convo: dict,
        message_history: list[dict],
        **kwargs,
    ) -> list[dict]:
        return self.stages[-1].build_messages(
            domain, turn, convo, message_history, **kwargs
        )

    def parse_response(self, raw_response: str, turn: dict, **kwargs) -> StageOutput:
        """Run each sub-stage's parser and merge results."""
        merged_parsed: dict[str, Any] = {}
        errors: list[str] = []
        for stage in self.stages:
            sub_output = stage.parse_response(raw_response, turn, **kwargs)
            merged_parsed.update(sub_output.parsed)
            if sub_output.error:
                errors.append(f'{stage.name}: {sub_output.error}')
        return StageOutput(
            raw=raw_response,
            parsed=merged_parsed,
            error='; '.join(errors) if errors else None,
        )

    def compute_reward(
        self, output: StageOutput, turn: dict, convo: dict, **kwargs
    ) -> StageReward:
        """Return weighted sum of sub-stage rewards."""
        total = 0.0
        all_details: dict[str, Any] = {}
        for stage, weight in zip(self.stages, self.weights):
            # Build a sub-output containing only this stage's parsed data
            sub_output = stage.parse_response(output.raw, turn, **kwargs)
            reward = stage.compute_reward(sub_output, turn, convo, **kwargs)
            total += weight * reward.reward
            all_details[stage.name] = {
                'reward': reward.reward,
                'weight': weight,
                'details': reward.details,
            }
        return StageReward(reward=total, details=all_details)

    def get_tool_specs(
        self, domain: str, turn: dict, convo: dict, **kwargs
    ) -> list[dict] | None:
        """Return tool specs from whichever sub-stage provides them."""
        for stage in self.stages:
            specs = stage.get_tool_specs(domain, turn, convo, **kwargs)
            if specs is not None:
                return specs
        return None


# ── Stage registry ──────────────────────────────────────────────

STAGE_REGISTRY: dict[str, type[PipelineStage]] = {
    'intent': IntentStage,
    'flow': FlowStage,
    'tool_selection': ToolSelectionStage,
    'param_extraction': ParamExtractionStage,
}
