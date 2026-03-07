"""Unit tests for score_tool_params() using schema-driven match methods."""

from __future__ import annotations

import pytest

from helpers.scoring import score_tool_params


def _schema_index(**entries):
    """Helper to build a param_schema_index from keyword args.

    Usage: _schema_index(revise_content__focus={'type': 'string', 'enum': [...]})
    Keys use double-underscore to separate tool and param names.
    """
    index = {}
    for key, schema in entries.items():
        tool, param = key.split('__', 1)
        index[(tool, param)] = schema
    return index


class TestExactEnumMatch:
    def test_match(self):
        result = score_tool_params(
            predicted_tools_with_args=[{'name': 'manage_schedule', 'args': {'action': 'schedule'}}],
            gold_target_tools={'manage_schedule': {'action': 'schedule'}},
            param_schema_index=_schema_index(
                manage_schedule__action={'type': 'string', 'enum': ['view', 'reschedule', 'cancel', 'schedule']},
            ),
        )
        assert result['param_accuracy'] == 1.0
        assert result['matched_params'] == 1

    def test_mismatch(self):
        result = score_tool_params(
            predicted_tools_with_args=[{'name': 'manage_schedule', 'args': {'action': 'view'}}],
            gold_target_tools={'manage_schedule': {'action': 'schedule'}},
            param_schema_index=_schema_index(
                manage_schedule__action={'type': 'string', 'enum': ['view', 'reschedule', 'cancel', 'schedule']},
            ),
        )
        assert result['param_accuracy'] == 0.0
        assert result['matched_params'] == 0


class TestFuzzyStringMatch:
    def test_fuzzy_match_with_evaluator(self):
        mock_fuzzy = lambda gold, pred: True  # always matches
        result = score_tool_params(
            predicted_tools_with_args=[{'name': 'revise_content', 'args': {'instructions': 'reorder sections'}}],
            gold_target_tools={'revise_content': {'instructions': 'reorder from Q/K/V into scaled dot-product'}},
            param_schema_index=_schema_index(
                revise_content__instructions={'type': 'string', 'description': 'Revision direction'},
            ),
            fuzzy_evaluator=mock_fuzzy,
        )
        assert result['param_accuracy'] == 1.0
        assert result['param_details'][0]['method'] == 'fuzzy'

    def test_fuzzy_fallback_case_insensitive(self):
        result = score_tool_params(
            predicted_tools_with_args=[{'name': 'revise_content', 'args': {'instructions': 'REORDER'}}],
            gold_target_tools={'revise_content': {'instructions': 'reorder'}},
            param_schema_index=_schema_index(
                revise_content__instructions={'type': 'string'},
            ),
        )
        assert result['param_accuracy'] == 1.0


class TestStructuredObjectMatch:
    def test_structured_match(self):
        result = score_tool_params(
            predicted_tools_with_args=[{'name': 'manage_schedule', 'args': {'datetime': {'day': 'monday', 'time': '14:00'}}}],
            gold_target_tools={'manage_schedule': {'datetime': {'day': 'monday', 'time': '14:00'}}},
            param_schema_index=_schema_index(
                manage_schedule__datetime={'type': 'object'},
            ),
        )
        assert result['param_accuracy'] == 1.0
        assert result['param_details'][0]['method'] == 'structured'

    def test_structured_mismatch(self):
        result = score_tool_params(
            predicted_tools_with_args=[{'name': 'manage_schedule', 'args': {'datetime': {'day': 'tuesday', 'time': '14:00'}}}],
            gold_target_tools={'manage_schedule': {'datetime': {'day': 'monday', 'time': '14:00'}}},
            param_schema_index=_schema_index(
                manage_schedule__datetime={'type': 'object'},
            ),
        )
        assert result['param_accuracy'] == 0.0


class TestNullGoldSkipped:
    def test_null_gold_not_scored(self):
        result = score_tool_params(
            predicted_tools_with_args=[{'name': 'revise_content', 'args': {'source_content': 'something'}}],
            gold_target_tools={'revise_content': {'source_content': None, 'instructions': 'reorder'}},
            param_schema_index=_schema_index(
                revise_content__source_content={'type': 'string'},
                revise_content__instructions={'type': 'string'},
            ),
        )
        # Only instructions should be scored (source_content is null gold)
        assert result['total_scored_params'] == 1


class TestMissingPredictedParam:
    def test_missing_param(self):
        result = score_tool_params(
            predicted_tools_with_args=[{'name': 'revise_content', 'args': {}}],
            gold_target_tools={'revise_content': {'instructions': 'reorder'}},
            param_schema_index=_schema_index(
                revise_content__instructions={'type': 'string'},
            ),
        )
        assert result['param_accuracy'] == 0.0
        assert result['param_details'][0]['method'] == 'missing'


class TestMissingPredictedTool:
    def test_missing_tool(self):
        result = score_tool_params(
            predicted_tools_with_args=[{'name': 'other_tool', 'args': {'x': 1}}],
            gold_target_tools={'revise_content': {'instructions': 'reorder'}},
            param_schema_index=_schema_index(
                revise_content__instructions={'type': 'string'},
            ),
        )
        assert result['param_accuracy'] == 0.0
        assert result['param_details'][0]['method'] == 'missing_tool'


class TestBackwardCompatNoSchema:
    """When param_schema_index is None, all string params default to fuzzy."""
    def test_no_schema_falls_back_to_fuzzy(self):
        result = score_tool_params(
            predicted_tools_with_args=[{'name': 'revise_content', 'args': {'instructions': 'reorder'}}],
            gold_target_tools={'revise_content': {'instructions': 'reorder'}},
        )
        assert result['param_accuracy'] == 1.0
        assert result['param_details'][0]['method'] == 'fuzzy'
