"""Validation tests: eval set params must be consistent with tool manifests.

These are data-integrity tests, not unit tests for scoring logic.
They load the actual eval sets and manifests and check for violations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers.schema_utils import build_param_schema_index, classify_match_method

ROOT = Path(__file__).resolve().parent.parent

DOMAINS = [
    ('hugo', ROOT / 'datasets' / 'hugo' / 'eval_set.json', ROOT / 'tools' / 'tool_manifest_hugo.json'),
    ('dana', ROOT / 'datasets' / 'dana' / 'eval_set.json', ROOT / 'tools' / 'tool_manifest_dana.json'),
]


def _load_json(path: Path) -> list | dict:
    with open(path) as f:
        return json.load(f)


def _iter_target_tool_params(eval_set: list[dict]):
    """Yield (convo_id, turn_num, tool_name, param_name, value) for all target_tools."""
    for convo in eval_set:
        convo_id = convo.get('convo_id', '?')
        for turn in convo.get('turns', []):
            for tool_name, params in turn.get('target_tools', {}).items():
                if not isinstance(params, dict):
                    continue
                for param_name, val in params.items():
                    yield convo_id, turn.get('turn_num', '?'), tool_name, param_name, val


# ── Test 1: No fuzzy wrappers remain ────────────────────────────

@pytest.mark.parametrize('domain,eval_path,manifest_path', DOMAINS, ids=['hugo', 'dana'])
def test_no_fuzzy_wrappers_remain(domain, eval_path, manifest_path):
    """Every param value must be a plain value, not {"value": ..., "fuzzy": true}."""
    eval_set = _load_json(eval_path)
    violations = []
    for cid, tnum, tool, param, val in _iter_target_tool_params(eval_set):
        if isinstance(val, dict) and 'fuzzy' in val:
            violations.append(f'{cid}/t{tnum}: {tool}.{param} = {val}')
    assert violations == [], f'{len(violations)} fuzzy wrappers remain:\n' + '\n'.join(violations[:20])


# ── Test 2: Enum values match manifest ───────────────────────────

@pytest.mark.parametrize('domain,eval_path,manifest_path', DOMAINS, ids=['hugo', 'dana'])
def test_enum_values_match_manifest(domain, eval_path, manifest_path):
    """For enum-constrained params, eval set values must be in the enum."""
    manifest = _load_json(manifest_path)
    eval_set = _load_json(eval_path)
    index = build_param_schema_index(manifest)

    violations = []
    for cid, tnum, tool, param, val in _iter_target_tool_params(eval_set):
        if val is None:
            continue
        schema = index.get((tool, param), {})
        enum_vals = schema.get('enum')
        if enum_vals is None:
            continue
        # For structured values (dicts), skip enum check
        if isinstance(val, dict):
            continue
        if val not in enum_vals:
            violations.append(f'{cid}/t{tnum}: {tool}.{param} = {val!r} not in {enum_vals}')

    assert violations == [], f'{len(violations)} enum violations:\n' + '\n'.join(violations[:30])


# ── Test 3: All target tool names exist in manifest ──────────────

@pytest.mark.parametrize('domain,eval_path,manifest_path', DOMAINS, ids=['hugo', 'dana'])
def test_all_target_tool_names_in_manifest(domain, eval_path, manifest_path):
    """Every tool name in eval set target_tools must exist in the manifest."""
    manifest = _load_json(manifest_path)
    eval_set = _load_json(eval_path)
    manifest_tools = {t['name'] for t in manifest}

    violations = []
    for cid, tnum, tool, param, val in _iter_target_tool_params(eval_set):
        if tool not in manifest_tools:
            violations.append(f'{cid}/t{tnum}: tool {tool!r} not in manifest')

    # Dedupe
    unique = sorted(set(violations))
    assert unique == [], f'{len(unique)} unknown tools:\n' + '\n'.join(unique[:20])


# ── Test 4: All target param names exist in manifest ─────────────

@pytest.mark.parametrize('domain,eval_path,manifest_path', DOMAINS, ids=['hugo', 'dana'])
def test_all_target_param_names_in_manifest(domain, eval_path, manifest_path):
    """Every param name in eval set target_tools must exist in the tool's schema."""
    manifest = _load_json(manifest_path)
    eval_set = _load_json(eval_path)
    index = build_param_schema_index(manifest)
    manifest_tools = {t['name'] for t in manifest}

    violations = []
    for cid, tnum, tool, param, val in _iter_target_tool_params(eval_set):
        if tool not in manifest_tools:
            continue  # Caught by test 3
        if (tool, param) not in index:
            violations.append(f'{cid}/t{tnum}: {tool}.{param} not in schema')

    unique = sorted(set(violations))
    assert unique == [], f'{len(unique)} unknown params:\n' + '\n'.join(unique[:20])


# ── Test 5: classify_match_method unit test ──────────────────────

class TestClassifyMatchMethod:
    def test_enum_is_exact(self):
        assert classify_match_method({'type': 'string', 'enum': ['a', 'b']}) == 'exact'

    def test_boolean_is_exact(self):
        assert classify_match_method({'type': 'boolean'}) == 'exact'

    def test_integer_is_exact(self):
        assert classify_match_method({'type': 'integer'}) == 'exact'

    def test_number_is_exact(self):
        assert classify_match_method({'type': 'number'}) == 'exact'

    def test_object_is_structured(self):
        assert classify_match_method({'type': 'object'}) == 'structured'

    def test_array_is_structured(self):
        assert classify_match_method({'type': 'array', 'items': {'type': 'string'}}) == 'structured'

    def test_plain_string_is_fuzzy(self):
        assert classify_match_method({'type': 'string'}) == 'fuzzy'

    def test_empty_schema_is_fuzzy(self):
        assert classify_match_method({}) == 'fuzzy'

    def test_string_with_description_is_fuzzy(self):
        assert classify_match_method({'type': 'string', 'description': 'anything'}) == 'fuzzy'
