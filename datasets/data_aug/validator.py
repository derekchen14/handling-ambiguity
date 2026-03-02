"""Post-generation quality checks for conversation data."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

log = logging.getLogger(__name__)


# ── Check Helpers ─────────────────────────────────────────────────────

def _user_utterances(convo: dict) -> list[str]:
    """Extract user utterances from a conversation."""
    return [
        t['utterance']
        for t in convo['turns']
        if t.get('speaker') == 'user' and t.get('utterance', '').strip()
    ]


def _agent_utterances(convo: dict) -> list[str]:
    """Extract agent utterances from a conversation."""
    return [
        t['utterance']
        for t in convo['turns']
        if t.get('speaker') == 'agent' and t.get('utterance', '').strip()
    ]


# Characters that should not appear in generated text (non-ASCII punctuation).
_BAD_CHARS = {
    '\u2014': 'em dash',
    '\u2013': 'en dash',
    '\u2018': 'left single quote',
    '\u2019': 'right single quote',
    '\u201c': 'left double quote',
    '\u201d': 'right double quote',
    '\u2026': 'ellipsis',
    '\u00a0': 'non-breaking space',
}


# Flow names that are common English words — can't reasonably be avoided
# in natural user utterances. Only flag truly distinctive flow names.
COMMON_WORD_FLOWS = {
    # Verbs used in everyday speech
    'add', 'check', 'compare', 'confirm', 'create', 'define', 'delete',
    'describe', 'design', 'expand', 'explain', 'export', 'fill', 'find',
    'format', 'insert', 'inspect', 'join', 'merge', 'outline', 'plot',
    'preview', 'recall', 'recommend', 'refine', 'reject', 'release',
    'replace', 'reshape', 'retrieve', 'schedule', 'search', 'split',
    'store', 'study', 'style', 'suggest', 'survey', 'tone', 'trend',
    'undo', 'update', 'validate', 'view', 'write',
    # Nouns used in everyday speech
    'audit', 'browse', 'calculate', 'cancel', 'chat', 'approve',
    'blank', 'dismiss', 'endorse', 'peek', 'pivot', 'query', 'reference',
    'remember', 'segment', 'summarize',
}


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two strings (token-level)."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# ── Individual Checks ─────────────────────────────────────────────────

def check_encoding(convo: dict) -> list[str]:
    """Check 0: No non-ASCII punctuation in utterances, responses, or scenario."""
    issues = []
    all_texts = _user_utterances(convo) + _agent_utterances(convo)
    all_texts.append(convo.get('scenario', ''))
    for text in all_texts:
        for char, name in _BAD_CHARS.items():
            if char in text:
                issues.append(
                    f'Non-ASCII char ({name}, U+{ord(char):04X}) in: '
                    f'"{text[:60]}"'
                )
    return issues


def check_format(convo: dict) -> list[str]:
    """Check 1: All 3 turns present, correct speakers and fields."""
    issues = []
    turns = convo.get('turns', [])

    if len(turns) != 3:
        issues.append(f'Expected 3 turns, got {len(turns)}')
        return issues

    # Turn 1: user with flow
    t1 = turns[0]
    if t1.get('speaker') != 'user':
        issues.append(f'Turn 1: expected speaker "user", got "{t1.get("speaker")}"')
    if not t1.get('utterance', '').strip():
        issues.append('Turn 1: missing utterance')
    if 'flow' not in t1:
        issues.append('Turn 1: missing flow')

    # Turn 2: agent
    t2 = turns[1]
    if t2.get('speaker') != 'agent':
        issues.append(f'Turn 2: expected speaker "agent", got "{t2.get("speaker")}"')
    if not t2.get('utterance', '').strip():
        issues.append('Turn 2: missing utterance')

    # Turn 3: user with flow
    t3 = turns[2]
    if t3.get('speaker') != 'user':
        issues.append(f'Turn 3: expected speaker "user", got "{t3.get("speaker")}"')
    if not t3.get('utterance', '').strip():
        issues.append('Turn 3: missing utterance')
    if 'flow' not in t3:
        issues.append('Turn 3: missing flow')

    return issues


def check_flow_name_leakage(convo: dict, flow_catalog: dict) -> list[str]:
    """Check 2: User utterances must NOT contain distinctive flow names as words.

    Skips flow names that are common English words (too many false positives).
    Only flags multi-word or domain-specific names like 'brainstorm', 'dedupe',
    'datatype', 'interpolate', 'syndicate', etc.
    """
    issues = []
    # Only check distinctive flow names (not common English words)
    flow_names = set(flow_catalog.keys()) - COMMON_WORD_FLOWS

    for utterance in _user_utterances(convo):
        for name in flow_names:
            pattern = rf'\b{re.escape(name)}\b'
            if re.search(pattern, utterance, re.IGNORECASE):
                issues.append(f'Flow name "{name}" found in utterance: "{utterance[:80]}"')

    return issues


def check_length(convo: dict) -> list[str]:
    """Check 3: User utterances >1 char, max 100 words; agent utterances >1 char, max 80 words."""
    issues = []

    for utterance in _user_utterances(convo):
        if len(utterance.strip()) <= 1:
            issues.append(f'User utterance too short: "{utterance[:60]}"')
        elif len(utterance.split()) > 100:
            issues.append(f'User utterance too long ({len(utterance.split())} words): "{utterance[:60]}"')

    for response in _agent_utterances(convo):
        if len(response.strip()) <= 1:
            issues.append(f'Agent utterance too short: "{response[:60]}"')
        elif len(response.split()) > 80:
            issues.append(f'Agent utterance too long ({len(response.split())} words): "{response[:60]}"')

    return issues


def check_uniqueness(convos: list[dict], threshold: float = 0.6) -> list[str]:
    """Check 4: No two user utterances share >threshold Jaccard overlap."""
    issues = []
    all_utterances = []

    for convo in convos:
        for utterance in _user_utterances(convo):
            all_utterances.append((convo['convo_id'], utterance))

    for i in range(len(all_utterances)):
        for j in range(i + 1, len(all_utterances)):
            id_a, utt_a = all_utterances[i]
            id_b, utt_b = all_utterances[j]
            sim = _jaccard_similarity(utt_a, utt_b)
            if sim > threshold:
                issues.append(
                    f'High similarity ({sim:.2f}) between {id_a} and {id_b}: '
                    f'"{utt_a[:50]}" vs "{utt_b[:50]}"'
                )

    return issues


def check_ambiguity_sanity(convo: dict, flow_catalog: dict) -> list[str]:
    """Check 5 (Cat C): Turn 1 shouldn't contain keywords that obviously disambiguate."""
    issues = []

    if convo.get('category') != 'ambiguous_first':
        return issues

    t1 = convo['turns'][0]
    utterance = t1.get('utterance', '')
    candidate_flows = t1.get('candidate_flows', [])

    # Check if the utterance contains description keywords from either candidate
    for fname in candidate_flows:
        flow = flow_catalog.get(fname, {})
        desc = flow.get('description', '')
        # Extract highly specific words from description (>7 chars, not common)
        common_words = {
            'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into',
            'will', 'have', 'been', 'does', 'what', 'when', 'where', 'which',
            'their', 'about', 'using', 'used', 'other', 'more', 'than',
            'based', 'data', 'user', 'post', 'content', 'specific', 'across',
            'should', 'could', 'would', 'without', 'between', 'through',
            'another', 'before', 'after', 'change', 'changes', 'section',
            'sections', 'column', 'columns', 'values', 'existing',
            'different', 'previous', 'current', 'requires', 'looking',
            'something', 'everything', 'anything', 'nothing', 'whether',
            'because', 'already', 'points', 'notes', 'ideas', 'status',
            'format', 'pattern', 'labels', 'dataset', 'metrics', 'rates',
            'structure', 'results', 'process', 'number', 'numbers',
            'bullet', 'heading', 'version', 'writing', 'review',
        }
        desc_words = {
            w.lower() for w in re.findall(r'\b\w+\b', desc)
            if len(w) > 7 and w.lower() not in common_words
        }

        for word in desc_words:
            pattern = rf'\b{re.escape(word)}\b'
            if re.search(pattern, utterance, re.IGNORECASE):
                # Only flag if the word is highly specific to one flow
                other_flow = [f for f in candidate_flows if f != fname]
                if other_flow:
                    other_desc = flow_catalog.get(other_flow[0], {}).get('description', '')
                    if word.lower() not in other_desc.lower():
                        issues.append(
                            f'Turn 1 contains disambiguating keyword "{word}" '
                            f'(specific to {fname}): "{utterance[:60]}"'
                        )

    return issues


def check_multi_request_sanity(convo: dict) -> list[str]:
    """Check 6 (Cat D): Turn 3 should show evidence of multiple requests."""
    issues = []

    if convo.get('category') != 'ambiguous_second':
        return issues

    t3 = convo['turns'][2]
    utterance = t3.get('utterance', '')

    # Look for conjunctions, causal connectors, or multiple sentences.
    # Natural multi-requests often use "so I can", "before", or just
    # two separate sentences — not bare "and" between unrelated intents.
    multi_indicators = [
        r'\band\b',
        r'\bthen\b',
        r'\balso\b',
        r'\bplus\b',
        r'\bwhile you\'re at it\b',
        r'\bafter that\b',
        r'\bon top of that\b',
        r'\bso\b',             # causal: "fix X so I can do Y"
        r'\bbefore\b',         # prerequisite: "clean X before we do Y"
        r'\btoo\b',            # additive: "stack on X too"
        r'\bmeaning\b',        # qualifier: "valid types, meaning the active ones"
        r';',
        r',\s*(and\s+)?(?:also|then)',
        r'[.?!]\s+[A-Z]',     # two sentences (sentence-ending punctuation + capital letter)
    ]

    has_indicator = any(
        re.search(pat, utterance, re.IGNORECASE)
        for pat in multi_indicators
    )

    if not has_indicator:
        issues.append(
            f'Turn 3 lacks multi-request indicators: "{utterance[:80]}"'
        )

    return issues


def check_keyword_match(convo: dict, flow_catalog: dict) -> list[str]:
    """Check 7: Utterances shouldn't be trivially classifiable by regex.

    Only flags distinctive flow names (not common English words).
    """
    issues = []

    trivial_patterns = {}
    for name, flow in flow_catalog.items():
        intent_val = flow['intent'].value if hasattr(flow['intent'], 'value') else str(flow['intent'])
        if intent_val in ('Plan', 'Internal'):
            continue
        if name in COMMON_WORD_FLOWS:
            continue
        trivial_patterns[name] = rf'\b{re.escape(name)}\b'

    for utterance in _user_utterances(convo):
        for fname, pattern in trivial_patterns.items():
            if re.search(pattern, utterance, re.IGNORECASE):
                issues.append(
                    f'Utterance contains trivial keyword "{fname}": '
                    f'"{utterance[:60]}"'
                )

    return issues


# ── Master Validator ──────────────────────────────────────────────────

def validate_conversation(convo: dict, flow_catalog: dict) -> dict:
    """Run all quality checks on a single conversation.

    Returns:
        {convo_id, passed, issues: [str]}
    """
    cid = convo.get('convo_id', '?')
    all_issues = []

    all_issues.extend(check_encoding(convo))
    all_issues.extend(check_format(convo))
    all_issues.extend(check_flow_name_leakage(convo, flow_catalog))
    all_issues.extend(check_length(convo))
    all_issues.extend(check_ambiguity_sanity(convo, flow_catalog))
    all_issues.extend(check_multi_request_sanity(convo))
    all_issues.extend(check_keyword_match(convo, flow_catalog))

    return {
        'convo_id': cid,
        'passed': len(all_issues) == 0,
        'issues': all_issues,
    }


def validate_all(
    convos: list[dict],
    flow_catalog: dict,
) -> dict:
    """Run all quality checks on a list of conversations.

    Returns:
        {
            total, passed, failed,
            pass_rate: float,
            per_convo: [validate_conversation results],
            uniqueness_issues: [str],
            failed_ids: [str],
        }
    """
    per_convo = [validate_conversation(c, flow_catalog) for c in convos]

    # Cross-conversation uniqueness check
    uniqueness_issues = check_uniqueness(convos)

    passed = sum(1 for r in per_convo if r['passed'])
    failed_results = [r for r in per_convo if not r['passed']]
    failed_ids = [r['convo_id'] for r in failed_results]

    return {
        'total': len(convos),
        'passed': passed,
        'failed': len(failed_results),
        'pass_rate': passed / len(convos) if convos else 0.0,
        'per_convo': per_convo,
        'uniqueness_issues': uniqueness_issues,
        'failed_ids': failed_ids,
    }


# ── File-Level Validation ────────────────────────────────────────────

def validate_file(
    file_path: str | Path,
    domain: str,
) -> dict:
    """Validate an eval file (JSON).

    Returns validate_all() result.
    """
    file_path = Path(file_path)

    if domain == 'hugo':
        from datasets.hugo.ontology import FLOW_CATALOG
    elif domain == 'dana':
        from datasets.dana.ontology import FLOW_CATALOG
    else:
        raise ValueError(f'Unknown domain: {domain}')

    with open(file_path) as f:
        data = json.load(f)
        if isinstance(data, list):
            convos = data
        else:
            convos = [data]

    return validate_all(convos, FLOW_CATALOG)
