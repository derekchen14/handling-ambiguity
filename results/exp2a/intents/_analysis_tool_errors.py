"""Comprehensive error analysis for dana_2_001_seed1.jsonl tool-calling results."""

import json
from collections import Counter, defaultdict

# Load data
results_path = "/Users/derekchen/Documents/repos/personal_assistants/experiments/results/exp2a/tools/dana_2_001_seed1.jsonl"
manifest_path = "/Users/derekchen/Documents/repos/personal_assistants/experiments/tools/tool_manifest_dana.json"

with open(manifest_path) as f:
    manifest = json.load(f)

# Build tool->flows map
tool_flow_map = {}
for tool in manifest:
    tool_flow_map[tool['name']] = tool.get('_flows', [])

# Build flow->tools reverse map (which tools can serve a given flow)
flow_tool_map = defaultdict(list)
for tool_name, flows in tool_flow_map.items():
    if flows == ['*']:
        continue  # skip wildcard tools
    for flow in flows:
        flow_tool_map[flow].append(tool_name)

# Parse all turns
all_turns = []
with open(results_path) as f:
    for line in f:
        convo = json.loads(line)
        for turn in convo['turns']:
            turn['convo_id'] = convo['convo_id']
            turn['category'] = convo['category']
            all_turns.append(turn)

print(f"Total turns: {len(all_turns)}")
print(f"Total correct: {sum(1 for t in all_turns if t['correct'])}")
print(f"Total incorrect: {sum(1 for t in all_turns if not t['correct'])}")
print(f"Accuracy: {sum(1 for t in all_turns if t['correct']) / len(all_turns) * 100:.1f}%")
print()

# ══════════════════════════════════════════════════════════════════
# A. NULL TOOL CALLS
# ══════════════════════════════════════════════════════════════════
print("=" * 80)
print("A. NULL TOOL CALLS (model declined to call any tool)")
print("=" * 80)

null_turns = [t for t in all_turns if t['tool_called'] is None]
non_null_turns = [t for t in all_turns if t['tool_called'] is not None]

print(f"\nNull tool calls: {len(null_turns)} / {len(all_turns)} ({len(null_turns)/len(all_turns)*100:.1f}%)")
print(f"Non-null tool calls: {len(non_null_turns)} / {len(all_turns)} ({len(non_null_turns)/len(all_turns)*100:.1f}%)")

# Null by expected flow
null_by_flow = Counter(t['flow'] for t in null_turns)
print(f"\nNull calls by expected flow (top 20):")
for flow, count in null_by_flow.most_common(20):
    total_for_flow = sum(1 for t in all_turns if t['flow'] == flow)
    print(f"  {flow:20s}: {count}/{total_for_flow} null ({count/total_for_flow*100:.0f}%)")

# Null by intent
null_by_intent = Counter(t.get('intent') or '(none)' for t in null_turns)
print(f"\nNull calls by intent:")
for intent, count in null_by_intent.most_common():
    total_for_intent = sum(1 for t in all_turns if (t.get('intent') or '(none)') == intent)
    print(f"  {str(intent):20s}: {count}/{total_for_intent} null ({count/total_for_intent*100:.0f}%)")

# Null by turn number
null_by_turn = Counter(t['turn_num'] for t in null_turns)
print(f"\nNull calls by turn number:")
for turn_num in sorted(null_by_turn.keys()):
    count = null_by_turn[turn_num]
    total_for_turn = sum(1 for t in all_turns if t['turn_num'] == turn_num)
    print(f"  Turn {turn_num}: {count}/{total_for_turn} null ({count/total_for_turn*100:.0f}%)")

# Null by category
null_by_cat = Counter(t['category'] for t in null_turns)
print(f"\nNull calls by category:")
for cat, count in null_by_cat.most_common():
    total_for_cat = sum(1 for t in all_turns if t['category'] == cat)
    print(f"  {cat:20s}: {count}/{total_for_cat} null ({count/total_for_cat*100:.0f}%)")

# Null but expected flow IS mapped
null_mapped = [t for t in null_turns if not t.get('unmapped', False)]
null_unmapped = [t for t in null_turns if t.get('unmapped', False)]
print(f"\nNull + flow is MAPPED (tool exists but model didn't call): {len(null_mapped)}")
print(f"Null + flow is UNMAPPED (no tool for that flow): {len(null_unmapped)}")

# Show 10 examples of null + mapped
print(f"\nExamples of null calls where a tool WAS available:")
for t in null_mapped[:10]:
    available_tools = flow_tool_map.get(t['flow'], [])
    print(f"  [{t['convo_id']} T{t['turn_num']}] flow={t['flow']}, intent={t['intent']}, tools_offered={t['tools_offered']}")
    print(f"    utterance: {t['utterance'][:120]}")
    print(f"    available tools for flow: {available_tools}")
    print()

# ══════════════════════════════════════════════════════════════════
# B. WRONG TOOL CALLS
# ══════════════════════════════════════════════════════════════════
print("=" * 80)
print("B. WRONG TOOL CALLS (tool was called but scored incorrect)")
print("=" * 80)

wrong_tool_turns = [t for t in all_turns if t['tool_called'] is not None and not t['correct']]
correct_tool_turns = [t for t in all_turns if t['tool_called'] is not None and t['correct']]

print(f"\nTurns with tool called: {len(non_null_turns)}")
print(f"  Correct: {len(correct_tool_turns)} ({len(correct_tool_turns)/len(non_null_turns)*100:.1f}%)")
print(f"  Wrong: {len(wrong_tool_turns)} ({len(wrong_tool_turns)/len(non_null_turns)*100:.1f}%)")

# What tools were called when wrong?
wrong_tool_called = Counter(t['tool_called'] for t in wrong_tool_turns)
print(f"\nWrong tool calls by tool called:")
for tool, count in wrong_tool_called.most_common():
    print(f"  {tool:25s}: {count}")

# Confusion pairs: (tool_called, expected_flow)
confusion = Counter((t['tool_called'], t['flow']) for t in wrong_tool_turns)
print(f"\nTop confusion pairs (tool_called -> expected_flow):")
for (tool, flow), count in confusion.most_common(20):
    tool_covers = tool_flow_map.get(tool, [])
    print(f"  {tool:25s} -> expected '{flow}' (tool covers: {tool_covers[:5]}{'...' if len(tool_covers)>5 else ''}) x{count}")

# Show specific examples
print(f"\nDetailed wrong-tool examples:")
for t in wrong_tool_turns[:15]:
    tool_covers = tool_flow_map.get(t['tool_called'], [])
    correct_tools = flow_tool_map.get(t['flow'], [])
    print(f"  [{t['convo_id']} T{t['turn_num']}] called={t['tool_called']}, expected_flow={t['flow']}, intent={t['intent']}")
    print(f"    utterance: {t['utterance'][:120]}")
    print(f"    tool's flows: {tool_covers[:8]}{'...' if len(tool_covers)>8 else ''}")
    print(f"    correct tools for '{t['flow']}': {correct_tools}")
    if t.get('tool_args'):
        print(f"    args: {json.dumps(t['tool_args'])[:150]}")
    print()


# ══════════════════════════════════════════════════════════════════
# C. TOOL COVERAGE
# ══════════════════════════════════════════════════════════════════
print("=" * 80)
print("C. TOOL COVERAGE (by expected flow)")
print("=" * 80)

# For each expected flow, how many turns, how many correct, tools_offered distribution
flow_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'null': 0, 'tools_offered': [], 'tools_called': Counter()})
for t in all_turns:
    f = t['flow']
    flow_stats[f]['total'] += 1
    flow_stats[f]['correct'] += 1 if t['correct'] else 0
    flow_stats[f]['null'] += 1 if t['tool_called'] is None else 0
    flow_stats[f]['tools_offered'].append(t['tools_offered'])
    if t['tool_called']:
        flow_stats[f]['tools_called'][t['tool_called']] += 1

print(f"\n{'Flow':<20s} {'Total':>5s} {'Correct':>7s} {'Acc%':>5s} {'Null':>5s} {'AvgTools':>8s} {'TopToolCalled':<30s}")
print("-" * 90)
for flow in sorted(flow_stats.keys()):
    s = flow_stats[flow]
    acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
    avg_tools = sum(s['tools_offered']) / len(s['tools_offered'])
    top_tool = s['tools_called'].most_common(1)[0] if s['tools_called'] else ('(none)', 0)
    mapped = 'Y' if flow in flow_tool_map else 'N'
    print(f"  {flow:<18s} {s['total']:>5d} {s['correct']:>7d} {acc:>5.1f} {s['null']:>5d} {avg_tools:>8.1f} {top_tool[0]+'('+str(top_tool[1])+')' if top_tool[0]!='(none)' else '(none)':30s}  mapped={mapped}")

# ══════════════════════════════════════════════════════════════════
# D. UNMAPPED FLOWS
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("D. UNMAPPED FLOWS")
print("=" * 80)

unmapped_turns = [t for t in all_turns if t.get('unmapped', False)]
print(f"\nTurns with unmapped=true: {len(unmapped_turns)} / {len(all_turns)} ({len(unmapped_turns)/len(all_turns)*100:.1f}%)")

unmapped_flows = Counter(t['flow'] for t in unmapped_turns)
print(f"\nUnmapped flow names:")
for flow, count in unmapped_flows.most_common():
    print(f"  {flow:20s}: {count} turns")

# How did the model handle unmapped turns?
unmapped_tool_calls = Counter(t['tool_called'] if t['tool_called'] else '(null)' for t in unmapped_turns)
print(f"\nHow model responded to unmapped flows:")
for tool, count in unmapped_tool_calls.most_common():
    print(f"  {tool:25s}: {count}")

unmapped_correct = sum(1 for t in unmapped_turns if t['correct'])
print(f"\nUnmapped turns correct: {unmapped_correct}/{len(unmapped_turns)}")

# ══════════════════════════════════════════════════════════════════
# E. PARAMETER QUALITY
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("E. PARAMETER QUALITY (correct tool calls — examining tool_args)")
print("=" * 80)

correct_with_args = [t for t in all_turns if t['correct'] and t['tool_called'] and t.get('tool_args')]
print(f"\nCorrect turns with tool_args: {len(correct_with_args)}")

print(f"\n--- GOOD parameter fills ---")
good_examples = []
bad_examples = []

for t in correct_with_args:
    args = t.get('tool_args', {})
    utterance = t['utterance']

    # Check for specificity: does args reference things from the utterance?
    args_str = json.dumps(args).lower()
    utt_lower = utterance.lower()

    # Quick heuristic: count how many arg values appear in the utterance
    value_matches = 0
    for v in args.values():
        if isinstance(v, str) and len(v) > 2 and v.lower() in utt_lower:
            value_matches += 1

    if value_matches >= 2:
        good_examples.append(t)
    elif value_matches == 0 and len(args) >= 2:
        bad_examples.append(t)

for t in good_examples[:8]:
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] tool={t['tool_called']}, flow={t['flow']}")
    print(f"    utterance: {t['utterance'][:120]}")
    print(f"    args: {json.dumps(t['tool_args'])[:200]}")

print(f"\n--- POOR/GENERIC parameter fills ---")
for t in bad_examples[:8]:
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] tool={t['tool_called']}, flow={t['flow']}")
    print(f"    utterance: {t['utterance'][:120]}")
    print(f"    args: {json.dumps(t['tool_args'])[:200]}")

# Also look at incorrect turns with args to see bad parameterization
print(f"\n--- Incorrect turns with tool_args (wrong tool or wrong mapping) ---")
wrong_with_args = [t for t in all_turns if not t['correct'] and t['tool_called'] and t.get('tool_args')]
for t in wrong_with_args[:8]:
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] tool={t['tool_called']}, expected_flow={t['flow']}")
    print(f"    utterance: {t['utterance'][:120]}")
    print(f"    args: {json.dumps(t['tool_args'])[:200]}")

# ══════════════════════════════════════════════════════════════════
# F. MULTI-TOOL POTENTIAL (arguably correct calls)
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("F. MULTI-TOOL POTENTIAL (wrong by scorer but arguably reasonable)")
print("=" * 80)

# Find wrong-tool turns where the called tool's flows overlap with related flows
arguably_ok = []
for t in wrong_tool_turns:
    if not t['tool_called'] or t['tool_called'] in ('no_tool_needed', 'handle_ambiguity'):
        continue
    tool_covers = set(tool_flow_map.get(t['tool_called'], []))
    expected = t['flow']

    # Check if the tool covers a semantically related flow
    # e.g., describe_stats covers many flows, execute_python is very general
    # Also check if tool was a reasonable alternative
    if t['tool_called'] in ('describe_stats', 'semantic_layer', 'execute_python', 'execute_sql'):
        arguably_ok.append(t)
    elif tool_covers and expected not in tool_covers:
        # Does the tool cover a flow from the same intent?
        arguably_ok.append(t)

print(f"\nArguably reasonable wrong calls: {len(arguably_ok)}")
for t in arguably_ok[:15]:
    tool_covers = tool_flow_map.get(t['tool_called'], [])
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] called={t['tool_called']}, expected={t['flow']}, intent={t['intent']}")
    print(f"    utterance: {t['utterance'][:120]}")
    print(f"    tool covers flows: {tool_covers[:8]}{'...' if len(tool_covers)>8 else ''}")
    if t.get('tool_args'):
        print(f"    args: {json.dumps(t['tool_args'])[:150]}")

# ══════════════════════════════════════════════════════════════════
# G. CATEGORY BREAKDOWN
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("G. CATEGORY & TURN NUMBER BREAKDOWN")
print("=" * 80)

# By category
cat_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
for t in all_turns:
    cat_stats[t['category']]['total'] += 1
    cat_stats[t['category']]['correct'] += 1 if t['correct'] else 0

print(f"\nAccuracy by category:")
for cat in sorted(cat_stats.keys()):
    s = cat_stats[cat]
    acc = s['correct'] / s['total'] * 100
    print(f"  {cat:25s}: {s['correct']}/{s['total']} = {acc:.1f}%")

# By turn number
turn_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
for t in all_turns:
    turn_stats[t['turn_num']]['total'] += 1
    turn_stats[t['turn_num']]['correct'] += 1 if t['correct'] else 0

print(f"\nAccuracy by turn number:")
for tn in sorted(turn_stats.keys()):
    s = turn_stats[tn]
    acc = s['correct'] / s['total'] * 100
    print(f"  Turn {tn}: {s['correct']}/{s['total']} = {acc:.1f}%")

# Category x Turn
print(f"\nAccuracy by category x turn:")
cat_turn = defaultdict(lambda: {'total': 0, 'correct': 0})
for t in all_turns:
    key = (t['category'], t['turn_num'])
    cat_turn[key]['total'] += 1
    cat_turn[key]['correct'] += 1 if t['correct'] else 0

for key in sorted(cat_turn.keys()):
    s = cat_turn[key]
    acc = s['correct'] / s['total'] * 100
    print(f"  {key[0]:25s} Turn {key[1]}: {s['correct']}/{s['total']} = {acc:.1f}%")

# By intent
intent_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
for t in all_turns:
    ikey = t.get('intent') or '(none)'
    intent_stats[ikey]['total'] += 1
    intent_stats[ikey]['correct'] += 1 if t['correct'] else 0

print(f"\nAccuracy by intent:")
for intent in sorted(intent_stats.keys(), key=str):
    s = intent_stats[intent]
    acc = s['correct'] / s['total'] * 100
    print(f"  {str(intent):20s}: {s['correct']}/{s['total']} = {acc:.1f}%")


# ══════════════════════════════════════════════════════════════════
# H. NO_TOOL_NEEDED CALLS
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("H. NO_TOOL_NEEDED CALLS")
print("=" * 80)

no_tool = [t for t in all_turns if t['tool_called'] == 'no_tool_needed']
print(f"\nTurns calling no_tool_needed: {len(no_tool)}")
for t in no_tool:
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] expected_flow={t['flow']}, intent={t['intent']}, correct={t['correct']}, unmapped={t.get('unmapped')}")
    print(f"    utterance: {t['utterance'][:120]}")
    if t.get('tool_args'):
        print(f"    args: {json.dumps(t['tool_args'])[:200]}")

# ══════════════════════════════════════════════════════════════════
# HANDLE_AMBIGUITY CALLS
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("HANDLE_AMBIGUITY CALLS")
print("=" * 80)

ambig_tool = [t for t in all_turns if t['tool_called'] == 'handle_ambiguity']
print(f"\nTurns calling handle_ambiguity: {len(ambig_tool)}")
for t in ambig_tool:
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] expected_flow={t['flow']}, intent={t['intent']}, correct={t['correct']}, category={t['category']}")
    print(f"    candidate_flows={t.get('candidate_flows')}")
    print(f"    utterance: {t['utterance'][:120]}")
    if t.get('tool_args'):
        print(f"    args: {json.dumps(t['tool_args'])[:200]}")

# ══════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nTotal turns: {len(all_turns)}")
print(f"Accuracy: {sum(1 for t in all_turns if t['correct'])/len(all_turns)*100:.1f}%")
print(f"Null tool calls: {len(null_turns)} ({len(null_turns)/len(all_turns)*100:.1f}%)")
print(f"  - of which unmapped: {len(null_unmapped)}")
print(f"  - of which mapped (model should have called): {len(null_mapped)}")
print(f"Non-null tool calls: {len(non_null_turns)} ({len(non_null_turns)/len(all_turns)*100:.1f}%)")
print(f"  - correct: {len(correct_tool_turns)} ({len(correct_tool_turns)/len(non_null_turns)*100:.1f}% of non-null)")
print(f"  - wrong: {len(wrong_tool_turns)} ({len(wrong_tool_turns)/len(non_null_turns)*100:.1f}% of non-null)")
print(f"no_tool_needed calls: {len(no_tool)}")
print(f"handle_ambiguity calls: {len(ambig_tool)}")
print(f"Unmapped turns total: {len(unmapped_turns)}")
