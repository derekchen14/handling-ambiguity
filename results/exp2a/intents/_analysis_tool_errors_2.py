"""Supplemental analysis: hallucinated tools, ambiguous categories, outline failures."""

import json
from collections import Counter, defaultdict

results_path = "/Users/derekchen/Documents/repos/personal_assistants/experiments/results/exp2a/tools/dana_2_001_seed1.jsonl"
manifest_path = "/Users/derekchen/Documents/repos/personal_assistants/experiments/tools/tool_manifest_dana.json"

with open(manifest_path) as f:
    manifest = json.load(f)

tool_names = {t['name'] for t in manifest}
tool_flow_map = {}
for tool in manifest:
    tool_flow_map[tool['name']] = tool.get('_flows', [])

all_turns = []
with open(results_path) as f:
    for line in f:
        convo = json.loads(line)
        for turn in convo['turns']:
            turn['convo_id'] = convo['convo_id']
            turn['category'] = convo['category']
            all_turns.append(turn)

# ══════════════════════════════════════════════════════════════════
# HALLUCINATED TOOL NAMES
# ══════════════════════════════════════════════════════════════════
print("=" * 80)
print("HALLUCINATED TOOL NAMES (called a tool not in the manifest)")
print("=" * 80)

hallucinated = [t for t in all_turns if t['tool_called'] and t['tool_called'] not in tool_names]
print(f"\nHallucinated tool calls: {len(hallucinated)}")
hall_names = Counter(t['tool_called'] for t in hallucinated)
for name, count in hall_names.most_common():
    print(f"  {name:30s}: {count}")

print(f"\nDetailed hallucinated tool calls:")
for t in hallucinated:
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] called='{t['tool_called']}', expected_flow={t['flow']}, intent={t.get('intent')}")
    print(f"    utterance: {t['utterance'][:150]}")
    if t.get('tool_args'):
        print(f"    args: {json.dumps(t['tool_args'])[:200]}")

# ══════════════════════════════════════════════════════════════════
# OUTLINE FLOW DEEP DIVE (32 turns, only 15.6% accuracy)
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("OUTLINE FLOW DEEP DIVE (expected_flow='outline')")
print("=" * 80)

outline_turns = [t for t in all_turns if t['flow'] == 'outline']
print(f"\nTotal outline turns: {len(outline_turns)}")
print(f"Correct: {sum(1 for t in outline_turns if t['correct'])}")
print(f"Null: {sum(1 for t in outline_turns if t['tool_called'] is None)}")

# What tools does outline map to?
outline_tools = []
for tool_name, flows in tool_flow_map.items():
    if 'outline' in flows:
        outline_tools.append(tool_name)
print(f"Tools mapped to 'outline': {outline_tools}")

# What did the model call?
outline_calls = Counter(t['tool_called'] if t['tool_called'] else '(null)' for t in outline_turns)
print(f"\nModel's tool choices for outline turns:")
for tool, count in outline_calls.most_common():
    print(f"  {tool:30s}: {count}")

# Show all outline utterances
print(f"\nAll outline utterances:")
for t in outline_turns:
    cat_info = f"cat={t['category']}"
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] correct={t['correct']}, called={t['tool_called']}, {cat_info}, tools_offered={t['tools_offered']}")
    print(f"    utterance: {t['utterance'][:180]}")
    if t.get('candidate_flows'):
        print(f"    candidate_flows: {t['candidate_flows']}")

# ══════════════════════════════════════════════════════════════════
# TREND FLOW DEEP DIVE (7 turns, 0% accuracy)
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("TREND FLOW DEEP DIVE (expected_flow='trend', 0% accuracy)")
print("=" * 80)

trend_turns = [t for t in all_turns if t['flow'] == 'trend']
trend_tools = []
for tool_name, flows in tool_flow_map.items():
    if flows != ['*'] and 'trend' in flows:
        trend_tools.append(tool_name)
print(f"Tools mapped to 'trend': {trend_tools}")

for t in trend_turns:
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] correct={t['correct']}, called={t['tool_called']}, cat={t['category']}")
    print(f"    utterance: {t['utterance'][:180]}")
    print(f"    tools_offered={t['tools_offered']}")

# ══════════════════════════════════════════════════════════════════
# DASHBOARD FLOW DEEP DIVE (5 turns, 0% accuracy)
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("DASHBOARD FLOW DEEP DIVE (expected_flow='dashboard', 0% accuracy)")
print("=" * 80)

dash_turns = [t for t in all_turns if t['flow'] == 'dashboard']
dash_tools = []
for tool_name, flows in tool_flow_map.items():
    if flows != ['*'] and 'dashboard' in flows:
        dash_tools.append(tool_name)
print(f"Tools mapped to 'dashboard': {dash_tools}")

for t in dash_turns:
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] correct={t['correct']}, called={t['tool_called']}, cat={t['category']}")
    print(f"    utterance: {t['utterance'][:180]}")
    print(f"    tools_offered={t['tools_offered']}")

# ══════════════════════════════════════════════════════════════════
# SUMMARIZE FLOW DEEP DIVE (6 turns, 0% accuracy)
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("SUMMARIZE FLOW DEEP DIVE (expected_flow='summarize', 0% accuracy)")
print("=" * 80)

summ_turns = [t for t in all_turns if t['flow'] == 'summarize']
summ_tools = []
for tool_name, flows in tool_flow_map.items():
    if flows != ['*'] and 'summarize' in flows:
        summ_tools.append(tool_name)
print(f"Tools mapped to 'summarize': {summ_tools}")

for t in summ_turns:
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] correct={t['correct']}, called={t['tool_called']}, cat={t['category']}")
    print(f"    utterance: {t['utterance'][:180]}")
    print(f"    tools_offered={t['tools_offered']}")

# ══════════════════════════════════════════════════════════════════
# COMPARE FLOW DEEP DIVE (3 turns, 0% accuracy)
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("COMPARE FLOW DEEP DIVE (expected_flow='compare', 0% accuracy)"  )
print("=" * 80)

comp_turns = [t for t in all_turns if t['flow'] == 'compare']
comp_tools = []
for tool_name, flows in tool_flow_map.items():
    if flows != ['*'] and 'compare' in flows:
        comp_tools.append(tool_name)
print(f"Tools mapped to 'compare': {comp_tools}")

for t in comp_turns:
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] correct={t['correct']}, called={t['tool_called']}, cat={t['category']}")
    print(f"    utterance: {t['utterance'][:180]}")
    print(f"    tools_offered={t['tools_offered']}")

# ══════════════════════════════════════════════════════════════════
# AMBIGUOUS CATEGORY ANALYSIS
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("AMBIGUOUS CATEGORY ANALYSIS (ambiguous_first + ambiguous_second)")
print("=" * 80)

ambig_turns = [t for t in all_turns if t['category'] in ('ambiguous_first', 'ambiguous_second')]
print(f"\nTotal ambiguous turns: {len(ambig_turns)}")
print(f"  correct: {sum(1 for t in ambig_turns if t['correct'])}")

# How are ambiguous turns handled?
# The flow field in ambiguous turns
ambig_flows = Counter(t['flow'] for t in ambig_turns)
print(f"\nExpected flows in ambiguous turns:")
for flow, count in ambig_flows.most_common():
    print(f"  {flow:20s}: {count}")

# Ambiguous turns with candidate_flows
ambig_with_candidates = [t for t in ambig_turns if t.get('candidate_flows')]
ambig_no_candidates = [t for t in ambig_turns if not t.get('candidate_flows')]
print(f"\nAmbiguous turns WITH candidate_flows: {len(ambig_with_candidates)}")
print(f"Ambiguous turns WITHOUT candidate_flows: {len(ambig_no_candidates)}")

# Show a few ambiguous_first turn 1 examples (should be 100% correct)
print(f"\nAmbiguous_first Turn 1 examples (all correct=True because flow='ambiguous' and unmapped=True):")
af_t1 = [t for t in all_turns if t['category'] == 'ambiguous_first' and t['turn_num'] == 1]
for t in af_t1[:5]:
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] flow={t['flow']}, correct={t['correct']}, unmapped={t.get('unmapped')}, called={t['tool_called']}")
    print(f"    utterance: {t['utterance'][:150]}")
    print(f"    candidate_flows: {t.get('candidate_flows')}")

# Show ambiguous_second Turn 3 examples (low accuracy)
print(f"\nAmbiguous_second Turn 3 examples (15.6% accuracy):")
as_t3 = [t for t in all_turns if t['category'] == 'ambiguous_second' and t['turn_num'] == 3]
for t in as_t3[:10]:
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] flow={t['flow']}, correct={t['correct']}, called={t['tool_called']}")
    print(f"    utterance: {t['utterance'][:150]}")
    print(f"    candidate_flows: {t.get('candidate_flows')}")
    print(f"    tools_offered: {t['tools_offered']}")

# ══════════════════════════════════════════════════════════════════
# TOOLS_OFFERED = 0 analysis (where no tools were offered)
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("TURNS WITH ZERO TOOLS OFFERED")
print("=" * 80)

zero_tools = [t for t in all_turns if t['tools_offered'] == 0]
print(f"\nTurns with tools_offered=0: {len(zero_tools)}")
print(f"Flows: {Counter(t['flow'] for t in zero_tools).most_common()}")
print(f"All correct: {all(t['correct'] for t in zero_tools)}")

# ══════════════════════════════════════════════════════════════════
# ACCURACY EXCLUDING UNMAPPED & AMBIGUOUS
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("ACCURACY ON MAPPED, CLEAR TURNS ONLY")
print("=" * 80)

mapped_clear = [t for t in all_turns if not t.get('unmapped', False) and not t.get('candidate_flows')]
print(f"\nMapped clear turns: {len(mapped_clear)}")
print(f"Correct: {sum(1 for t in mapped_clear if t['correct'])}")
print(f"Accuracy: {sum(1 for t in mapped_clear if t['correct'])/len(mapped_clear)*100:.1f}%")

# Of these, null vs non-null
mc_null = [t for t in mapped_clear if t['tool_called'] is None]
mc_nonnull = [t for t in mapped_clear if t['tool_called'] is not None]
print(f"  Null calls: {len(mc_null)} ({len(mc_null)/len(mapped_clear)*100:.1f}%)")
print(f"  Non-null calls: {len(mc_nonnull)} ({len(mc_nonnull)/len(mapped_clear)*100:.1f}%)")
print(f"  Non-null accuracy: {sum(1 for t in mc_nonnull if t['correct'])/len(mc_nonnull)*100:.1f}%")

# ══════════════════════════════════════════════════════════════════
# TOOLS_OFFERED distribution
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("TOOLS OFFERED DISTRIBUTION")
print("=" * 80)
tools_offered_dist = Counter(t['tools_offered'] for t in all_turns)
for n in sorted(tools_offered_dist.keys()):
    ct = tools_offered_dist[n]
    correct_at_n = sum(1 for t in all_turns if t['tools_offered'] == n and t['correct'])
    print(f"  tools_offered={n}: {ct} turns, {correct_at_n} correct ({correct_at_n/ct*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════
# NULL CALL TURNS — TURN 3 CONTEXT DEPENDENCE
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("NULL CALL TURN 3 EXAMPLES (context-dependent utterances the model missed)")
print("=" * 80)

null_t3_mapped = [t for t in all_turns if t['tool_called'] is None and t['turn_num'] == 3 and not t.get('unmapped', False)]
print(f"\nNull Turn 3 mapped turns: {len(null_t3_mapped)}")
for t in null_t3_mapped[:15]:
    print(f"\n  [{t['convo_id']} T{t['turn_num']}] flow={t['flow']}, intent={t.get('intent')}, cat={t['category']}")
    print(f"    utterance: {t['utterance'][:180]}")
    print(f"    tools_offered: {t['tools_offered']}")
