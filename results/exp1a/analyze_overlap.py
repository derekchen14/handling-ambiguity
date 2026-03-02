"""
Compare accuracy across Gemini 2.5 Pro, 3.0 Pro, and 3.1 Pro
on the overlap set of conversations where ALL three models have predictions.
"""
import json
from collections import defaultdict

def load_jsonl(path):
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["convo_id"]] = rec
    return records

# Load Hugo seed1 for all three models
base = "/Users/derekchen/Documents/repos/personal_assistants/experiments/results/exp1a"
g25 = load_jsonl(f"{base}/hugo_1a_025_seed1.jsonl")
g30 = load_jsonl(f"{base}/hugo_1a_006_seed1.jsonl")
g31 = load_jsonl(f"{base}/hugo_1a_007_seed1.jsonl")

print(f"=== DATASET SIZES ===")
print(f"Gemini 2.5 Pro (hugo seed1): {len(g25)} conversations")
print(f"Gemini 3.0 Pro (hugo seed1): {len(g30)} conversations")
print(f"Gemini 3.1 Pro (hugo seed1): {len(g31)} conversations")

# Find overlap
overlap_ids = set(g25.keys()) & set(g30.keys()) & set(g31.keys())
print(f"\nOverlap (all 3 models): {len(overlap_ids)} conversations")
print(f"  IDs in 2.5 but not in overlap: {len(set(g25.keys()) - overlap_ids)}")
print(f"  IDs in 3.0 but not in overlap: {len(set(g30.keys()) - overlap_ids)}")
print(f"  IDs in 3.1 but not in overlap: {len(set(g31.keys()) - overlap_ids)}")

# Helper: compute accuracy stats from a set of records filtered to given ids
def compute_stats(records, ids):
    total_turns = 0
    correct_turns = 0
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    by_turn = defaultdict(lambda: {"correct": 0, "total": 0})

    for cid in sorted(ids):
        rec = records[cid]
        cat = rec["category"]
        for turn in rec["turns"]:
            total_turns += 1
            is_correct = turn["correct"]
            if is_correct:
                correct_turns += 1

            by_category[cat]["total"] += 1
            if is_correct:
                by_category[cat]["correct"] += 1

            tn = f"turn_{turn['turn_num']}"
            by_turn[tn]["total"] += 1
            if is_correct:
                by_turn[tn]["correct"] += 1

    return {
        "total_turns": total_turns,
        "correct_turns": correct_turns,
        "accuracy": correct_turns / total_turns if total_turns > 0 else 0,
        "by_category": dict(by_category),
        "by_turn": dict(by_turn),
    }

# Compute stats for each model on the overlap set
models = {
    "Gemini 2.5 Pro": g25,
    "Gemini 3.0 Pro": g30,
    "Gemini 3.1 Pro": g31,
}

print(f"\n{'='*70}")
print(f"=== ACCURACY ON OVERLAP SET ({len(overlap_ids)} conversations) ===")
print(f"{'='*70}")

stats = {}
for name, records in models.items():
    s = compute_stats(records, overlap_ids)
    stats[name] = s
    print(f"\n--- {name} ---")
    print(f"  Overall: {s['correct_turns']}/{s['total_turns']} = {s['accuracy']:.1%}")

    print(f"  By category:")
    for cat in ["same_flow", "switch_flow", "ambiguous_first", "ambiguous_second"]:
        if cat in s["by_category"]:
            c = s["by_category"][cat]
            acc = c["correct"] / c["total"] if c["total"] > 0 else 0
            print(f"    {cat:20s}: {c['correct']:3d}/{c['total']:3d} = {acc:.1%}")

    print(f"  By turn:")
    for tn in sorted(s["by_turn"].keys()):
        t = s["by_turn"][tn]
        acc = t["correct"] / t["total"] if t["total"] > 0 else 0
        print(f"    {tn:10s}: {t['correct']:3d}/{t['total']:3d} = {acc:.1%}")

# Head-to-head comparisons
print(f"\n{'='*70}")
print(f"=== HEAD-TO-HEAD ON OVERLAP SET ===")
print(f"{'='*70}")

def head_to_head(records_a, records_b, name_a, name_b, ids):
    both_correct = 0
    a_only = 0
    b_only = 0
    both_wrong = 0

    for cid in sorted(ids):
        rec_a = records_a[cid]
        rec_b = records_b[cid]
        for i, (ta, tb) in enumerate(zip(rec_a["turns"], rec_b["turns"])):
            ca = ta["correct"]
            cb = tb["correct"]
            if ca and cb:
                both_correct += 1
            elif ca and not cb:
                a_only += 1
            elif not ca and cb:
                b_only += 1
            else:
                both_wrong += 1

    total = both_correct + a_only + b_only + both_wrong
    print(f"\n{name_a} vs {name_b} ({total} turns):")
    print(f"  Both correct:           {both_correct:3d} ({both_correct/total:.1%})")
    print(f"  {name_a:15s} only:  {a_only:3d} ({a_only/total:.1%})")
    print(f"  {name_b:15s} only:  {b_only:3d} ({b_only/total:.1%})")
    print(f"  Both wrong:             {both_wrong:3d} ({both_wrong/total:.1%})")

    # McNemar-style: who wins the disagreements?
    disagree = a_only + b_only
    if disagree > 0:
        print(f"  Disagreement wins: {name_a} {a_only}/{disagree} ({a_only/disagree:.1%}) vs {name_b} {b_only}/{disagree} ({b_only/disagree:.1%})")

head_to_head(g25, g30, "Gemini 2.5 Pro", "Gemini 3.0 Pro", overlap_ids)
head_to_head(g25, g31, "Gemini 2.5 Pro", "Gemini 3.1 Pro", overlap_ids)
head_to_head(g30, g31, "Gemini 3.0 Pro", "Gemini 3.1 Pro", overlap_ids)

# Selection bias check: 2.5 Pro full vs overlap
print(f"\n{'='*70}")
print(f"=== SELECTION BIAS CHECK: Gemini 2.5 Pro ===")
print(f"{'='*70}")

full_stats = compute_stats(g25, set(g25.keys()))
overlap_stats = stats["Gemini 2.5 Pro"]
non_overlap_ids = set(g25.keys()) - overlap_ids
non_overlap_stats = compute_stats(g25, non_overlap_ids)

print(f"\nFull dataset ({len(g25)} convos):     {full_stats['accuracy']:.1%}")
print(f"Overlap only ({len(overlap_ids)} convos):    {overlap_stats['accuracy']:.1%}")
print(f"Non-overlap  ({len(non_overlap_ids)} convos):     {non_overlap_stats['accuracy']:.1%}")

print(f"\nFull dataset by category:")
for cat in ["same_flow", "switch_flow", "ambiguous_first", "ambiguous_second"]:
    full_c = full_stats["by_category"].get(cat, {"correct": 0, "total": 0})
    over_c = overlap_stats["by_category"].get(cat, {"correct": 0, "total": 0})
    full_acc = full_c["correct"] / full_c["total"] if full_c["total"] > 0 else 0
    over_acc = over_c["correct"] / over_c["total"] if over_c["total"] > 0 else 0
    print(f"  {cat:20s}: full={full_acc:.1%} ({full_c['correct']}/{full_c['total']})  overlap={over_acc:.1%} ({over_c['correct']}/{over_c['total']})")

# Detailed disagreement analysis: what convos/turns differ?
print(f"\n{'='*70}")
print(f"=== DETAILED DISAGREEMENTS (3.1 Pro correct, 2.5 Pro wrong) ===")
print(f"{'='*70}")

for cid in sorted(overlap_ids):
    rec25 = g25[cid]
    rec31 = g31[cid]
    for i, (t25, t31) in enumerate(zip(rec25["turns"], rec31["turns"])):
        if not t25["correct"] and t31["correct"]:
            print(f"\n  {cid} turn {t25['turn_num']} ({rec25['category']}):")
            print(f"    Ground truth: {t25['flow']}")
            print(f"    2.5 Pro detected: {t25['detected_flows']}")
            print(f"    3.1 Pro detected: {t31['detected_flows']}")

print(f"\n{'='*70}")
print(f"=== DETAILED DISAGREEMENTS (2.5 Pro correct, 3.1 Pro wrong) ===")
print(f"{'='*70}")

for cid in sorted(overlap_ids):
    rec25 = g25[cid]
    rec31 = g31[cid]
    for i, (t25, t31) in enumerate(zip(rec25["turns"], rec31["turns"])):
        if t25["correct"] and not t31["correct"]:
            print(f"\n  {cid} turn {t25['turn_num']} ({rec25['category']}):")
            print(f"    Ground truth: {t25['flow']}")
            print(f"    2.5 Pro detected: {t25['detected_flows']}")
            print(f"    3.1 Pro detected: {t31['detected_flows']}")

# Latency comparison
print(f"\n{'='*70}")
print(f"=== AVERAGE LATENCY ON OVERLAP SET ===")
print(f"{'='*70}")

for name, records in models.items():
    latencies = []
    for cid in overlap_ids:
        for turn in records[cid]["turns"]:
            if "latency_ms" in turn:
                latencies.append(turn["latency_ms"])
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"  {name:20s}: {avg:.0f}ms avg ({min(latencies)}ms - {max(latencies)}ms)")
