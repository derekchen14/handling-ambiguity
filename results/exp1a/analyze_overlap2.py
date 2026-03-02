"""
Supplementary analysis: category distribution in overlap vs full,
and where 3.1 Pro fixes errors that both 2.5 and 3.0 get wrong.
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

base = "/Users/derekchen/Documents/repos/personal_assistants/experiments/results/exp1a"
g25 = load_jsonl(f"{base}/hugo_1a_025_seed1.jsonl")
g30 = load_jsonl(f"{base}/hugo_1a_006_seed1.jsonl")
g31 = load_jsonl(f"{base}/hugo_1a_007_seed1.jsonl")

overlap_ids = set(g25.keys()) & set(g30.keys()) & set(g31.keys())

# Category distribution
print("=== CATEGORY DISTRIBUTION ===")
print(f"\n{'Category':25s} {'Overlap':>10s} {'Full 2.5':>10s} {'Non-overlap':>12s}")
cats_overlap = defaultdict(int)
cats_full = defaultdict(int)
cats_non = defaultdict(int)

for cid in overlap_ids:
    cats_overlap[g25[cid]["category"]] += 1
for cid in g25:
    cats_full[g25[cid]["category"]] += 1
for cid in set(g25.keys()) - overlap_ids:
    cats_non[g25[cid]["category"]] += 1

for cat in ["same_flow", "switch_flow", "ambiguous_first", "ambiguous_second"]:
    print(f"  {cat:25s} {cats_overlap[cat]:10d} {cats_full[cat]:10d} {cats_non[cat]:12d}")

# Where 3.1 beats BOTH 2.5 and 3.0
print(f"\n=== TURNS WHERE 3.1 Pro IS CORRECT BUT BOTH 2.5 AND 3.0 ARE WRONG ===")
count = 0
for cid in sorted(overlap_ids):
    for i in range(len(g25[cid]["turns"])):
        t25 = g25[cid]["turns"][i]
        t30 = g30[cid]["turns"][i]
        t31 = g31[cid]["turns"][i]
        if t31["correct"] and not t25["correct"] and not t30["correct"]:
            count += 1
            print(f"\n  {cid} turn {t25['turn_num']} ({g25[cid]['category']}):")
            print(f"    Utterance: {t25['utterance']}")
            print(f"    Ground truth: {t25['flow']}")
            print(f"    2.5 detected: {t25['detected_flows']}")
            print(f"    3.0 detected: {t30['detected_flows']}")
            print(f"    3.1 detected: {t31['detected_flows']}")
print(f"\nTotal: {count} turns")

# Where ALL three models fail
print(f"\n=== TURNS WHERE ALL THREE MODELS ARE WRONG ===")
all_wrong = []
for cid in sorted(overlap_ids):
    for i in range(len(g25[cid]["turns"])):
        t25 = g25[cid]["turns"][i]
        t30 = g30[cid]["turns"][i]
        t31 = g31[cid]["turns"][i]
        if not t25["correct"] and not t30["correct"] and not t31["correct"]:
            all_wrong.append((cid, t25["turn_num"], g25[cid]["category"], t25["flow"], t25["detected_flows"], t30["detected_flows"], t31["detected_flows"]))

print(f"Total: {len(all_wrong)} turns")
for cid, tn, cat, flow, d25, d30, d31 in all_wrong:
    agree = "AGREE" if d25 == d30 == d31 else "DIFFER"
    print(f"  {cid} t{tn} ({cat:20s}) truth={flow:12s} | 2.5={d25} 3.0={d30} 3.1={d31} [{agree}]")

# Accuracy on ambiguous_first turn 1 (which is always "ambiguous" ground truth)
# These are scored as incorrect by design when the model picks one candidate
print(f"\n=== NOTE ON AMBIGUOUS_FIRST TURN 1 ===")
for name, records in [("2.5 Pro", g25), ("3.0 Pro", g30), ("3.1 Pro", g31)]:
    correct_t1 = 0
    total_t1 = 0
    correct_t3 = 0
    total_t3 = 0
    for cid in overlap_ids:
        rec = records[cid]
        if rec["category"] == "ambiguous_first":
            for t in rec["turns"]:
                if t["turn_num"] == 1:
                    total_t1 += 1
                    if t["correct"]: correct_t1 += 1
                elif t["turn_num"] == 3:
                    total_t3 += 1
                    if t["correct"]: correct_t3 += 1
    print(f"  {name}: ambig_first turn 1 = {correct_t1}/{total_t1}, turn 3 = {correct_t3}/{total_t3}")

# Same check for ambiguous_second
print(f"\n=== NOTE ON AMBIGUOUS_SECOND TURNS ===")
for name, records in [("2.5 Pro", g25), ("3.0 Pro", g30), ("3.1 Pro", g31)]:
    correct_t1 = 0
    total_t1 = 0
    correct_t3 = 0
    total_t3 = 0
    for cid in overlap_ids:
        rec = records[cid]
        if rec["category"] == "ambiguous_second":
            for t in rec["turns"]:
                if t["turn_num"] == 1:
                    total_t1 += 1
                    if t["correct"]: correct_t1 += 1
                elif t["turn_num"] == 3:
                    total_t3 += 1
                    if t["correct"]: correct_t3 += 1
    print(f"  {name}: ambig_second turn 1 = {correct_t1}/{total_t1}, turn 3 = {correct_t3}/{total_t3}")

# Summary table
print(f"\n{'='*70}")
print(f"=== SUMMARY TABLE ===")
print(f"{'='*70}")
print(f"\n{'Metric':35s} {'2.5 Pro':>10s} {'3.0 Pro':>10s} {'3.1 Pro':>10s}")
print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*10}")

def acc_str(records, ids, filter_fn=None):
    correct = 0
    total = 0
    for cid in ids:
        rec = records[cid]
        for t in rec["turns"]:
            if filter_fn and not filter_fn(rec, t):
                continue
            total += 1
            if t["correct"]:
                correct += 1
    if total == 0:
        return "  n/a"
    return f"{correct/total:>8.1%} ({correct}/{total})"

# Hack for shorter display
def a(records):
    return lambda ids, ff=None: acc_str(records, ids, ff)

a25, a30, a31 = a(g25), a(g30), a(g31)

print(f"{'Overall':35s} {a25(overlap_ids):>18s} {a30(overlap_ids):>18s} {a31(overlap_ids):>18s}")
for cat in ["same_flow", "switch_flow", "ambiguous_first", "ambiguous_second"]:
    ff = lambda rec, t, c=cat: rec["category"] == c
    print(f"{'  ' + cat:35s} {a25(overlap_ids, ff):>18s} {a30(overlap_ids, ff):>18s} {a31(overlap_ids, ff):>18s}")
for tn_num in [1, 3]:
    ff = lambda rec, t, n=tn_num: t["turn_num"] == n
    print(f"{'  turn ' + str(tn_num):35s} {a25(overlap_ids, ff):>18s} {a30(overlap_ids, ff):>18s} {a31(overlap_ids, ff):>18s}")
