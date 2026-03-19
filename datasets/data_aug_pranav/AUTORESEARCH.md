# Autonomous Research Agent — Synthetic Data Optimization

You are an expert autonomous researcher. Your mission is to iteratively optimize the quality and diversity of synthetic training data for two NLU domains (Hugo and Dana). You will run the pipeline, measure quality, identify weaknesses, modify prompts and pipeline code, re-run, and repeat until convergence. The end goal is a fully documented, production-ready pipeline that can be run independently to generate millions of high-quality, extremely diverse training examples.

## 0. Bootstrap

Before doing anything else:

1. Read `CLAUDE.md` at the project root — it describes the full project architecture.
2. Read `datasets/data_aug_pranav/README.md` and every `.py` file in that directory to understand the pipeline.
3. Read `datasets/data_aug_pranav/notes.md` for prior experiment history.
4. Ensure `.env` exists at the project root with: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPEN_ROUTER_API_KEY`.

---

## 1. Pipeline Reference

The synthetic data pipeline has 4 steps. Each step reads the previous step's output. All commands run from the **project root**.

### Step 1 — Generate Scenarios
Produce diverse scenario objects (description + example utterances) grounded in domain flows.

```bash
.venv/bin/python3 datasets/data_aug_pranav/generate_scenarios.py \
    --domain {domain} --target {N} --batch-size 12 --seed 42
```

- Output: `datasets/data_aug_pranav/data/scenarios_{domain}.jsonl`
- Uses 4 LLM providers round-robin: anthropic (claude-sonnet-4-6), openai (gpt-5.2), gemini (google/gemini-3-pro-preview via openrouter), deepseek (deepseek/deepseek-chat via openrouter)
- Supports resume: if output file exists, only generates the delta to reach `--target`
- Add `--dry-run` to print prompts without API calls

### Step 2 — Enrich Scenarios
Add natural flow sequences (5-7 flows), edge flow pairs, and tool assignments to each scenario.

```bash
.venv/bin/python3 datasets/data_aug_pranav/enrich_scenarios.py \
    --domain {domain} --batch-size 8 --seed 42 --max-threads 20
```

- Input: `data/scenarios_{domain}.jsonl`
- Output: `data/scenarios_{domain}_enriched.jsonl`
- Uses anchor flow round-robin for uniform coverage
- Tool assignment via round-robin cursor across flows

To backfill tool assignments on already-enriched data:
```bash
.venv/bin/python3 datasets/data_aug_pranav/enrich_scenarios.py \
    --domain {domain} --backfill-tools --seed 42
```

### Step 3 — Deduplicate Scenarios
Semantic deduplication using LLM-based analysis (catches near-duplicates Jaccard missed).

```bash
.venv/bin/python3 datasets/data_aug_pranav/dedup_scenarios.py \
    --domain {domain} --batch-size 60 --seed 43
```

- Input: `data/scenarios_{domain}_enriched.jsonl`
- Output: `data/scenarios_{domain}_enriched_deduped.jsonl`
- Two-phase: within-batch dedup, then cross-batch consolidation
- Add `--skip-backfill` to dedup without regenerating removed scenarios

### Step 4 — Generate Conversations
Generate 3-turn conversations (user -> agent -> user) from enriched scenarios, one per category.

```bash
.venv/bin/python3 datasets/data_aug_pranav/generate_conversations.py \
    --domain {domain} --seed 42 --max-threads 8
```

- Input: `data/scenarios_{domain}_enriched_deduped.jsonl`
- Output: `data/conversations_{domain}_raw.jsonl` (intermediate) + `data/conversations_{domain}.json` (final, sorted)
- Categories: `same_flow`, `switch_flow`, `ambiguous_first`, `ambiguous_second` (equal split)
- Each scenario gets assigned to exactly one category with pre-assigned flow and tool constraints

### Metrics — Compute Scorecard
```bash
# Full metrics (always run full — no --skip-llm):
.venv/bin/python3 datasets/data_aug_pranav/compute_metrics.py \
    --domain {domain} --seed 42 --concurrency 10

# Both domains:
.venv/bin/python3 datasets/data_aug_pranav/compute_metrics.py \
    --domain both --seed 42 --concurrency 10
```

- Output: `datasets/data_aug_pranav/analysis/metrics_{domain}.json`

### Analysis — Plots + Report
```bash
.venv/bin/python3 datasets/data_aug_pranav/analysis/analyze_synth_vs_eval.py \
    --domain both
```

- Output: PNG plots + `analysis/synth_vs_eval_report.md`

---

## 2. What to Optimize

### Primary Scorecard: Section I Quality Heuristics + Diversity

These are the **only** optimization targets. No comparative metrics — we do not optimize synth-vs-eval distribution similarity.

| Metric | What it measures | Green | Red | Direction |
|--------|-----------------|-------|-----|-----------|
| **Diversity** | | | | |
| `flow_entropy_ratio` | Flow uniformity (entropy / max_entropy) | >= 0.85 | < 0.70 | Higher = better |
| `tool_entropy_ratio` | Tool uniformity | >= 0.85 | < 0.70 | Higher = better |
| **Quality — Regex Heuristics** | | | | |
| `heuristic_filler_rate` | Filler phrases in user turns (I.2) | <= 0.05 | > 0.15 | Lower = better |
| `heuristic_overack_rate` | Agent over-acknowledgment (I.2) | <= 0.05 | > 0.15 | Lower = better |
| `heuristic_unicode_rate` | Curly quotes, em-dashes, etc. (I.7) | <= 0.02 | > 0.10 | Lower = better |
| `heuristic_leakage_rate` | Flow/intent/tool names in user utterances (I.8) | <= 0.02 | > 0.10 | Lower = better |
| `heuristic_multireq_rate` | ambiguous_second turn-3 has multi-req connector (I.6) | >= 0.90 | < 0.70 | Higher = better |
| `heuristic_t3_terse_rate` | Turn-3 user utterances ≤9 words (I.2) | >= 0.70 | < 0.50 | Higher = better |
| **Quality — LLM Judges** | | | | |
| `turn_dependency_mean` | Turn-3 depends on prior context (I.1) | >= 3.5 | < 2.5 | Higher = better |
| `agent_leak_rate` | Agent turn leaks labels/metadata (I.2) | <= 0.10 | > 0.25 | Lower = better |
| `naturalness_mean` | Mean naturalness score (LLM judge) | >= 3.5 | < 2.5 | Higher = better |
| **Label Quality** | | | | |
| `label_agreement_intent` | Multi-model intent label agreement | >= 0.95 | < 0.85 | Higher = better |
| `label_agreement_flow` | Multi-model flow label agreement | >= 0.85 | < 0.70 | Higher = better |
| `label_agreement_tool` | Multi-model tool label agreement | >= 0.85 | < 0.70 | Higher = better |

**ALL metrics must be measured every iteration.** Always run `compute_metrics.py` without `--skip-llm`. The LLM-based metrics (turn dependency, agent leak, naturalness, label agreement) are primary targets, not optional checkpoints.

---

## 3. Iteration Protocol

### Batch Size
- **Prefer the smallest batch that gives signal** — go lower for speed, only up to 40 if necessary
- Optimize for iteration speed, not throughput

### Loop

```
for each iteration:
    1. Run pipeline steps 1-4 for {domain}
    2. Run full metrics (compute_metrics.py without --skip-llm, always)
    3. Read the scorecard JSON
    4. Read a sample of the actual generated conversations — don't just trust aggregate numbers.
       Look at real examples to spot issues metrics miss (awkward phrasing, repetitive patterns,
       unrealistic tool usage, etc.). Use what you see to motivate next steps, new metrics, and
       hypotheses.
    5. Identify the worst 2-3 signals (from scorecard AND manual inspection)
    6. Hypothesize a fix (prompt change, filtering step, pipeline modification, new metric)
    7. Implement the fix
    8. Re-run pipeline and compare scorecards
    9. Log result in notes.md
   10. If breakthrough: commit + push
```

### Reading the Scorecard

After running `compute_metrics.py`, read:
```bash
cat datasets/data_aug_pranav/analysis/metrics_{domain}.json
```

Focus on:
- `intrinsic_scorecard` → the color-coded pass/warn/fail for each metric
- Look at individual metric values, not just pass/fail
- Pay special attention to LLM judge metrics — they catch issues regex can't

### What Counts as Progress
- Any metric moving from red -> yellow or yellow -> green
- Multiple metrics improving without any regressing
- Consistency across both domains (hugo AND dana)
- Label agreement improving (intent toward 0.95, flow/tool toward 0.85)
- Heuristic violation rates decreasing
- LLM judge scores improving (turn dependency, naturalness up; agent leak down)

---

## 4. What You Can Change

### Full Freedom
- **Prompts**: Full freedom to modify any prompt in any pipeline step. Add context, few-shot examples, chain-of-thought, constraints, richer instructions — whatever improves output quality and diversity. This is encouraged and expected.
- **Pipeline structure**: Add new steps, remove steps, modify control flow. If you change pipeline structure, update BOTH this file's Pipeline Reference section AND `datasets/data_aug_pranav/README.md` so the pipeline can be run independently.
- **Filtering**: Add filtering between steps (e.g., reject scenarios by embedding distance, naturalness score, or other quality signals)
- **Packages**: `uv add {package}` — log what you added and why in `notes.md`

### With Care
- **Scorecards** (`compute_metrics.py` thresholds/logic): Only to fix bugs or genuinely improve measurement. NEVER to inflate scores. If you change thresholds or scoring logic, document exactly what changed and why in `notes.md`.
- **Adding intrinsic metrics**: You can add new intrinsic metrics to `compute_metrics.py` if they capture something genuinely useful that existing metrics miss (e.g., something you noticed by reading examples). The bar: the metric must be hard to game — it should only go green when the data is actually better, not because the pipeline learned to exploit a shortcut. Document new metrics in `notes.md` and add them to the scorecard tables in this file.
- **Pipeline file I/O paths**: Only if you're restructuring the pipeline and updating all downstream consumers.

### Do Not Change
- `datasets/{hugo,dana}/eval_set.json` — these are ground truth
- `datasets/{hugo,dana}/ontology.py` — domain definitions are fixed
- `tools/tool_manifest_{domain}.json` — tool definitions are fixed
- `tools/flow_tool_mapping_{domain}.md` — flow-tool mappings are fixed

---

## 5. Label Disagreement Handling

When running full metrics (without `--skip-llm`), the label agreement checks use multiple LLM judges. Handle disagreements as follows:

| Situation | Action |
|-----------|--------|
| Models agree with each other, disagree with the assigned label | **Flip the label** to match the models. Log the flip in `notes.md` (convo_id, old label, new label, which models agreed). |
| Models disagree with each other AND disagree with the label | **Quarantine** — flag for human review. Write quarantined IDs to `notes.md` with the disagreement details. |
| Models disagree with each other entirely (no majority) | **Quarantine** — flag for human review. Same logging. |

When flipping labels, modify the conversation JSON file directly. When quarantining, do NOT delete the conversation — just log it. I will review quarantined items manually.

---

## 6. Required Optimization Techniques

These are core techniques you MUST use in the optimization loop, not optional ideas. Apply them every iteration:

- **Embedding diversity filtering**: After Step 4, compute pairwise cosine similarities of generated utterances. Reject conversations whose utterances are all within a tight cluster (e.g., mean pairwise similarity > 0.85). This directly boosts `flow_entropy_ratio` and `tool_entropy_ratio`.

- **Naturalness-gated generation**: After generating a batch, score naturalness (quick LLM judge call). Reject conversations with naturalness < 3.0 and regenerate. This directly improves `naturalness_mean`.

- **Closed-loop prompt optimization**: After each iteration, inject the worst-performing metric names and values into the generation prompts. E.g., if `tool_entropy_ratio` is 0.72, tell the generator "Current tool coverage is uneven — prioritize underrepresented tools: [list]."

- **Targeted underrepresentation**: If certain flows/tools have low representation in the scorecard, bias Step 1 scenario generation toward those flows by adjusting `grounding_flows` sampling weights.

- **Utterance style transfer**: If naturalness is low because utterances sound robotic, add few-shot examples of natural, terse, phone-typed messages to the system prompt. Pull examples from the eval set (but paraphrase, don't copy verbatim).

- **Category-aware quality**: Break down metrics by category. If `ambiguous_first` has worse naturalness than `same_flow`, the ambiguity generation prompts need work specifically.

These are starting points. You are expected to invent new optimization strategies beyond this list — any technique that meaningfully improves the scorecard is fair game.

---

## 7. Logging & Checkpointing

### notes.md

`datasets/data_aug_pranav/notes.md` is your lab notebook. Log every iteration:

```markdown
## Iteration N — {date} — {domain}

**Target**: {which metric(s) you're trying to improve}
**Change**: {what you modified — which prompt, which step, what filtering}
**Result**:
- heuristic_filler_rate: 0.12 -> 0.04
- turn_dependency_mean: 2.8 -> 3.6
- (list all changed metrics)
**Verdict**: {worked / partially worked / didn't work / made things worse}
```

### Commits

On every meaningful breakthrough (metric goes green, or multi-metric improvement):

```bash
git add datasets/data_aug_pranav/
git commit -m "synth data: {brief description of what improved}"
git push
```

Include the scorecard delta in the commit message body.

### Checkpoints

Every 3 iterations, run full metrics for both domains and log the complete scorecard in `notes.md`.

---

## 8. Domain Notes

### Both Domains Required
Run both `hugo` and `dana` every iteration. Don't optimize one and forget the other.

### Domain Characteristics
- **Hugo** (blogging): 42 user-facing flows, intents = Research, Draft, Revise, Publish, Converse
- **Dana** (data analysis): ~48 user-facing flows, intents = Clean, Transform, Analyze, Report, Converse
- Plan and Internal intents are excluded from user-facing flow sets

### Narrow Domains Get More Attention
If one domain has fewer flows or less natural diversity, it needs MORE iterations to reach entropy thresholds, not fewer. Don't give up on a narrow domain — work harder on prompt diversity.

---

## 9. Stopping Condition

Stop when ALL of the following are true:

1. **Convergence**: 5 consecutive iterations (for each domain) show < 5% relative improvement across ALL scorecard metrics (diversity, heuristic, and LLM judge metrics).
2. **All metrics are green or yellow** — no reds remaining in the intrinsic scorecard.
3. **Both domains** have reached convergence independently.

When you stop:

1. Write a final summary in `notes.md`:
   - Total iterations per domain
   - Starting vs ending scorecard values
   - Key breakthroughs and what caused them
   - Remaining yellow metrics and why they're hard to push further
2. Run full metrics + analysis for both domains (final checkpoint)
3. Commit and push everything
4. Open a GitHub issue summarizing convergence: `gh issue create --title "Synth data optimization converged" --body "..."`

---

## 10. Constraints

- **No cost cap** — optimize for quality and speed of iteration, not cost
- **No time cap** — run as many iterations as needed to converge
- **Batch size <= 40** per domain per iteration
- **Always run from project root** — all paths are relative to project root
- **Use `.venv/bin/python3`** for running pipeline scripts (not `python` or `uv run` for the data aug scripts)
- **Seed consistency**: use `--seed 42` for all pipeline steps unless you have a specific reason to vary it (e.g., `--seed 43` for dedup is fine as established)
- **Always run full metrics** — never use `--skip-llm`. LLM judge metrics are primary targets, not optional.
