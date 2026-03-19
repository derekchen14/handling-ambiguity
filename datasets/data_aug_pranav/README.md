# Synthetic Data Augmentation Pipeline

Step-by-step pipeline for generating synthetic training data using multiple LLM families.

## Steps

### Step 1: Generate Scenarios

Produce diverse scenario objects (description + example utterances) grounded in the domain ontology.

```bash
# Dry run — prints prompts, no API calls
uv run datasets/data_aug_pranav/generate_scenarios.py \
    --domain hugo --target 24 --dry-run

# Single-provider smoke test
uv run datasets/data_aug_pranav/generate_scenarios.py \
    --domain hugo --target 12 --models anthropic

# Full run — round-robins across all 4 providers
.venv/bin/python3 datasets/data_aug_pranav/generate_scenarios.py \
    --domain hugo --target 400 --batch-size 10 --seed 42
```

**Output:** `data/scenarios_<domain>.jsonl` — one JSON object per line with scenario description, example utterances, grounding flows/intents, diversity axis, and source model.

### Step 2: Enrich Scenarios with Flow Sequences

For each scenario, generate a natural sequence of 5-7 flows a user would progress through. These sequences later support assigning each scenario to the 4 conversation categories.

```bash
# Dry run — prints prompts, no API calls
uv run datasets/data_aug_pranav/enrich_scenarios.py \
    --domain hugo --dry-run

# Single-provider smoke test
uv run datasets/data_aug_pranav/enrich_scenarios.py \
    --domain hugo --models anthropic --batch-size 4

# Full run — enriches all unenriched scenarios across all 4 providers
.venv/bin/python3 datasets/data_aug_pranav/enrich_scenarios.py \
    --domain hugo --batch-size 8 --seed 42 --max-threads 20
```

**Input:** `data/scenarios_<domain>.jsonl` (from Step 1)

**Output:** `data/scenarios_<domain>_enriched.jsonl` — preserves all original fields and adds `flow_sequence`, `edge_flow_pairs`, `enrichment_model`, `enrichment_provider`.

Resumable — rerun the same command to pick up where it left off.

### Step 3: Semantic Deduplication

Step 1 applies Jaccard dedup (threshold 0.5), which only catches lexical overlap. This step uses LLMs to find semantic duplicates — scenarios with the same domain/topic and user goal but different wording — then backfills via Steps 1+2 to restore the original count.

**Two-phase approach:**
- **Phase 1 (within-batch):** Split scenarios into batches of ~60, each sent to an LLM to identify duplicate clusters.
- **Phase 2 (cross-batch):** Send all survivors in a single consolidation call to catch duplicates that spanned different Phase 1 batches.

```bash
# Dry run — prints prompts and batch composition, no API calls
python datasets/data_aug_pranav/dedup_scenarios.py \
    --domain hugo --dry-run

# Dedup only — inspect results before backfilling
python datasets/data_aug_pranav/dedup_scenarios.py \
    --domain hugo --skip-backfill

# Full run — dedup + backfill to restore original count
python datasets/data_aug_pranav/dedup_scenarios.py \
    --domain hugo --batch-size 60 --seed 43 --max-threads 6
```

**Input:** `data/scenarios_<domain>_enriched.jsonl` (from Step 2)

**Output:**
- `data/scenarios_<domain>_enriched_deduped.jsonl` — final deduped enriched scenarios (new file, originals untouched until backfill)
- `data/scenarios_<domain>_dedup_meta.json` — duplicate clusters, counts, and model distribution

During backfill, deduped IDs are removed from the base and enriched JSONL files, then `generate_scenarios` and `enrich_scenarios` are called to fill the gap.

### Step 4: Generate Conversations

Convert deduped enriched scenarios into 3-turn conversations (user → agent → user). Scenarios are split into 4 equal categories: `same_flow`, `switch_flow`, `ambiguous_first`, `ambiguous_second`. Models are assigned round-robin across 4 providers.

```bash
# Dry run — prints prompts, no API calls
python datasets/data_aug_pranav/generate_conversations.py \
    --domain hugo --dry-run

# Single-provider smoke test
python datasets/data_aug_pranav/generate_conversations.py \
    --domain hugo --models anthropic --seed 42

# Full run — round-robins across all 4 providers
python datasets/data_aug_pranav/generate_conversations.py \
    --domain hugo --seed 42 --max-threads 8
```

**CLI:** `--domain`, `--seed`, `--models` (comma-separated), `--max-threads`, `--dry-run`

**Input:** `data/scenarios_<domain>_enriched_deduped.jsonl` (from Step 3)

**Output:**
- `data/conversations_<domain>_raw.jsonl` — one JSON object per line, appended incrementally
- `data/conversations_<domain>.json` — sorted JSON array (final format matching eval_set.json)
- `data/conversations_<domain>_meta.json` — generation counts, category splits, and model distribution

Resumable — rerun the same command to pick up where it left off.

### Step 5: Compute Metrics

Compare synthetic conversations to the eval set across 16+ dimensions.

```bash
# Fast iteration (skip LLM judges):
.venv/bin/python3 datasets/data_aug_pranav/compute_metrics.py \
    --domain both --seed 42 --skip-llm

# Full metrics (includes naturalness + ambiguity LLM judges):
.venv/bin/python3 datasets/data_aug_pranav/compute_metrics.py \
    --domain both --seed 42 --concurrency 10
```

**Output:** `analysis/metrics_<domain>.json` — intrinsic + comparative scorecards with green/yellow/red ratings.

### Step 6: Analysis Plots

```bash
.venv/bin/python3 datasets/data_aug_pranav/analysis/analyze_synth_vs_eval.py --domain both
```

**Output:** PNG plots + `analysis/synth_vs_eval_report.md`

---

## Full Pipeline (one-shot)

```bash
# Generate for both domains (takes ~30-60 min total):
for domain in hugo dana; do
    .venv/bin/python3 datasets/data_aug_pranav/generate_scenarios.py --domain $domain --target 400 --batch-size 10 --seed 42
    .venv/bin/python3 datasets/data_aug_pranav/enrich_scenarios.py --domain $domain --batch-size 8 --seed 42 --max-threads 20
    .venv/bin/python3 datasets/data_aug_pranav/dedup_scenarios.py --domain $domain --batch-size 60 --seed 43 --skip-backfill
    .venv/bin/python3 datasets/data_aug_pranav/generate_conversations.py --domain $domain --seed 42 --max-threads 8
done

# Compute metrics:
.venv/bin/python3 datasets/data_aug_pranav/compute_metrics.py --domain both --seed 42 --skip-llm
```

**Scaling:** To generate more, increase `--target` in Step 1. Pipeline handles resume automatically — outputs are appended to existing files. Typical yield: ~80% enrichment success, ~10-15% dedup removal, giving ~300 conversations from 400 scenarios.

---

**Providers:** Anthropic (Claude Sonnet), OpenAI (GPT-5.2), Google (Gemini via OpenRouter), DeepSeek (via OpenRouter). Step 3 skips Gemini by default.

**Required env vars:** `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPEN_ROUTER_API_KEY` (set in `.env` at project root).
