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
uv run datasets/data_aug_pranav/generate_scenarios.py \
    --domain hugo --target 200 --batch-size 12 --seed 42
```

**Output:** `scenarios_<domain>.jsonl` — one JSON object per line with scenario description, example utterances, grounding flows/intents, diversity axis, and source model.

### Step 2: Enrich Scenarios with Flow Sequences

For each scenario, generate a natural sequence of 5-7 flows a user would progress through. These sequences later support assigning each scenario to the 4 conversation categories.

```bash
# Dry run — prints prompts, no API calls
uv run datasets/data_aug_pranav/enrich_scenarios.py \
    --domain hugo --dry-run

# Single-provider smoke test
uv run datasets/data_aug_pranav/enrich_scenarios.py \
    --domain hugo --models anthropic --batch-size 4

# Full run — splits 384 scenarios across all 4 providers (96 each)
uv run datasets/data_aug_pranav/enrich_scenarios.py \
    --domain hugo --max-threads 20
```

**Input:** `scenarios_<domain>.jsonl` (from Step 1)

**Output:** `scenarios_<domain>_enriched.jsonl` — preserves all original fields and adds `flow_sequence`, `edge_flow_pairs`, `enrichment_model`, `enrichment_provider`.

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

**Input:** `scenarios_<domain>_enriched.jsonl` (from Step 2)

**Output:**
- `scenarios_<domain>_enriched_deduped.jsonl` — final deduped enriched scenarios (new file, originals untouched until backfill)
- `scenarios_<domain>_dedup_meta.json` — duplicate clusters, counts, and model distribution

During backfill, deduped IDs are removed from the base and enriched JSONL files, then `generate_scenarios` and `enrich_scenarios` are called to fill the gap.

---

**Providers:** Anthropic (Claude), OpenAI (GPT), Google (Gemini via OpenRouter), DeepSeek (via OpenRouter). Step 3 skips Gemini by default.

**Required env vars:** `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPEN_ROUTER_API_KEY` (set in `.env` at project root).
