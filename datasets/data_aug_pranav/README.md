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

---

**Providers:** Anthropic (Claude), OpenAI (GPT), Google (Gemini via OpenRouter), DeepSeek (via OpenRouter).

**Required env vars:** `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPEN_ROUTER_API_KEY` (set in `.env` at project root).
