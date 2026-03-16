# Synthetic Data Augmentation Pipeline

Step-by-step pipeline for generating synthetic training data using multiple LLM families.

## Steps

### Step 1: Generate Scenarios

Produce diverse scenario objects (description + example utterances) grounded in the domain ontology.

```bash
# Dry run — prints prompts, no API calls
python datasets/data_aug_pranav/generate_scenarios.py \
    --domain hugo --target 24 --dry-run

# Single-provider smoke test
python datasets/data_aug_pranav/generate_scenarios.py \
    --domain hugo --target 12 --models anthropic

# Full run — round-robins across all 4 providers
python datasets/data_aug_pranav/generate_scenarios.py \
    --domain hugo --target 200 --batch-size 12 --seed 42
```

**Output:** `scenarios_<domain>.jsonl` — one JSON object per line with scenario description, example utterances, grounding flows/intents, diversity axis, and source model.

**Providers:** Anthropic (Claude), OpenAI (GPT), Google (Gemini via OpenRouter), DeepSeek (via OpenRouter).

**Required env vars:** `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPEN_ROUTER_API_KEY` (set in `.env` at project root).
