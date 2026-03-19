# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project comparing two NLU architectures for multi-turn dialogue:
- **Staged Funnel** (Exp 1): Intent → Flow Detection → Tool Selection (progressive scope reduction)
- **Flat Tool-Calling** (Exp 2): Direct tool selection from utterance

Two domains: **Hugo** (blogging, 42 flows) and **Dana** (data analysis, ~48 flows). Each domain has an ontology, tool manifest, eval set, and flow-tool mapping.

## Commands

```bash
# Setup
uv sync                              # Install core deps
uv sync --extra training             # Include training deps (wandb, sglang, peft)

# Tests
pytest tests_i/                      # Run all tests
pytest tests_i/test_score_tool_params.py  # Single test file

# Experiment runners
uv run exp1_runner.py --domain hugo --config 1a_004 --seeds 1
uv run exp2_runner.py --domain hugo --config 2_001 --seeds 1 --mode tool

# Synthetic data pipeline (use .venv python for deps)
.venv/bin/python3 datasets/data_aug_pranav/enrich_scenarios.py --domain hugo --seed 42
.venv/bin/python3 datasets/data_aug_pranav/generate_conversations.py --domain hugo --seed 42
.venv/bin/python3 datasets/data_aug_pranav/enrich_scenarios.py --domain hugo --backfill-tools --seed 42

# Training
uv run python -m training.train_nlu --mode sft --stages flow --domain hugo \
    --model_name Qwen/Qwen3-0.6B --data_path datasets/hugo/eval_set.json

# Adding packages
uv add ,,,
```

All scripts expect a `.env` file at project root with `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPEN_ROUTER_API_KEY`.

## Architecture

### Pipeline Stages (the "funnel")
```
Utterance → Intent (1 of 6) → Flow (~12 candidates) → Tool (5-7 per flow) → Params
```
Each stage is a prompt builder in `prompts/` that returns (system_prompt, user_prompt). The `helpers/harness.py` `ExperimentRunner` orchestrates running these across eval sets.

### Key Abstractions

- **`helpers/client.py` — `UnifiedLLMClient`**: Multi-provider LLM client (Anthropic, Google, OpenAI, OpenRouter, Qwen, DeepSeek) with per-provider rate limiting, tier-based model registry (low/medium/high), and retry logic.
- **`helpers/harness.py` — `ExperimentRunner`**: Runs experiments with checkpointing, parallel processing, and result aggregation. Methods: `run_exp1a()`, `run_exp1b()`, `run_exp2_*()`.
- **`helpers/scoring.py`**: Centralized scoring — `score_turn()` for clear/ambiguous turns, `score_turn_ensemble()` for multi-voter, `score_tool_turn()` for tool-calling, `build_fuzzy_evaluator()` for semantic parameter matching.

### Domain Ontology Structure

Each domain (`datasets/{hugo,dana}/ontology.py`) defines:
- `FLOW_CATALOG` — dict of flows, each with `intent`, `dax` (semantic action code), `description`, `slots`, `edge_flows`, `policy_path`
- `DACT_CATALOG` — dialogue act definitions
- `Intent` — enum of intents (6 per domain)
- `eval_set.json` — 128 conversations across 4 categories: `same_flow`, `switch_flow`, `ambiguous_first`, `ambiguous_second`

### Tool Manifests (`tools/`)

`tool_manifest_{domain}.json` — array of tool definitions with `name`, `description`, `input_schema`, `_flows` (which flows use this tool), `internal_component` (boolean). Flow-tool mappings in companion `.md` files.

### Synthetic Data Pipeline (`datasets/data_aug_pranav/`)

4-step pipeline, each script reads the previous step's output:
1. `generate_scenarios.py` — Generate scenario seeds from ontology
2. `enrich_scenarios.py` — Add flow sequences via multi-LLM (4 providers), with anchor flow round-robin for uniformity, tool assignment via round-robin cursor, and `--backfill-tools` for existing data
3. `dedupe_scenarios.py` — Remove near-duplicates
4. `generate_conversations.py` — Generate 3-turn conversations per category with pre-assigned tool constraints in prompts

### Training (`training/`)

- `stages.py` — `PipelineStage` abstraction with `STAGE_REGISTRY`; stages can be composed with weighted rewards
- `train_nlu.py` — Entry point for SFT and RL/PPO modes
- `utils/trainer.py` — PPOTrainer + `run_sft()` helper
- `utils/dataset.py` — Dataset classes for RL trajectories and SFT examples

## Experiment Configs

Stored in `helpers/configs/`:
- `exp1a_configs.json` — 30+ model configs (provider, tier, model_id, temperature)
- `exp2_configs.json` — 8 configs for pipeline mode comparison
- `exp1b_ensembles_resolved.json` — Ensemble definitions

## Conventions

- Conversation categories: `same_flow`, `switch_flow`, `ambiguous_first`, `ambiguous_second`
- Intents — Hugo: Research, Draft, Revise, Publish, Converse, Plan; Dana: Clean, Transform, Analyze, Report, Converse, Plan
- Plan and Internal intents are excluded from user-facing flow sets
- Results output as JSONL (per-conversation) + JSON summary; organized under `results/exp{1a,1b,2a,2b,2c}/`
- Dax codes are 6-bit hex semantic tokens encoding flow actions (e.g., `{012}` = chat+search+outline)
