# Confidence Score Experiment — Plan v0.6

Status: **DRAFT — iterating on plan structure**

---

## 0. Context

### The Narrowing Funnel

The NLU pipeline is designed as progressive scope reduction:

```
User utterance
  ↓
Intent classification: choose 1 of 6 intents
  ↓
Flow detection: choose 1 of ~12 candidate flows
  (intent's flows + edge flows from adjacent intents)
  ↓
Tool selection: flow determines 5-7 available tools
  (3 component tools + 1-3 domain-specific tools)
  ↓
Policy execution with the narrowed tool set
```

Each step reduces the decision space. Intent narrows 42 flows to ~12 candidates. Flow narrows ~40+ tools to 5-7. The question is whether this staged narrowing adds value over a flat approach.

### Current State Across Domains

| Assistant | Domain | Total Flows | User-Facing (excl. Plan, Internal) | Domain Tools (current) | Domain Tools (per spec) | Status |
|-----------|--------|-------------|-------------------------------------|:----------------------:|:-----------------------:|--------|
| **Hugo** | Blogging | 42 | 31 | 10 | 42+ | Working (tools incomplete) |
| **Dana** | Data Analysis | ~48 | ~35 | 0 | ~48+ | **Blocked** |

Two domains selected for paper viability: blogging (content creation) and data analysis (technical/analytical). Kalli (onboarding meta-agent) excluded — hard to explain in a paper since it's a meta-agent that builds other assistants.

**Critical gap**: The tool_smith spec says "a domain has at least as many tools as flows." Currently tools are grouped by intent (oversight); they should be scoped by flow (1-3 domain tools per flow). The shared backbone tools exist; dedicated per-flow tools have not been designed.

### Publication Path

1. **Internal validation**: Run experiments on Hugo + Dana (this document)
2. **External validation**: Replicate on a public dataset (e.g., ATIS, SNIPS, or a multi-domain task-oriented dialogue benchmark)
3. **Arxiv**: If results hold across internal + external → publish

### Experiment Overview

This document describes **two experiments**, each with sub-phases, run across **2 domains** on **all user-facing flows** per domain:

| | Experiment 1: NLU Confidence | Experiment 2: Tool-Calling vs. Funnel |
|---|---|---|
| **Question** | Which model config gives the best-calibrated confidence for flow detection? | Can a single tool-calling LLM replace the full NLU funnel? |
| **Architecture** | Staged funnel: 6 intents → ~12 flows → 5-7 tools | Flat: utterance → choose from all ~40 tools directly |
| **Sub-phases** | 1A: single-model accuracy → 1B: 28 voting ensembles → 1C: calibrate top 10 ensembles + top 4 single models | Single-phase (uses Exp 1A models) |
| **Blog angle** | Parameter sensitivity + progressive narrowing | "We built a 3-stage funnel. Then we tried skipping it." |

---

## A. Goal

### Primary questions

**Experiment 1A** — Which individual model (type × level × thinking effort) is most accurate at flow detection across the full user-facing catalog (~35 flows per domain)? Is a single strong model (e.g., Opus 4.6 + high thinking) sufficient?

**Experiment 1B** — Do multi-voter ensembles always outperform the best single voter? What ensemble compositions and temperatures give the best accuracy boost?

**Experiment 1C** — Among the top 10 ensembles + top 4 single-model baselines (14 configs total), which produce the best-calibrated confidence scores?

**Experiment 2** — If we skip the funnel and present all ~40 tools to a single LLM, how does accuracy compare?

### Sub-questions
1. Do optimal configs transfer across domains, or is the answer domain-specific?
2. Is the marginal accuracy of 3 voters over 1 voter worth the cost?
3. What about 5 or 10 voters — where does the return diminish?
4. Which parameter (model type, level, thinking) has the largest effect on accuracy?
5. Among calibrated configs, which has the best accuracy × cost tradeoff?

### Why this matters
- **Overconfident** → agent picks wrong flow silently → frustration
- **Underconfident** → too many clarification questions → feels dumb
- If 1 strong model matches 3-voter ensemble, we save 2/3 of NLU cost
- If tool-calling matches NLU, the entire funnel is unnecessary

### What we're NOT testing
- Intent classification in isolation (held constant in Exp 1; eliminated in Exp 2)
- Slot-filling accuracy in isolation (Exp 2 tests it implicitly via tool arguments)
- End-to-end task completion or policy execution quality
- Prompt wording variations

### Success looks like
1. Clear accuracy ranking of individual models across 2 domains
2. Quantified value of voting (is 3 voters always better than 1?)
3. 2-3 calibrated configs that Pareto-dominate on accuracy × ECE × cost
4. Determination of whether tool-calling matches or falls short of the funnel
5. Chart-ready data for a paper (internal domains) + reproducibility on an external dataset

---

## B. Prerequisite: Flow-Specific Tool Design

Before Experiment 2 can run, every flow in each domain needs tool bindings — not just the 16 evaluated flows. This is a production fix, not just an experiment prerequisite. The tools don't need to be functional; we only need descriptions of what each tool *would* do (name, description, input schema).

### B.1 What Exists vs. What's Needed

**Current** (flow-specific tool manifests built):
- Hugo: 41K tool manifest (flow-specific bindings)
- Dana: 38K tool manifest (flow-specific bindings)

**Needed** (per tool_smith spec): 1-3 domain tools per flow + 3 component tools = 5-7 total per flow.

### B.2 Tool Design Process (from tool_smith.md)

For every flow in each domain (42+ for Hugo, ~48 for Dana):
1. Identify external system (API/DB/service)
2. Identify CRUD operation (search, get, create, update, etc.)
3. Map flow slots to tool parameters
4. Name using `entity_verb` pattern
5. Write input/output JSON Schema

### B.3 Example: Hugo Flow-Tool Mapping (Sample)

| Flow | Intent | Domain Tools (proposed) | Total w/ components |
|------|--------|------------------------|:-------------------:|
| search | Research | post_search | 4 |
| browse | Research | post_search, topic_list | 5 |
| view | Research | post_get | 4 |
| check | Research | post_search, status_summary | 5 |
| outline | Draft | content_generate, post_create | 5 |
| create | Draft | post_create, content_generate | 5 |
| expand | Draft | post_get, content_generate, post_update | 6 |
| write | Draft | post_get, content_generate, post_update | 6 |
| brainstorm | Draft | content_generate | 4 |
| rework | Revise | post_get, content_generate, post_update | 6 |
| polish | Revise | post_get, content_generate, post_update | 6 |
| format | Revise | post_get, content_format, post_update | 6 |
| release | Publish | post_get, platform_publish | 5 |
| syndicate | Publish | post_get, content_format, platform_publish | 6 |
| chat | Converse | post_search | 4 |
| next | Converse | post_search, status_summary | 5 |

The above is a representative sample. Full flow-tool mappings for all user-facing flows are in `experiments/tools/tool_manifest_{domain}.json`.

---

## C. Experiment 1A — Single-Model Accuracy (Appendix)

### C.1 Purpose

Establish the baseline accuracy of every individual model at flow detection across the full catalog. No voting, no temperature variation. Pure model capability.

**Hypothesis to test**: "Maybe Opus 4.6 with high thinking effort is all you need."

### C.2 Parameters

| # | Parameter | Levels | Count |
|---|-----------|--------|:-----:|
| 1 | Model Type | Claude, Gemini, OpenAI (full) + Qwen (reduced) | 4 providers |
| 2 | Model Level | Low, Medium, High | 3 |
| 3 | Thinking Effort | Low, High, Extended | 3 (core providers only) |

**Temperature**: Fixed at 0.0 (deterministic; no need to vary for single-model accuracy).

**Qwen runs a reduced sweep**: 3 model levels only, no thinking effort variation, evaluated on the full eval set (256 labels). This is a sanity check to see where Qwen ranks, not a full parameter exploration.

#### Model Mapping (5 Providers)

| Level | Claude | Google | OpenAI | Qwen | DeepSeek |
|-------|--------|--------|--------|------|----------|
| Low | Haiku 4.5 | Gemma 27B | GPT-5-nano | Qwen2.5-7B | DeepSeek-V3 (chat) |
| Medium | Sonnet 4.6 | Gemini Flash 3 | GPT-5-mini | Qwen3-80B | DeepSeek-V3 (chat) |
| High | Opus 4.6 | Gemini Pro 3.1 | GPT-5.2 | Qwen3-235B | DeepSeek-R1 (reasoner) |

> **Note**: Google has two model families — **Gemini** (proprietary, via Gemini API) and **Gemma** (open-weight, separate API path). Gemma 27B is Google's low-tier entry; Gemini Flash is mid-tier; Gemini Pro is high-tier. In experiment configs, Gemma uses `provider="gemma"` for API routing but belongs to the Google provider for analysis.

Thinking/reasoning effort was tested (Claude `budget_tokens`, OpenAI `reasoning_effort`, Gemini thinking config) but showed negligible accuracy gains (+1-2pp) at 2× cost. All 1A configs run with thinking disabled.

### C.3 Configs

One config per model, 15 total. Each config specifies `{provider, model_id, temperature}` — no thinking/effort parameters.

### C.4 Run Count

| Item | Count |
|------|------:|
| Configs | 15 |
| × Domains | × 2 |
| × Seeds | × 5 |
| = Runs | **150** |
| Labels/run | 256 (128 convos × 2 user turns) |
| = API calls | **38,400** |

All 15 models are evaluated in 1A. Several 1B ensembles include DeepSeek voters (5v-1, 5v-2, 10v-1), and their 1A predictions enable free bootstrapping.

### C.5 What We Measure

- **Top-1 accuracy** per config per domain
- **Per-intent accuracy** (which intents are hardest?)
- **Confusion matrix** (which flows get confused?)
- **Latency** (p50, p95)
- **Cost** (tokens per utterance)

### C.6 What We Learn

Rank all 15 models by accuracy. Identify:
- Best model per provider
- Best model overall
- Whether results are consistent across domains
- Provider-level patterns (do all models from one provider cluster together?)

### C.7 Internal Diagnostic: Seed Agreement (not in paper)

All 1A configs run at temp=0 (deterministic). With 5 seeds, the same model should produce the same answer for the same utterance every time. But server-side non-determinism (batching, floating point, GPU scheduling) may cause occasional disagreements.

**Metric**: Fleiss' kappa across 5 seeds, per config. Measures inter-seed agreement treating each seed as an independent "annotator."

**What we expect**: κ ≈ 1.0 for all configs. Anything below 0.95 is worth investigating.

**What we're curious about**:
- Does non-determinism vary by provider? (e.g., is Gemini more deterministic than OpenAI?)
- Does it vary by model level? (e.g., are larger models more stable?)
- Which utterances trigger disagreements? (probably the hard/confusable ones)
- Per-domain comparison: is one domain more susceptible to non-determinism?

**Cost**: Zero — uses existing 1A data. Pure analysis.

This is a sanity check for ourselves. If κ is consistently high, it confirms that seed variation at temp=0 is negligible and our 3-of-5 sampling strategy is sound. If κ is unexpectedly low for certain providers, that's a finding worth noting internally (and it means those providers contribute "free" diversity even at temp=0, which has implications for self-consistency ensemble design).

### C.8 Output → Feeds into 1B

1. **Ranked model list**: Top ~10-12 models become the voter pool for ensemble composition. Weak models (accuracy < 0.60) eliminated.
2. **Per-utterance prediction data**: Every model's prediction for every utterance across 5 seeds. This is the raw data that 1B's cross-model ensembles bootstrap from — no additional API calls needed for those ensembles.

---

## D. Experiment 1B — Voting Ensembles

### D.1 Purpose

Test whether multi-voter ensembles outperform the best single model from 1A. Each ensemble is a specific composition of voters — diversity comes from **different models**, **different temperatures**, **different thinking efforts**, or some combination thereof. Temperature is no longer a separate sweep dimension; it's embedded in the ensemble design.

**Core hypothesis**: Having 3+ voters always outperforms a single voter.
**Counter-hypothesis**: A single strong model (Opus 4.6 + high thinking) is all you need.

**Confidence rule**: In all cases, voter consistency (agreement) = confidence. More voters agreeing on the same flow → higher confidence score.

### D.2 Sources of Diversity

An ensemble achieves diversity through one or more of these axes:

| Diversity axis | How it creates variation | Example |
|----------------|--------------------------|---------|
| **Provider** | Different architectures, training data | Opus + GPT-5.2 + Gemini Pro + QwQ + DeepSeek-R1 |
| **Family level** | Same provider, different capability tiers | Opus + Sonnet + Haiku |
| **Temperature** | Same model, temp > 0 causes stochastic variation | Opus × 3 at temp=0.2 |
| **Thinking effort** | Same model, different reasoning depth | Opus standard + Opus high effort + Opus temp=0.1 |
| **Prompt** | Slight variations in prompt wording | (Exp 1B only tests this at 5-voter level) |

### D.3 Ensemble Compositions

#### 3 Voters — 16 Combinations (8 hand-picked + 8 data-driven)

**Hand-picked ensembles** (themes chosen to isolate specific diversity axes):

| ID | Voter 1 | Voter 2 | Voter 3 | Theme |
|----|---------|---------|---------|-------|
| 3v-1 | Opus × 3 @ temp=0.2 | | | **Temperature diversity**: single strong model, stochastic variation |
| 3v-2 | Sonnet × 3 @ temp=0.2 | | | **Temperature diversity**: mid-tier, stochastic variation |
| 3v-3 | Gemini Flash × 3 @ temp=0.2 | | | **Temperature diversity**: non-Claude, stochastic variation |
| 3v-4 | Opus | GPT-5.2 | Gemini Pro | **Provider diversity**: top 3 providers at default settings |
| 3v-5 | Opus | Sonnet | Haiku | **Claude family**: level diversity within one provider |
| 3v-6 | Gemini Pro | Gemini Flash | Gemma 27B | **Google family**: level diversity within one provider |
| 3v-7 | GPT-5.2 | GPT-5-mini | GPT-5-nano | **OpenAI family**: level diversity within one provider |
| 3v-8 | Opus @ temp=0.0 | Opus @ temp=0.1 | Opus @ temp=0.2 | **Single model, mixed temps**: temperature diversity |

**Data-driven ensembles** (composed after 1A results):

| ID | Composition | Selection method | Objective |
|----|-------------|------------------|-----------|
| 3v-9 | Top 3 from 1A | Highest individual accuracy | Accuracy |
| 3v-10 | Top 3 by accuracy, one per provider | Best-of-each, guaranteed provider diversity | Accuracy |
| 3v-11 | 1st + 2nd + most-different-from-1st | Maximize prediction disagreement | Diversity |
| 3v-12 | 3 models with lowest pairwise agreement | Maximum complementary errors | Diversity |
| 3v-13 | Top 3 from 1A (Accuracy) | (TBD after 1A) | Accuracy |
| 3v-14 | Top 3 from 1A (Accuracy) | (TBD after 1A) | Accuracy |
| 3v-15 | Top 3 from 1A (Diversity) | (TBD after 1A) | Diversity |
| 3v-16 | Top 3 from 1A (Diversity) | (TBD after 1A) | Diversity |

Half of the 8 data-driven slots maximize accuracy (pick the strongest individuals), half maximize diversity (pick models that disagree the most). This makes the story cleaner: does assembling the best individuals or the most diverse individuals produce the better ensemble?

#### 5 Voters — 8 Combinations (4 hand-picked + 4 data-driven)

**Hand-picked ensembles**:

| ID | Composition | Theme |
|----|-------------|-------|
| 5v-1 | Opus + GPT-5.2 + Gemini Pro + QwQ + DeepSeek-R1 | **1 per provider**: best from each, maximum architectural diversity |
| 5v-2 | Best Claude + Best Gemini + Best OpenAI + Best Qwen + Best DeepSeek | **1 per provider**: top from 1A rankings (may differ from 5v-1) |
| 5v-3 | Sonnet, Opus, Gemini, DeepSeek-R1, GPT-5 Mini  | **Best hand-picked diversity**: oracle model selection |
| 5v-4 | 5 voters with slight prompt variations (same model) | **Prompt diversity**: test if rephrasing the flow list changes answers |

**Data-driven ensembles**:

| ID | Composition | Selection method | Objective |
|----|-------------|------------------|-----------|
| 5v-5 | Top 5 from 1A | Highest individual accuracy | Accuracy |
| 5v-6 | Top 5 by accuracy, 1 per provider | Guaranteed provider diversity | Accuracy |
| 5v-7 | 5 models with most diverse predictions | Maximum complementary errors | Diversity |
| 5v-8 | (TBD after 1A) | | Diversity |

#### 10 Voters — 4 Combinations (2 hand-picked + 2 data-driven)

**Hand-picked ensembles**:

| ID | Composition | Theme |
|----|-------------|-------|
| 10v-1 | Maximize diversity across all axes | 2 from each provider (high + medium tiers) |
| 10v-2 | Sonnet × 10 @ temp=0.3 | **Pure self-consistency at max scale** |

**Data-driven ensembles**:

| ID | Composition | Selection method | Objective |
|----|-------------|------------------|-----------|
| 10v-3 | Top 10 from 1A | Highest individual accuracy | Accuracy |
| 10v-4 | 10 most diverse models | Maximum complementary errors | Diversity |

**Total**: 16 + 8 + 4 = **28 ensemble configs**.

### D.4 Temperature in Ensembles

Temperature is not a separate sweep — it's part of the ensemble design:

- **Self-consistency ensembles** (same model × N): temp=0.2 to induce variation between votes. If all voters are identical at temp=0, every vote is the same → useless ensemble.
- **Cross-model ensembles** (different models): each voter runs at temp=0.0 (deterministic). Diversity comes from model differences, not stochastic sampling.
- **Mixed-setting ensembles** (e.g., 3v-8): each voter has its own temp/thinking config. Diversity comes from the settings delta.

### D.5 Bootstrapping from 1A

**Key optimization**: Cross-model ensembles (all voters at temp=0, different models) don't need new API calls. Each voter's predictions already exist in 1A's per-utterance data. We compose ensembles computationally by combining predictions.

With 5 seeds per model in 1A:
- **Paired-seed**: Use the same seed index for all voters → 5 ensemble samples per config. Report mean of 3 sampled from the 5.

**What can't be bootstrapped**: Self-consistency ensembles (same model × N at temp>0.2) and mixed-setting ensembles with temp>0 voters. These need actual API calls because the stochastic variation at temp>0 is what creates voter diversity.

### D.6 Run Count

**Bootstrapped from 1A (0 API calls):**

| ID | Ensemble | Why free |
|----|----------|----------|
| 3v-4 | Opus + GPT-5.2 + Gemini Pro | Cross-model, all in 1A |
| 3v-5 | Opus + Sonnet + Haiku | Claude family, all in 1A |
| 3v-6 | Gemini Pro + Flash + Gemma 27B | Google family, all in 1A |
| 3v-7 | GPT-5.2 + mini + nano | OpenAI family, all in 1A |
| 3v-9 to 3v-16 | Data-driven (8 ensembles) | Cross-model from 1A rankings |
| 5v-1 | 1 per provider (5 providers) | Cross-model, all in 1A (incl. DeepSeek) |
| 5v-2 | Best per provider | Cross-model, all in 1A |
| 5v-5 to 5v-8 | Data-driven (4 ensembles) | Cross-model from 1A rankings |
| 10v-1 | 2 per provider | Cross-model, all in 1A |
| 10v-3, 10v-4 | Data-driven (2 ensembles) | Cross-model from 1A rankings |
| **Subtotal** | **21 ensembles** | **0 API calls** |

**Actual runs needed (self-consistency + mixed):**

| ID | Ensemble | Voters | New API calls per domain per seed |
|----|----------|:------:|----------------------------------:|
| 3v-1 | Opus × 3 @ temp=0.2 | 3 | 3 × 256 = 768 |
| 3v-2 | Sonnet × 3 @ temp=0.2 | 3 | 768 |
| 3v-3 | Gemini Flash × 3 @ temp=0.2 | 3 | 768 |
| 3v-8 | Opus mixed temps (0.0, 0.1, 0.2) | 2 new † | 512 |
| 5v-3 | Opus × 5 @ temp=0.2 | 5 | 1,280 |
| 5v-4 | 5 prompt variations | 4 new ‡ | 1,024 |
| 10v-2 | Opus × 10 @ temp=0.2 | 10 | 2,560 |
| **Subtotal** | **7 ensembles** | | **7,424 per domain per seed** |

† 3v-8 has 3 voters (Opus@temp=0.0, 0.1, 0.2). The temp=0.0 voter is in 1A; temp=0.1 and 0.2 need new runs.
‡ 5v-4 has 5 prompt variations of the same model. 1 uses the standard prompt (in 1A); 4 need new runs.

**Total 1B API calls (actual runs only):**

| Item | Count |
|------|------:|
| New calls per domain per seed | 7,424 |
| × 2 domains | × 2 |
| × 5 seeds | × 5 |
| **= Total API calls** | **74,240** |

**Savings vs. running all 28 ensembles from scratch**: 21 of 28 ensembles are free. The 7 that need runs are dominated by the self-consistency configs (especially 10v-2 at 10 voters).

### D.7 What We Measure

- **Top-1 accuracy** per ensemble per domain
- **Accuracy lift over best single voter**: Does ensemble > best from 1A?
- **Marginal value of additional voters**: 1 → 3 → 5 → 10 curve
- **Diversity axis comparison**: Which source of diversity helps most? (provider > family > temperature > thinking > prompt?)
- **Self-consistency vs. cross-model**: Same model × N vs. different models at same voter count
- **Cost-accuracy tradeoff**: Accuracy per dollar (bootstrapped ensembles have zero marginal inference cost!)
- **Confidence (agreement) distribution**: How often do ensembles fully agree vs. split?

### D.8 Key Plots for Blog

1. **Accuracy vs. voter count curve**: x = voters (1, 3, 5, 10), y = accuracy. One line per diversity strategy. Shows diminishing returns.
2. **Diversity axis bar chart**: For 3-voter ensembles, group by diversity theme — which axis drives the most accuracy gain?
3. **Scatter: accuracy vs. cost**: Each ensemble is a point. Pareto frontier highlighted. Single best model (from 1A) as reference line.
4. **Heatmap: ensemble × domain**: Do the best ensembles transfer across domains?
5. **Agreement distribution**: Histogram of voter agreement rates (3/3, 2/3, etc.) — how often do ensembles disagree?

### D.9 Output → Feeds into 1C

Rank all 1B ensembles on accuracy. Take the **top 10** for the calibration deep-dive, plus the **top 4 single-model baselines** from 1A as calibration reference points (14 total).

Single models get a natural confidence signal from their output: each detected flow is treated as a pseudo-voter, so `confidence = 1 / len(detected_flows)`. No verbalized confidence needed for any config type.

Several 1B slots already cover this conversion:
- 3v-1/3v-2/3v-3: pre-selected self-consistency ensembles
- 5v-3: 5-voter self-consistency
- 10v-2: 10-voter self-consistency
- Data-driven slots (3v-9 through 3v-16): should include "best 1A single → self-consistency" if the top 1A model isn't already represented in hand-picked self-consistency slots

---

## E. Experiment 1C — Confidence Calibration

### E.1 Purpose

The top 10 ensembles from 1B + top 4 single-model baselines from 1A (14 configs total). Now we test: **are their confidence scores well-calibrated?** A model can be accurate but overconfident (says 0.95 when it's right 0.75 of the time) or underconfident (says 0.50 when it's right 0.85 of the time).

Single models serve as calibration baselines — they reveal how much ensemble voting improves calibration over raw model output.

### E.2 Configs

- **Top 10 ensembles** selected by accuracy from 1B (cross-model, self-consistency, and mixed-temp)
- **Top 4 single models** from 1A baselines (calibration reference points)

### E.3 Confidence Derivation

| Config type | Confidence source |
|-------------|-------------------|
| Cross-model ensemble | Weighted agreement: `sum(weights for winning flow) / sum(all weights)` |
| Self-consistency ensemble | Agreement rate: `votes_for_winner / total_votes` |
| Mixed-setting ensemble | Weighted agreement (weights may vary by voter) |
| Single model | Decomposed pseudo-voters: each detected flow becomes a voter → confidence = `1 / len(detected_flows)` |

For ensembles, **consistency = confidence**. For single models, outputting multiple flows signals uncertainty — a model that says `[expand, rework]` is 50% confident, while `[expand]` is 100% confident. This gives single models a natural confidence signal rather than forcing them to 1.0.

### E.4 Run Count

**0 API calls.** 1C is purely an analysis phase.

All 1C configs are 1B ensembles. The per-voter prediction data already exists:
- **Cross-model ensembles** (bootstrapped from 1A): per-voter predictions stored in 1A
- **Self-consistency ensembles** (actual 1B runs): per-voter predictions stored in 1B

1C computes calibration metrics from this existing data. No re-runs needed — provided that 1A and 1B store per-voter, per-utterance predictions (not just the majority answer). This is a hard requirement for the data storage schema.

### E.5 Calibration Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Expected Calibration Error (ECE)** | Binned mean |accuracy − confidence| | ≤ 0.10 |
| **Maximum Calibration Error (MCE)** | Worst bin | ≤ 0.20 |
| **Brier Score** | Mean (confidence − correct)² | ≤ 0.15 |
| **Reliability Diagram** | Accuracy vs. confidence in 10 bins | Visual |
| **Overconfidence Rate** | % wrong predictions with confidence > 0.64 | ≤ 0.15 |
| **Underconfidence Rate** | % correct predictions with confidence < 0.64 | ≤ 0.20 |
| **Confidence-Accuracy Correlation** | Spearman rank | ≥ 0.50 |

### E.6 Key Plots for Blog

1. **Reliability diagrams**: One per config, overlaid. Perfect calibration = diagonal line.
2. **ECE bar chart**: All 14 configs side by side, colored by group.
3. **Accuracy vs. ECE scatter**: The Pareto frontier — which configs are both accurate AND calibrated?
4. **Overconfidence heatmap**: By category × config. Where do models over-promise?

### E.7 Output

The final recommended configuration(s):
- **Best overall**: Highest accuracy with ECE ≤ 0.10
- **Best budget**: Cheapest config with accuracy ≥ 0.80 and ECE ≤ 0.15
- **Domain-specific** (if warranted): Different recommendations per domain

---

## F. Experiment 2 — Tool-Calling vs. the Funnel

Experiment 2 is split into three sub-experiments:

- **Exp 2A** — Staged NLU Funnel: measures each stage of the funnel independently (intent classification, slot-filling, scoped tool selection)
- **Exp 2B** — Direct Tool-Calling: a single LLM call picks from the full tool manifest (~56 tools)
- **Exp 2C** — Direct Tool-Calling + Ambiguity Hint: same as 2B but with an explicit prompt paragraph instructing the model to prefer `handle_ambiguity` on unclear requests

Results live in `results/exp2a/` (with `intents/`, `slots/`, `tools/` subdirs), `results/exp2b/`, and `results/exp2c/`.

### F.1 Core Comparison

| | Exp 2A: NLU Funnel (staged) | Exp 2B: Direct Tool-Calling (flat) | Exp 2C: Tool-Calling + Hint |
|---|---|---|---|
| **Step 1** | Intent: choose 1 of 6 | — | — |
| **Step 2** | Flow: choose 1 of ~12 candidates | — | — |
| **Step 3** | Slots: extract parameters | — | — |
| **Step 4** | Tools: 5-7 per flow (scoped by flow) | Tools: choose from all ~56 directly | Tools: all ~56 + ambiguity hint |
| **LLM calls** | 1 (intent) + N (flow voters) + 1 (slots) + 1 (tool) | 1 | 1 |
| **Decision space** | 6 → 12 → 2 → 5-7 (staged) | ~56 (flat) | ~56 (flat, prompted) |

### F.2 Parameters

Uses the same model pool as Experiment 1A (no voting — single call):

| # | Parameter | Levels |
|---|-----------|--------|
| 1 | Model Type | Claude, Gemini, OpenAI, Qwen, DeepSeek |
| 2 | Model Level | Low, Medium, High |
| 3 | Thinking Effort | Low, High, Extended |

**Temperature**: Fixed at 0.0 (deterministic single call).

~21 valid configs × 2 domains × 3 seeds = **~126 runs**.

### F.3 Tool Definitions

All unique domain tools across user-facing flows (~20-25 per domain) presented as tool definitions in the API call. These are the **actual tools**, not flow-as-tool wrappers.

### F.4 Accuracy is Not Directly Comparable Across Experiments

Exp 1 measures **flow detection only** — a single classification task with ~12 candidates per intent. Exp 2 is **strictly harder** because it operates end-to-end: the model must select the correct tool from a larger candidate set (~56 tools in 2B) AND fill parameters correctly. Lower accuracy in Exp 2 vs Exp 1 does not mean the pipeline is worse — it means the task subsumes flow detection plus additional complexity.

The meaningful comparisons are:
- **Within Exp 2**: 2A funnel vs 2B flat (same E2E task, different approaches)
- **Exp 2A intent stage vs Exp 1A**: intent classification (6-way) is comparable to, but simpler than, flow detection (~12-way). The intent stage is a coarser version of the same problem.
- **Model ranking stability**: do the same models rank similarly across experiments?

### F.5 Confidence from Tool-Calling

Three approaches (tested for the top 5 Exp 2 configs only):
1. **Verbalized**: System prompt asks for `confidence` field
2. **Self-consistency**: N=3 parallel calls at temp=0.1, agreement rate
3. **Binary**: Tool called → 1.0, text response → 0.0

### F.6 Hypothesis Space

Comparisons are between 2A and 2B (both E2E), not with Exp 1 (flow-only):

- **H1**: 2B flat matches 2A funnel → staged narrowing is overhead
- **H2**: 2A funnel wins on accuracy, 2B wins on cost/latency → trade-off
- **H3**: 2A dominates both accuracy and calibration → staged narrowing adds value
- **H4**: 2B wins on distinct-tool flows, 2A wins where tools overlap across flows
- **H5**: Domain-dependent — funnel helps more in domains with confusable tools

### F.7 Experiment 2C — Ambiguity Hint Ablation

Flat tool-calling (Exp 2B) struggles on ambiguous turns. The scoped pipeline (2A) handles ambiguity upstream via the NLU funnel, but flat mode gives the model all ~56 tools and expects it to independently decide when to call `handle_ambiguity`. Exp 2C tests whether an explicit prompt hint about ambiguity handling can close that gap.

**Intervention**: Insert one paragraph after instruction #4 in the system prompt (the existing `handle_ambiguity` mention):

> **Important — Ambiguous Requests**: Users frequently make requests that are ambiguous — they could map to multiple distinct operations, or the intended action is unclear from context alone. Examples: "Help me with the intro", "Look at the sales data", "Fix this". In these cases, you MUST call `handle_ambiguity` rather than guessing which tool to use. When in doubt, prefer `handle_ambiguity` over committing to a specific tool.

**Implementation**: `build_tool_calling_prompt(domain, context, mode='hint')` — the `mode='hint'` flag conditionally inserts the paragraph. Runner method: `run_exp2c` in `helpers/harness.py`. CLI: `exp2_runner.py --mode hint`.

**Hypothesis**: The hint improves `ambiguous_first` and `ambiguous_second` categories (model correctly calls `handle_ambiguity`) but may hurt `same_flow` and `switch_flow` (over-triggering ambiguity on clear requests).

**Key metric**: Category-level accuracy deltas between Exp 2C and Exp 2B, especially `ambiguous_first` and `ambiguous_second`.

**Files**: `results/exp2c/` (JSONL + summary JSONs).

---

## G. Metrics Summary

### G.1 Tool-Calling Scoring: Strict Precision Gate

All Exp 2 results (2A scoped, 2B flat, 2C hint) use **strict scoring** for tool-calling accuracy. The precision gate penalises over-prediction because tool calls have side effects — calling `revise_content`, `format_content`, and `normalize_structure` when only `format_content` was needed means the user gets unwanted revisions alongside the intended formatting fix.

**Scoring rules:**

- **Recall floor**: `ceil(n/2)` for n > 1 gold tools, else n. (1 gold → hit 1, 2 gold → hit 1, 3 gold → hit 2.)
- **Precision gate**: Triggers at **3+ non-freebie** tool predictions. At that point, precision must be **≥ 0.50** (at least half of non-freebie calls must be gold).
- **Freebie exemption**: Read-only precursor tools (Hugo: `read_post`, `read_outline`; Dana: `describe_stats`, `list_datasets`) are excluded from the precision denominator. Calling `read_post` before `revise_content` is fine.
- **Correct** = recall passes AND (non-freebie predicted ≤ 2 OR precision ≥ 0.50).

**Special cases:**

- `handle_ambiguity` on an ambiguous turn (has `candidate_flows`) → always correct.
- Null call (no tools predicted) with gold tools → always incorrect.
- `conversational_response` is a regular tool — must appear in the gold set to count.

**History**: An earlier lenient scheme (gate at 4+, threshold 0.25) was used for some initial runs. All results were rescored to strict in March 2026. The impact was ~1–3pp on Hugo (where models tend to over-call analysis tools), ~0pp on Dana.

### G.2 Accuracy Metrics (Experiments 1A, 1B, 2)

| Metric | Definition | Target |
|--------|-----------|--------|
| **Top-1 Accuracy** | Correct flow / tool | ≥ 0.80 |
| **Top-3 Accuracy** | Expected in top 3 | ≥ 0.95 |
| **Per-Intent Accuracy** | By intent category | No intent < 0.70 |
| **Near-Miss Rate** | Error is in `confusable_with` | Track |
| **Confusion Matrix** | N × N (N = user-facing flow count per domain) | Visual |

### G.3 Calibration (Experiment 1C)

| Metric | Definition | Target |
|--------|-----------|--------|
| **ECE** | Binned mean |accuracy − confidence| | ≤ 0.10 |
| **MCE** | Worst bin | ≤ 0.20 |
| **Brier Score** | Mean (confidence − correct)² | ≤ 0.15 |
| **Reliability Diagram** | 10-bin plot | Visual |
| **Overconfidence Rate** | % wrong with conf > 0.64 | ≤ 0.15 |
| **Underconfidence Rate** | % correct with conf < 0.64 | ≤ 0.20 |

### G.4 Practical

| Metric | Definition |
|--------|-----------|
| **Latency (p50, p95)** | Wall-clock per utterance |
| **Cost per utterance** | Total tokens |
| **Failure rate** | % API errors/timeouts |

### G.5 Cross-Domain

| Metric | Definition |
|--------|-----------|
| **Config Rank Correlation (Kendall's τ)** | Same configs rank same across both domains? |
| **Domain × Parameter Interaction** | Parameter effects differ by domain? |
| **Universal Top-K** | Configs in top N for both domains |

### G.6 Exp 1 vs. Exp 2

| Metric | Definition |
|--------|-----------|
| **Accuracy Delta** | Best Exp1 − Best Exp2 |
| **ECE Delta** | Best Exp1C − Best Exp2 (calibrated) |
| **Latency Ratio** | Exp1 / Exp2 |
| **Cost Ratio** | Exp1 / Exp2 |
| **Tool Accuracy** (Exp2 only) | % correct tool selection (= flow accuracy, 1:1 mapping) |

---

## H. Evaluation Set Design

### H.1 Scope

**128 multi-turn conversations per domain** (2 domains = 256 conversations, 512 labels). All user-facing flows are eligible (excluding Plan and Internal):
- **Hugo**: 35 flows across 5 intents (Research 6, Draft 7, Revise 8, Publish 7, Converse 7)
- **Dana**: 37 flows across 5 intents (Clean 8, Transform 8, Analyze 7, Report 7, Converse 7)

Models see all ~35-37 user-facing flows in the prompt (full candidate set per domain). Conversations are organized into 4 categories of 32 each, testing different aspects of flow detection in multi-turn context.

### H.2 Conversation Structure

Each conversation is 3 turns: user → assistant → user. Turns 1 and 3 carry flow labels; turn 2 is a simulated assistant response. This tests whether models can maintain or shift flow detection across conversational context.

**128 conversations per domain = 32 per category × 4 categories:**

| Cat | Name | Turn 1 Label | Turn 3 Label | What it tests |
|-----|------|-------------|-------------|---------------|
| A | `same_flow` | flow_X | flow_X | Consistency — same flow in context |
| B | `switch_flow` | flow_X | flow_Y | Context shift — can model ignore prior flow? |
| C | `ambiguous_first` | `ambiguous` (candidate_flows) | flow_X (clarified) | Detecting genuine ambiguity |
| D | `ambiguous_second` | flow_X | Plan orchestrator + candidate_flows | Detecting multi-request utterances |

### H.3 Format

Files live at `experiments/eval/gen_{domain}.json`. Compact JSON format with turn metadata on one line, utterance on the next.

**Standard turn (categories A, B, and resolved turns in C/D):**
```json
{
  "convo_id": "hugo_001", "category": "same_flow",
  "scenario": "Technical ML/AI blog -- writing about transformer architectures",
  "turns": [
    {
      "turn_num": 1, "flow": "expand", "intent": "Draft", "speaker": "user",
      "utterance": "I've got bullet points for the 'noise pollution' section. Flesh them out into actual paragraphs."
    }, {
      "turn_num": 2, "speaker": "agent",
      "utterance": "I'll develop those bullet points into full prose while keeping your voice."
    }, {
      "turn_num": 3, "flow": "expand", "intent": "Draft", "speaker": "user",
      "utterance": "Same thing for the 'commuting' section, those notes need fleshing out too."
    }
  ]
}
```

**Category C — ambiguous first turn, clarified third turn:**
```json
{
  "turn_num": 1, "flow": "ambiguous", "candidate_flows": ["expand", "rework"],
  "candidate_intents": ["Draft", "Revise"], "speaker": "user",
  "utterance": "The overnight train section is pretty bare bones, just scattered notes..."
}
```

**Category D — clear first turn, multi-request third turn routed to Plan:**
```json
{
  "turn_num": 3, "flow": "outline", "intent": "Plan",
  "candidate_flows": ["format", "plot"], "candidate_intents": ["Clean", "Report"],
  "speaker": "user",
  "utterance": "Fix the date formats and then chart the revenue by month"
}
```

Category D routes to the domain's Plan orchestrator (`outline` for Dana, `blueprint` for Hugo) since the user is requesting multiple actions. The `candidate_flows` field preserves the individual flows for post-hoc analysis.

**Field conventions:**
- `convo_id`: `{domain}_{NNN}` (sequential, renumbered on save)
- `speaker`: `"user"` or `"agent"` (unified across all turns)
- `utterance`: unified field for both user and agent text
- `flow` / `intent`: only on user turns (the label)
- `candidate_flows` / `candidate_intents`: on ambiguous turns (C turn 1, D turn 3)
- Ordering within file: same_flow first, then switch_flow, ambiguous_first, ambiguous_second

### H.4 Generation Pipeline

Conversations are produced via synthetic data augmentation using `experiments/data_aug/` with:
- **Model assignment**: All categories use Opus 4.6 (`claude-opus-4-6`). Sonnet was tested and found insufficient for generating natural, non-formulaic conversations
- **DAX conditioning**: Each flow's DAX code is decomposed into dact primitives and included in the prompt for semantic grounding
- **Scenario banks**: 10 scenarios per domain, sampled round-robin
- **Diversity mechanisms**: temperature 0.8, persona hints (~30%), anti-keyword rules
- **Label attachment**: Labels come from the sampling step (not the LLM), preventing label drift
- **Quality validation**: 8 automated checks (encoding, format, flow-name leakage, length, uniqueness, ambiguity sanity, multi-request sanity, keyword-match avoidance)
- **Post-processing**: Hand-review and rewrite of same_flow conversations; LLM-generated conversations trimmed and sanitized

### H.5 Shared Across All Phases

The same 128 conversations per domain (256 total, 512 labels across Hugo + Dana) are used for 1A, 1B, 1C, and Experiment 2. No data leakage between phases because there's no training — these are evaluation-only.

**Prompt constancy**: The system prompt (flow list, instructions, candidate set) is identical across all providers within a domain. No per-provider formatting. This ensures a fair comparison — any accuracy difference reflects model capability, not prompt engineering.

### H.6 Seeds: Collect 5, Report 3

Every configuration is run **5 times** with different random seeds. For analysis and reporting, each trial **samples 3 of the 5 seeds** and reports the mean. This gives us more data than we strictly need, creating robustness against any single bad seed while keeping the paper narrative simple.

- **1A**: 5 seeds collected per config. Per-utterance predictions stored for all 5 → used to bootstrap cross-model ensembles in 1B. Analysis samples 3.
- **1B cross-model** (bootstrapped): Composed from 1A predictions. With 5 seeds per voter, paired-seed gives 5 ensemble samples; analysis uses 3.
- **1B self-consistency** (actual runs): 5 seeds collected per config. Analysis samples 3.
- **1C**: Pure analysis of 1B data. Uses the same 3-seed samples as 1B.
- **Exp 2**: 3 seeds per config (no bootstrapping benefit from extra seeds).

Paper sentence: "Each configuration was evaluated with 5 random seeds; we report the mean of 3 randomly selected seeds per trial."

### H.8 Scoring Rules

All scores are boolean (correct / incorrect). No partial credit.

#### Clear turns (no `candidate_flows`)

Exactly one detected flow matching the gold label: correct. Any other output (wrong flow, multiple flows, no flows): incorrect.

#### Ambiguous turns (`candidate_flows` present)

Two criteria, both required:

1. **Recognise ambiguity**: the model/ensemble must signal that the turn is ambiguous.
   - *Single model*: must output 2+ flows. A single-flow answer fails immediately, even if that flow is in the candidate set.
   - *Ensemble*: ambiguity is recognised when any voter outputs 2+ flows, OR when voters disagree (no single answer-set holds a majority). A unanimous single-flow answer fails.

2. **Predicted label in candidate set**: pick one flow, check if it's in the gold candidate set.
   - *Single model*: pick one detected flow at random.
   - *Ensemble*: flatten all voters' outputs into individual flow mentions, count each flow, pick the highest-count flow. Ties broken by coin flip.

**Single-model examples** (`candidate_flows = [expand, rework]`):

| Detected flows | Criterion 1 | Outcome | Why |
|----------------|:-----------:|:-------:|-----|
| `[expand, rework]` | pass | always correct | random pick is always in candidates |
| `[expand, polish]` | pass | coin flip | 1/2 expand (in set), 1/2 polish (not) |
| `[expand]` | **fail** | incorrect | Only 1 flow — ambiguity not recognised |
| `[tone, polish, rework]` | pass | 1/3 chance | only rework is in candidates |

**Ensemble examples** (`candidate_flows = [expand, rework]`, 3 voters):

| Votes | Flatten | Criterion 1 | Predicted | Outcome |
|-------|---------|:-----------:|-----------|:-------:|
| `[expand,rework]`, `[expand,polish]`, `[expand]` | expand=3, rework=1, polish=1 | pass (voters 1,2 output 2+) | expand | correct |
| `[polish]`, `[tone]`, `[rework]` | polish=1, tone=1, rework=1 | pass (no majority) | coin flip | 1/3 chance correct |
| `[expand]`, `[expand]`, `[expand]` | expand=3 | **fail** (unanimous) | — | incorrect |
| `[expand]`, `[polish]`, `[rework]` | expand=1, polish=1, rework=1 | pass (no majority) | coin flip | 2/3 chance correct |

---

## I. Conversation Quality Guidelines

These patterns emerged from iterative hand-review of generated conversations and apply to both hand-written and LLM-generated data.

### I.1 Core Principle: Turn Dependency

**If the user turns were reversed, the labels would change or otherwise be indecipherable.** Turns must be contextually dependent on each other, not two independent requests that happen to use the same flow.

### I.2 Naturalness

- **Imagine the user is typing on their smartphone.** This is the single best heuristic for natural utterances. Mobile users are terse, rely on shared context, skip pleasantries, and don't repeat what's already on screen. If an utterance feels too long or formal for a phone keyboard, it's wrong.
- **Real utterances are short.** Most are 10-40 words, many under 15 words. LLM-generated utterances are almost always too long.
- **Users observe, they don't command.** Real users describe what they see or what's wrong, not what the agent should do. "The ICD codes aren't matching" beats "Flag anything that doesn't match." "I think I see repeated employees?" beats "Check for duplicate employee IDs and clean those out." The agent figures out the action; the user states the problem.
- **Drop imperatives.** State the problem without telling the agent what to do. "Signup dates are all over the place." Period. Don't append "Standardize them." The agent infers the action from context.
- **Turn 3 is extremely terse.** Context from turns 1-2 makes elaboration unnecessary. Turn 3 averages 2-9 words. "Which sites?" (2 words), "What's the percent change?" (5 words), "Show me likes vs shares by platform." (7 words). If turn 3 restates context from earlier turns, it's too long.
- **Turn 3 can include follow-up questions.** "What's the percent change?" or "Which sites?" are natural follow-ups that still map to a flow. Not every turn 3 has to be a command.
- **Use domain abbreviations.** Real users say "MoM" not "month-over-month", "CTR" not "click-through rate", "P&L" not "profit and loss." Abbreviations signal domain fluency and shorten utterances naturally.
- **Agent explains WHY.** Good agent responses include reasoning: "since that is the most common format I see in the data" or "Biggest jump we've seen this quarter." This gives the user something to react to and makes turn 3 flow naturally.
- **Don't directly give away the answer.** Even unambiguous turns shouldn't spell out the operation in technical terms. Wrong: "Can you expand and develop my bullet points into fully written paragraphs?" Right: "Flesh them out into actual paragraphs."
- **Turn 3 should depend on turns 1-2.** Use anaphora ("Same thing for...", "That one too", "What about X instead?") and implicit context ("Too smooth now, try 3 days instead" only makes sense after the agent proposed a window size).
- **More implicit, less explicit.** Users assume shared context and don't re-explain the domain. "Too smooth now, try 3 days instead" beats "Can you also perform the same rolling average smoothing operation on the revenue column using a 3-day window?"
- **Minimize dashes and fancy punctuation.** Em dashes are an LLM tell. Use commas, periods, colons, or restructure the sentence. Agent utterances should be equally clean.

### I.3 Same-Flow Patterns (Category A)

Three valid patterns for why a user invokes the same flow twice in a conversation:

**(a) Slot-value missing, agent clarifies:**
Turn 1 omits a required parameter. Agent asks. Turn 3 provides it.
```
User: "Can you go over it to see what else might be missing before scheduling?"
Agent: "Sure, which post were you referring to?"
User: "The one about butternut squash soup."
```

**(b) User builds on previous request:**
Turn 1 requests flow on entity X. Turn 3 requests same flow on entity Y, referencing prior action.
```
User: "The intro reads clunky, tighten it up."
Agent: "I'll smooth out the transitions and word choice..."
User: "Same thing for the benchmarks section, it's a bit wordy."
```

**(c) User responds to agent building on previous:**
Turn 1 makes a request. Agent proposes specifics (threshold, names, window). Turn 3 adjusts/corrects the proposal.
```
User: "The weekly spend numbers are pretty noisy. Smooth them out with a rolling average."
Agent: "I've computed a weekly rolling average on Annual_Spend (USD). How does that look?"
User: "Too smooth now, try 3 days instead."
```
```
User: "Username has values like 'Instagram_fashionista23' smooshed together. Break that apart."
Agent: "I'll separate at the underscore. The two new columns will be platform and username, does that work?"
User: "I want platform and account_name."
```

**Anti-pattern: two independent same-flow requests.** "Rename column A" then "Rename column B" are just two unrelated requests that coincidentally use the same flow. The turns should have a narrative thread.

**Prefer naturally complex flows for same-flow.** Some flows are inherently multi-step — they require clarification, parameter negotiation, or iterative refinement. These are ideal for same-flow conversations because the back-and-forth is genuine, not contrived. When sampling for LLM generation, weight these flows higher.

*Hugo naturally complex flows:* refine, rework, format, create, schedule, compare, inspect
*Dana naturally complex flows:* join, fill, validate, split, pivot, dashboard, define

**Anti-pattern: Converse same-flow.** Converse flows (preference, chat, etc.) are almost always contrived as same-flow conversations because they don't change data. Two back-to-back preference settings or two chat questions feel like independent requests stapled together, violating the dependency principle.

### I.4 Switch-Flow Patterns (Category B)

The switch should feel organic, not forced. Good patterns:
- brainstorm -> endorse: "How should I start?" / agent suggests / "Yea, let's go with that."
- tone -> preference: "Make it storytelling vibe" / agent adjusts / "Can you remember that going forward?"
- explain -> browse: "Wait, what'd you just do?" / agent explains / "Cool, so what else have I got sitting around?"
- replace -> reject: "Expand APAC to Asia-Pacific everwhere" / agent does region column only / "Everywhere means everywhere, not just in the region column."

The second flow should emerge naturally from the conversation.

**Converse label placement:** Minimize Converse flows overall since they don't change data and are less interesting as test cases. When Converse labels do appear, they work best as **turn 3** (natural follow-ups like undo, endorse, reject, preference) rather than turn 1. Two Converse labels in a single conversation is a yellow flag.

**Avoid near-synonyms in switch_flow.** Don't pair flows that are functionally similar (e.g., confirm and endorse). The switch should test real discrimination, not distinguish near-identical operations.

### I.5 Ambiguity Levels and Category Mapping

The four ambiguity levels from the [Ambiguity Handler spec](../components/ambiguity_handler.md) map directly to the conversation categories. This mapping determines how the agent should respond in turn 2.

| Level | What's unknown | Category | Agent turn 2 behavior |
|---|---|---|---|
| **Specific** | Slot value missing | same_flow (A) | Ask for the missing parameter |
| **Partial** | Two flows confusable, can't resolve from data | ambiguous_first (C) | Ask a clarifying question to distinguish |
| **Confirmation** | Two flows confusable, but context clues exist | ambiguous_first (C) | Inspect data, propose an action, ask user to confirm |
| **General** | Intent unclear, possibly multi-flow or truly vague | ambiguous_second (D) | Route to Plan orchestrator, or ask broad clarifying question |

Category C (ambiguous_first) contains both **Partial** and **Confirmation** sub-types. The difference is whether the agent can gather evidence from the data to propose an answer:

- **Partial**: "Do I have anything on fermentation?" — the agent can't tell if the user wants to browse topics or search posts without asking. The ambiguity is about intent/preference, not data.
- **Confirmation**: "There are gaps in the salary data." — the agent can check correlations and propose interpolation from related columns. The ambiguity can be partially resolved by inspecting context.

The pilot set targets a 50/50 split between Partial and Confirmation within Category C. Each ambiguous_first conversation carries a `rationale` field on turn 1 explaining why the utterance is ambiguous.

### I.6 Multi-Request Patterns (Category D)

Category D turn 3 contains two operations that route to the Plan orchestrator. The two requests must read as a single natural utterance, not two commands stapled together with "and".

**Anti-pattern: bare "and" between unrelated intents.** "Fix the dates and chart revenue by quarter" or "Rename the column and profile the usage data" feel mechanical. Real users don't issue two unrelated commands connected by "and".

**Natural multi-request connectors:**

| Pattern | Example | Why it works |
|---|---|---|
| **Causal** ("so", "so I can") | "Fill the nulls with the median so I can see session duration over time." | One operation enables the other |
| **Prerequisite** ("before") | "I need clean dates before we set up the weekly readmission report." | Temporal dependency |
| **Sequential** ("and then") | "Show me what changed since last week, and then start a fresh post." | Ordered steps |
| **Embedded qualifier** | "Chart DAUs per quarter for all valid subscription types, meaning the active ones." | Second operation baked into the first as a constraint |
| **Implicit** | "Rewrite the code examples for Express 5 so we can send it off on Friday." | Scheduling implied by "send it off on Friday" — never names the operation |
| **Two sentences** | "Also flesh out the Laos section. Pull the Cambodia post down, photos are wrong." | Period separates; no conjunction needed |
| **Question + request** | "Also rename 'cust_id' to 'customer_id'. What do the usage columns look like?" | Second request is a question, not a command |

The key test: would a real person type this on their phone? If the turn 3 reads like two bullet points in a to-do list, rewrite it.

Each ambiguous_second conversation carries a `rationale` field on turn 3 explaining why the multi-request maps to the Plan orchestrator and what the two candidate flows are.

**Additional lessons (from pilot review):**

- **Same-intent candidate_flows are valid.** Two operations from the same intent still route to Plan if they require different flows (e.g., `replace` + `update`, both Clean). Multi-request does not require cross-intent operations.
- **Observation-implies-action pattern.** One request can be implicit: the user observes a problem ("I think there are two Chris Moore's in there") and the action (deduplication) is implied, not commanded. Pair with a question ("How many rows are there now?") for a natural multi-request that doesn't use any conjunction.

### I.7 LLM Generation Anti-Patterns

Issues to fix in post-processing when conversations are LLM-generated:

- **Em dashes are LLM tells.** Replace `--` with varied punctuation (~35% comma, 20% period, 15% semicolon, 15% single dash, 15% keep). Capitalize after period replacements.
- **Unicode special characters.** Smart quotes, em dashes (U+2014), en dashes (U+2013), ellipsis (U+2026), non-breaking spaces (U+00A0) must all be replaced with ASCII equivalents.
- **Verbosity.** LLM utterances routinely exceed 100 words; real ones are 10-40. Trim filler and restated context, keep the core request.
- **Over-explanation.** LLMs have the user explain exactly what they want in precise detail. Real users are vague, assume shared context, and use shorthand.
- **Bare "and" between unrelated intents.** LLMs join two operations with "X and Y" mechanically. Real users use causal ("so I can"), prerequisite ("before"), or implicit connectors. See I.6 for the full pattern list.

### I.8 Validator Thresholds

| Check | Threshold |
|-------|-----------|
| User min length | >1 character |
| User max words | 100 |
| Agent min length | >1 character |
| Agent max words | 80 |
| Uniqueness (Jaccard) | 0.6 |
| Flow name leakage | word-boundary match, skip common English words |
| Multi-request (Cat D) | Must contain: and, then, also, plus, so, before, too, meaning, two sentences, or semicolon |

---

## J. Run Budget Summary

### J.1 API Calls by Phase

| Phase | What runs | Configs | × Domains | × Seeds | Labels/run | API calls |
|-------|-----------|--------:|:---------:|:-------:|:----------:|----------:|
| **1A** (core) | Claude, Gemini, OpenAI sweep | 18 | 2 | 5 | 256 | **46,080** |
| **1A** (Qwen) | Qwen model levels | 3 | 2 | 5 | 256 | **7,680** |
| **1A** (DeepSeek pool) | DeepSeek voter predictions | 3 | 2 | 5 | 256 | **7,680** |
| **1A total** | | **24** | | | | **61,440** |
| **1B** (bootstrapped) | 21 cross-model ensembles | 21 | — | — | — | **0** |
| **1B** (self-consistency + mixed) | 7 ensembles | 7 | 2 | 5 | 256 | **74,240** |
| **1B total** | | **28** | | | | **74,240** |
| **1C** | Analysis only | 14 | — | — | — | **0** |
| **Exp 1 total** | | | | | | **135,680** |
| **Exp 2** | Tool-calling sweep | ~21 | 2 | 3 | 256 | **~32,300** |
| **Exp 2 (calibration)** | Top 5 × 3 methods | ~15 | 2 | 3 | 256 | **~23,000** |
| **Exp 2 total** | | | | | | **~55,300** |
| | | | | | **Grand total** | **~191,000** |

Exp 1 uses 5 seeds (for bootstrapping benefit); Exp 2 uses 3 seeds (no bootstrapping). All results reported as mean of 3 sampled seeds.

Note: The 4× increase over the previous 64-utterance design (256 labels per run vs 64) yields richer multi-turn evaluation data while keeping per-call cost identical. At ~500 tokens per call, that's ~96M tokens.

**Pipeline**: 1A collects all predictions (only phase that costs money for cross-model) → 1B bootstraps cross-model ensembles for free, runs self-consistency ensembles at temp>0.2 → 1C is pure analysis, zero API calls.

---

## K. Storage and Tracking

### K.1 Directory Structure

```
experiments/
├── configs/
│   ├── exp1a_configs.json           # 24 single-model configs (18 core + 3 Qwen + 3 DeepSeek)
│   ├── exp1b_ensembles.json         # 28 ensemble compositions (hand-picked + data-driven per tier)
│   ├── (exp1c_top8.json removed)    # Selection now auto-computed in build_report_1c.py
│   └── exp2_configs.json            # ~20 single-model configs (same as 1A)
├── eval/
│   ├── eval_hugo.json               # 128 multi-turn conversations (256 labels)
│   └── eval_dana.json               # 128 multi-turn conversations (256 labels)
├── tools/
│   ├── tool_manifest_hugo.json      # Flow-specific tool bindings
│   └── tool_manifest_dana.json
├── results/
│   ├── exp1a/                       # Per-domain JSONL files
│   ├── exp1b/
│   ├── exp1c/
│   ├── exp2a/                      # Staged NLU funnel (3 modes)
│   │   ├── intents/                # Intent classification
│   │   ├── slots/                  # Slot-filling (given gold flow)
│   │   └── tools/                  # Scoped tool selection (given gold intent+flow+slots)
│   ├── exp2b/                      # Direct tool-calling (flat, all ~56 tools)
│   └── combined.jsonl               # All phases merged
├── analyze.ipynb
└── blog/
    ├── charts/
    └── tables/
```

### K.2 Result Schema

```json
{
  "run_id": "exp1a_hugo_003_seed1",
  "experiment": "1A",
  "domain": "hugo",
  "seed": 1,
  "timestamp": "2026-02-18T14:30:00Z",
  "config": {
    "model_type": "claude",
    "model_level": "high",
    "model_id": "claude-opus-4-6",
    "temperature": 0.0,
    "voter_count": 1,
    "voters": null,
    "confidence_method": null
  },
  "summary": {
    "accuracy_top1": 0.89,
    "accuracy_top3": 0.97,
    "near_miss_rate": 0.70,
    "latency_p50_ms": 3200,
    "latency_p95_ms": 5800,
    "cost_tokens_total": 48000,
    "failure_rate": 0.00
  },
  "per_utterance": [
    {
      "eval_id": "hugo_e001",
      "expected_flow": "create",
      "detected_flow": "create",
      "correct": true,
      "near_miss": false,
      "confidence": null,
      "latency_ms": 3100
    }
  ]
}
```

**Critical for bootstrapping**: 1A results MUST store the raw `detected_flow` per utterance per seed. This is the data that 1B cross-model ensembles bootstrap from. Without it, we'd need to re-run 1A.

For 1B (self-consistency runs), `config.voters` is a list of voter specs and each utterance stores all individual voter predictions (not just the majority). For 1C, `confidence` is computed from voter agreement. For Exp 2A, each mode adds stage-specific fields (`detected_intent`, `detected_slots`, or scoped `tool_called`). For Exp 2B, `tool_called`, `tool_args`, `detected_flows`, `unmapped`, `used_wildcard`, and `ambiguity_flagged` are added.

---

## L. Execution Plan

### Stage 0: Prerequisites ✓
- [x] Build Dana (data analysis assistant) — ontology, flow catalog, all user-facing flows
- [x] Build Hugo (blogging assistant) — ontology, flow catalog, all user-facing flows
- [x] Set up API keys for Anthropic, Google, OpenAI, Qwen (via Together.AI), DeepSeek
- [x] Design flow-specific tool bindings for ALL flows per domain → `tool_manifest_{domain}.json` (Hugo 42K, Dana 39K)
- [x] Build eval sets: 128 multi-turn conversations × 2 domains → `eval_{domain}.json` (via `experiments/data_aug/`)

### Stage 1: Infrastructure ✓
- [x] Build experiment runner (supports 1A single-model, 1B ensemble, 1C calibration, Exp 2 tool-calling)
- [x] Implement accuracy metrics + confusion matrix
- [x] Implement calibration metrics (ECE, Brier, reliability diagram)
- [x] Enumerate configs → `exp1a_configs.json` (24 configs), `exp2_configs.json` (24 configs)
- [x] Seed management: runner collects all seeds; 3-of-5 sampling deferred to analysis phase
- [x] Smoke-tested all 5 providers end-to-end

### Stage 2: Experiment 1A — Data Collection
- [x] Run 15 configs × 2 domains × 5 seeds = 150 runs — 14/15 models complete (1280/1280 convos), Gemini 3.1 Pro pending quota (seed 1 only)
- [ ] Backfill Gemini 3.1 Pro (1a_006) rewritten conversations + remaining seeds 2-5
- [x] **Critical**: Store per-utterance, per-model predictions (not just accuracy summaries). This data feeds 1B bootstrapping.
- [x] Relabel or re-write dataset based on review of 1A results (`experiments/eval/mislabeled.md`)
- [x] Rescore relabeled entries across all JSONL files (`rescore_exp1a.py`)
- [x] Re-run inference on 20 rewritten conversations across all configs/seeds (`prep_rewrites.py`, `run_rewrites.py`)
- [x] Build HTML report with ranking table, confusion analysis, domain/family comparison charts (`results/build_report.py`)
- [x] Analyze: rank all 15 models by accuracy, per-category breakdown, cross-domain stability, provider line-chart

### Stage 3: Experiment 1B — Voting Ensembles
- [x] Compose 28 ensembles from 1A voter pool → `exp1b_ensembles.json` (resolved to `exp1b_ensembles_resolved.json`)
- [x] **Bootstrap** 21 cross-model ensembles computationally from 1A prediction data (0 API calls)
- [x] **Run** 7 self-consistency + mixed ensembles × 2 domains × 5 seeds
- [x] Analyze: accuracy lift, diversity axis comparison, voter count curve, cost-accuracy tradeoff (`build_report_1b.py`)
- [x] Rank all ensembles → select top 10 + top 4 single-model baselines (auto-computed in `build_report_1c.py`)

### Stage 4: Experiment 1C — Calibration (Analysis Only)
- [x] Compute calibration metrics from existing 1A/1B data (0 API calls)
- [x] Analyze: ECE, Brier, reliability diagrams, overconfidence rate (14 configs: 10 ensembles + 4 baselines) (`build_report_1c.py`)
- [x] Output: final recommended config(s) for production (deployment recommendations in 1C report findings)
- [x] Build HTML reports and study results for 1A, 1B, 1C

### Stage 5: Experiment 2A — Staged NLU Funnel
- [x] Run intent classification: 15 configs × 2 domains × 3 seeds = 90 runs (`exp2a/intents/`)
- [x] Run scoped tool selection: 15 configs × 2 domains × 3 seeds = 90 runs (`exp2a/tools/`)
- [x] Run flat tool-calling (all ~56 tools): 15 configs × 2 domains × 3 seeds = 90 runs (`exp2b/`)
- [x] Build Exp 2 report (`results/build_report_2.py`)
- [X] Compare 2A funnel vs 2B flat (same E2E task, different approaches)

### Stage 6: Make Results more Robust
- [ ] Run slot-filling: 15 configs × 2 domains × 3 seeds = 90 runs (`exp2a/slots/`), compare to selecting tool parameters
- [X] Ablation — Exp 2C ambiguity hint: 3 configs × 2 domains × seed 1 = 6 runs (`exp2c/`), see §F.7
- [ ] Select public benchmark (e.g., TauBench, MultiWOZ, or similar)
- [ ] Replicate key experiments on external data
- [ ] Train a model which does ambiguity handling natively

### Stage 7: Write-up and Submission
- [ ] Finalize narrative arc (1A → 1B → 1C → 2A → 2B)
- [ ] Draft paper (internal domains: Hugo + Dana)
- [ ] Combine internal + external results
- [ ] Publication-quality charts and tables
- [ ] Final paper draft
- [ ] Submit to Arxiv

---

## M. Resolved Decisions

All originally open questions have been resolved:

| # | Question | Decision |
|---|----------|----------|
| 1 | Dana readiness | Build Dana fully — the scaffolding was originally for data analysis. Next priority. |
| 2 | Flow scope | All user-facing flows evaluated (35 Hugo, 37 Dana). The original top-16 selection was superseded by comprehensive multi-turn eval coverage. |
| 3 | Tool design scope | Design tools for ALL flows (42+ per domain). Descriptions only — tools don't need to work. |
| 4 | External dataset | Need a dataset with tool-calls. Target a popular SWE or agent benchmark. |
| 5 | Thinking effort | Tested; negligible gains at 2× cost. Dropped from all configs. |
| 6 | 1B data-driven selection | Half maximize accuracy, half maximize diversity. No cost-efficiency objective (not a research-level question). |
| 7 | 1C expansion | Top 10 ensembles + top 4 single-model baselines (14 total). |
| 8 | Seeds | Collect 5 seeds per config; report mean of 3 sampled seeds per trial. Extra seeds provide robustness; 3-seed reporting keeps the paper simple. |
| 9 | Shared-tool disambiguation | No disambiguation needed. Correct tool = correct flow (1:1 mapping by design). |
| 10 | Prompt constancy | Identical prompts across providers within a domain. Fair comparison. |