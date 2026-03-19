# Synthetic Data Optimization — Lab Notebook

## Iteration 1 — 2026-03-18 — both domains

**Target**: Utterance length (KS 0.51), vocabulary overlap (Jaccard 0.18-0.20), terse rate (3.4% vs 14% eval)
**Change**:
- `generate_conversations.py` system prompt: Added explicit length constraints (T1: 8-18w, T3: 4-12w), style examples from eval, terse follow-up guidance per category, anti-verbosity rules
- `generate_scenarios.py`: Reduced example_utterances range from 10-30w to 5-18w, required at least one < 8w
**Result** (40 scenarios/domain, ~20 after dedup):
- T1 avg: 25w -> 13w (both domains)
- T3 avg: 25w -> 9.5w (both domains)
- T3 terse rate: 3.4% -> 45-48%
- Length KS: 0.51 -> 0.34 (improved, still red, threshold 0.3)
- Intrinsic: Flow entropy 0.95 GREEN, Tool entropy 0.95 GREEN (Hugo), 0.84 YELLOW (Dana)
- Comparative: Still mostly RED but sample too small (20-21 convos vs 128 eval) to be meaningful
**Issue**: 30% dedup rate — scenario diversity too low. Biggest cluster: 7 "small business blogger" scenarios deduped to 1.
**Verdict**: Utterance length/style fix WORKED massively. Scenario diversity is the bottleneck now.

## Iteration 2 — 2026-03-18 — both domains

**Target**: Scenario diversity (reduce 30% dedup rate), scale up volume
**Change**:
- Expanded DIVERSITY_AXES from 8 broad to 16 specific axes (e.g., "enterprise IT" -> "enterprise IT and cloud infrastructure teams", added K-12 teachers, healthcare, legal, real estate, etc.)
- Added rule 6 to scenario generation: "Each scenario MUST be about a genuinely different task/goal, not just different topics"
- Made dedup less aggressive: added more NON-duplicate examples showing that same action + different topic is NOT a duplicate
- Scaled up to 200 scenarios per domain (80% enrichment success → ~165 enriched → ~60-110 after dedup)
- Calibrated T3 length distribution: initial overcorrection made T3 too short (9w), then too long (16w), finally settled on "average 12-14w with ~15% terse"
**Result** (Hugo 65 convos, Dana 112 convos):
- Length KS: 0.51 -> 0.14 (Hugo YELLOW!), 0.17 (Dana YELLOW!) — was the biggest RED signal, now resolved
- Vocab Jaccard: 0.18 -> 0.26 (improving, still RED, threshold 0.30)
- Intrinsic: Flow entropy GREEN both, Tool entropy GREEN (Hugo 0.92), YELLOW (Dana 0.77)
- Intent JSD: YELLOW both (0.10 Hugo, 0.13 Dana)
- Tool coverage: 0.80 YELLOW (Hugo), 0.78 RED (Dana)
- Flow pair coverage: 0.25 (Hugo), 0.38 (Dana) — still RED
- Hugo dedup rate still 60% — the blogging domain has inherently similar scenarios
**Verdict**: Length distribution FIXED. Vocab and coverage improving with volume. Hugo needs more scenarios to overcome aggressive dedup.

## Iteration 3 — 2026-03-19 — both domains

**Target**: Vocab overlap (Jaccard 0.26→0.30+), flow/tool coverage, Dana tool entropy
