"""Semantic deduplication of enriched scenarios using LLM-based analysis.

Step 3 of the synthetic data pipeline: find semantically duplicate scenarios
(same domain/topic AND same user goal, even with different wording) that
Jaccard dedup in Step 1 missed, remove them, then backfill via Steps 1+2
to restore the original count.

Two-phase approach:
  Phase 1 — Within-batch: split scenarios into batches of ~60, each sent to
            an LLM to identify duplicate clusters.
  Phase 2 — Cross-batch consolidation: send ALL remaining scenarios in a
            single call to catch duplicates that spanned different Phase 1
            batches.

Usage:
    python datasets/data_aug_pranav/dedup_scenarios.py \
        --domain hugo --batch-size 60 --seed 43

    # Dedup only, no regeneration:
    python datasets/data_aug_pranav/dedup_scenarios.py \
        --domain hugo --skip-backfill

    # Dry run (print prompts, no API calls):
    python datasets/data_aug_pranav/dedup_scenarios.py \
        --domain hugo --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

# ── Path setup ───────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SCRIPT_DIR = Path(__file__).resolve().parent

# ── Logging ──────────────────────────────────────────────────────────

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    {
        'name': 'anthropic',
        'provider': 'anthropic',
        'model_id': 'claude-sonnet-4-6',
    },
    {
        'name': 'openai',
        'provider': 'openai',
        'model_id': 'gpt-5.2',
    },
    {
        'name': 'deepseek',
        'provider': 'openrouter',
        'model_id': 'deepseek/deepseek-chat',
    },
]

DEDUP_TEMPERATURE = 0.2

MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0


# ── Async provider call functions ────────────────────────────────────

async def _call_anthropic_async(system_prompt: str, user_prompt: str) -> str:
    """Call Anthropic API using async client."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    try:
        resp = await client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=4096,
            temperature=DEDUP_TEMPERATURE,
            system=system_prompt,
            messages=[{'role': 'user', 'content': user_prompt}],
        )
        return resp.content[0].text if resp.content else ''
    finally:
        await client.close()


async def _call_openai_async(system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI API using async client."""
    import openai

    client = openai.AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
    try:
        resp = await client.chat.completions.create(
            model='gpt-5.2',
            max_completion_tokens=4096,
            temperature=DEDUP_TEMPERATURE,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        )
        return resp.choices[0].message.content or ''
    finally:
        await client.close()


async def _call_openrouter_async(system_prompt: str, user_prompt: str, model_id: str) -> str:
    """Call OpenRouter's OpenAI-compatible API using async client."""
    import openai

    client = openai.AsyncOpenAI(
        api_key=os.environ['OPEN_ROUTER_API_KEY'],
        base_url='https://openrouter.ai/api/v1',
    )
    try:
        resp = await client.chat.completions.create(
            model=model_id,
            max_completion_tokens=4096,
            temperature=DEDUP_TEMPERATURE,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        )
        if not resp or not resp.choices:
            raise RuntimeError('OpenRouter returned empty choices (transient error)')
        return resp.choices[0].message.content or ''
    finally:
        await client.close()


async def _call_model_async(config: dict, system_prompt: str, user_prompt: str) -> str:
    """Async dispatch to the right provider with retries."""
    provider = config['provider']
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            if provider == 'anthropic':
                return await _call_anthropic_async(system_prompt, user_prompt)
            elif provider == 'openai':
                return await _call_openai_async(system_prompt, user_prompt)
            elif provider == 'openrouter':
                return await _call_openrouter_async(system_prompt, user_prompt, config['model_id'])
            else:
                raise ValueError(f'Unknown provider: {provider}')
        except Exception as e:
            last_error = e
            err_str = f'{type(e).__name__}: {e}'.lower()
            is_retryable = any(kw in err_str for kw in (
                'ratelimit', 'rate_limit', 'rate limit', 'resource_exhausted',
                'timeout', 'internal', 'server', '429', '500', '503',
            ))
            if not is_retryable:
                raise
            if attempt < MAX_RETRIES:
                delay = max(BACKOFF_BASE_S * (2 ** attempt), 5.0 if '429' in err_str else 1.0)
                log.warning(
                    'Retry %d/%d for %s after %s: %s',
                    attempt + 1, MAX_RETRIES, config['name'], type(e).__name__, e,
                )
                await asyncio.sleep(delay)

    raise last_error  # type: ignore[misc]


async def _run_wave(batch_specs: list[dict], system_prompt: str, semaphore: asyncio.Semaphore) -> list:
    """Run a wave of batches concurrently, bounded by semaphore."""
    async def _run_one(spec):
        async with semaphore:
            return await _call_model_async(spec['model_config'], system_prompt, spec['user_prompt'])
    return await asyncio.gather(*[_run_one(s) for s in batch_specs], return_exceptions=True)


# ── Prompt builders ──────────────────────────────────────────────────

def _build_dedup_system_prompt() -> str:
    """Build the system prompt for deduplication analysis."""
    return """You are a deduplication analyst. Your task is to identify clusters of semantically duplicate scenarios from a list.

Two scenarios are duplicates ONLY if they would produce essentially the same conversation — same specific topic, same user action, same context. Sharing the same broad industry or user action is NOT enough.

Examples of duplicates:
- "Corporate cybersecurity newsletter for executives" and "Writing a security newsletter aimed at C-suite readers" (same specific topic AND same audience)
- "Food blog post about sourdough techniques" and "Baking blog — explaining sourdough starter maintenance" (same specific topic: sourdough)

Examples of NON-duplicates (DO NOT flag these as duplicates):
- "Cloud IT team scheduling a post about AWS migration" and "Cloud IT team scheduling a post about chaos engineering" (different specific topics — scheduling is the action, not the scenario)
- "Enterprise blogger adjusting tone for CISO audience" and "Enterprise blogger adjusting tone for IT leadership" (different audiences, different posts)
- "IT team searching past posts on Terraform" and "IT team searching past posts on container security" (different topics despite same action)
- "Food blog about sourdough" and "Food blog about fermented vegetables" (different topic despite same domain)
- "Analyzing Q4 sales data" and "Analyzing employee attrition data" (different datasets and goals)

Be CONSERVATIVE — when in doubt, keep both scenarios. Only flag true near-duplicates where the conversations would be interchangeable.

## Output Format

Return a JSON object with a single key "clusters". Each cluster has:
- "keep": the scenario_id of the best/most specific version to keep
- "remove": list of scenario_ids that are duplicates of the kept one
- "reason": brief explanation of why these are duplicates

Only include clusters where you found actual duplicates. If no duplicates exist, return {"clusters": []}.

Return valid JSON only. No markdown fences, no explanation outside the JSON."""


def _build_phase1_user_prompt(scenarios: list[dict]) -> str:
    """Build user prompt for Phase 1 (within-batch dedup)."""
    lines = []
    for s in scenarios:
        lines.append(f'{s["scenario_id"]}: {s["scenario"]}')
    scenario_list = '\n'.join(lines)

    return f"""Identify clusters of semantically duplicate scenarios from this batch.

{scenario_list}

Return a JSON object: {{"clusters": [{{"keep": "id", "remove": ["id", ...], "reason": "..."}}]}}"""


def _build_phase2_user_prompt(scenarios: list[dict]) -> str:
    """Build user prompt for Phase 2 (cross-batch consolidation)."""
    lines = []
    for s in scenarios:
        lines.append(f'{s["scenario_id"]}: {s["scenario"]}')
    scenario_list = '\n'.join(lines)

    return f"""Final consolidation pass — find any remaining semantically duplicate scenarios that may span different groups. Be thorough but conservative: only flag true semantic duplicates (same domain/topic AND same user goal).

{scenario_list}

Return a JSON object: {{"clusters": [{{"keep": "id", "remove": ["id", ...], "reason": "..."}}]}}"""


# ── Response parser ──────────────────────────────────────────────────

def _parse_dedup_response(raw: str, valid_ids: set[str]) -> list[dict]:
    """Parse dedup response and validate IDs.

    Returns list of cluster dicts with validated keep/remove IDs.
    """
    # Strip markdown fences if present
    cleaned = raw.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON object in the response
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                log.warning('Failed to parse JSON from dedup response')
                return []
        else:
            log.warning('No JSON object found in dedup response')
            return []

    if not isinstance(parsed, dict):
        log.warning('Parsed result is not a dict: %s', type(parsed))
        return []

    clusters_raw = parsed.get('clusters', [])
    if not isinstance(clusters_raw, list):
        log.warning('clusters is not a list: %s', type(clusters_raw))
        return []

    valid_clusters = []
    for cluster in clusters_raw:
        if not isinstance(cluster, dict):
            continue
        keep = cluster.get('keep')
        remove = cluster.get('remove', [])
        reason = cluster.get('reason', '')

        if not isinstance(keep, str) or keep not in valid_ids:
            log.debug('Skipping cluster with unknown keep ID: %s', keep)
            continue

        if not isinstance(remove, list):
            continue

        valid_remove = [r for r in remove if isinstance(r, str) and r in valid_ids]
        if not valid_remove:
            continue

        # Don't let keep appear in remove
        valid_remove = [r for r in valid_remove if r != keep]
        if not valid_remove:
            continue

        valid_clusters.append({
            'keep': keep,
            'remove': valid_remove,
            'reason': reason,
        })

    return valid_clusters


# ── Cluster merger (transitive closure) ──────────────────────────────

def _merge_clusters(phase1_clusters: list[dict], phase2_clusters: list[dict]) -> set[str]:
    """Merge Phase 1 and Phase 2 clusters, returning the final set of IDs to remove.

    Uses transitive closure: if Phase 1 says keep A remove B, and Phase 2
    says keep C remove A, then C survives and both A+B are removed.
    """
    # Build a mapping: removed_id -> kept_id
    # Process phase 1 first, then phase 2 overrides
    kept_by: dict[str, str] = {}  # removed_id -> the id that replaced it

    for cluster in phase1_clusters:
        for rid in cluster['remove']:
            kept_by[rid] = cluster['keep']

    for cluster in phase2_clusters:
        for rid in cluster['remove']:
            kept_by[rid] = cluster['keep']

    # Transitive closure: follow chains to find ultimate survivor
    def find_survivor(sid: str) -> str:
        visited = set()
        current = sid
        while current in kept_by and current not in visited:
            visited.add(current)
            current = kept_by[current]
        return current

    # Every ID that was ever in a "remove" list — check if its survivor is also removed
    all_removed = set(kept_by.keys())
    final_remove = set()

    for rid in all_removed:
        survivor = find_survivor(rid)
        if survivor != rid:
            final_remove.add(rid)
        # If the survivor itself was also removed (cycle), keep the last one in chain
        # The find_survivor function handles this by stopping at visited nodes

    # Also check: if a "keep" from phase 1 got removed in phase 2,
    # all its dependents should also be removed
    for rid in list(final_remove):
        # Find all IDs that were kept by this removed ID
        dependents = [r for r, k in kept_by.items() if k == rid]
        final_remove.update(dependents)

    return final_remove


# ── File I/O ─────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to a JSONL file (overwrite)."""
    with open(path, 'w') as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def _remove_ids_from_jsonl(path: Path, ids_to_remove: set[str]) -> int:
    """Remove records with matching scenario_ids from a JSONL file.

    Uses atomic temp-file rename to avoid corruption.
    Returns count of removed records.
    """
    if not path.exists():
        return 0

    records = _load_jsonl(path)
    original_count = len(records)
    filtered = [r for r in records if r.get('scenario_id') not in ids_to_remove]
    removed_count = original_count - len(filtered)

    if removed_count > 0:
        # Atomic write via temp file + rename
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                for obj in filtered:
                    f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            shutil.move(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    return removed_count


# ── Phase 1 + Phase 2 runners ───────────────────────────────────────

def _run_phase1(
    scenarios: list[dict],
    active_models: list[dict],
    batch_size: int,
    max_threads: int,
    dry_run: bool = False,
) -> list[dict]:
    """Phase 1: within-batch dedup. Returns list of cluster dicts."""
    system_prompt = _build_dedup_system_prompt()

    # Split into batches
    rng = random.Random(42)
    shuffled = list(scenarios)
    rng.shuffle(shuffled)

    batches: list[list[dict]] = []
    for i in range(0, len(shuffled), batch_size):
        batches.append(shuffled[i:i + batch_size])

    log.info('Phase 1: %d scenarios in %d batches of ~%d', len(scenarios), len(batches), batch_size)

    if dry_run:
        for batch_num, batch in enumerate(batches):
            model_config = active_models[batch_num % len(active_models)]
            user_prompt = _build_phase1_user_prompt(batch)
            print(f'\n{"="*70}')
            print(f'PHASE 1 — BATCH {batch_num + 1}/{len(batches)}')
            print(f'Model: {model_config["name"]} ({model_config["model_id"]})')
            print(f'Scenarios: {len(batch)} ({batch[0]["scenario_id"]} ... {batch[-1]["scenario_id"]})')
            print(f'{"="*70}')
            print(f'\n--- SYSTEM PROMPT ---\n{system_prompt[:400]}...')
            print(f'\n--- USER PROMPT ---\n{user_prompt[:800]}...')
        return []

    # Build wave specs
    all_wave_specs: list[dict] = []
    all_batch_ids: list[set[str]] = []

    for batch_num, batch in enumerate(batches):
        model_config = active_models[batch_num % len(active_models)]
        user_prompt = _build_phase1_user_prompt(batch)
        batch_ids = {s['scenario_id'] for s in batch}

        all_wave_specs.append({
            'model_config': model_config,
            'user_prompt': user_prompt,
            'batch_num': batch_num,
        })
        all_batch_ids.append(batch_ids)

    # Execute in waves
    all_clusters: list[dict] = []
    wave_cursor = 0
    pbar = tqdm(total=len(batches), unit='batches', desc='Phase 1')

    while wave_cursor < len(all_wave_specs):
        wave_end = min(wave_cursor + max_threads, len(all_wave_specs))
        wave_specs = all_wave_specs[wave_cursor:wave_end]
        wave_ids = all_batch_ids[wave_cursor:wave_end]

        semaphore = asyncio.Semaphore(max_threads)
        results = asyncio.run(_run_wave(wave_specs, system_prompt, semaphore))

        for spec, batch_ids, result in zip(wave_specs, wave_ids, results):
            bnum = spec['batch_num'] + 1
            mname = spec['model_config']['name']

            if isinstance(result, BaseException):
                pbar.write(f'ERROR: Phase 1 batch {bnum} failed ({mname}): {result}')
                continue

            clusters = _parse_dedup_response(result, batch_ids)
            if clusters:
                pbar.write(f'Batch {bnum} ({mname}): found {len(clusters)} duplicate clusters')
                all_clusters.extend(clusters)
            else:
                pbar.write(f'Batch {bnum} ({mname}): no duplicates found')

            pbar.update(1)

        wave_cursor = wave_end

    pbar.close()
    return all_clusters


def _run_phase2(
    scenarios: list[dict],
    active_models: list[dict],
    max_threads: int,
    dry_run: bool = False,
) -> list[dict]:
    """Phase 2: cross-batch consolidation. Returns list of cluster dicts."""
    system_prompt = _build_dedup_system_prompt()
    user_prompt = _build_phase2_user_prompt(scenarios)
    valid_ids = {s['scenario_id'] for s in scenarios}

    # Use the first model for the single consolidation call
    model_config = active_models[0]

    log.info('Phase 2: consolidation pass with %d scenarios using %s',
             len(scenarios), model_config['name'])

    if dry_run:
        print(f'\n{"="*70}')
        print(f'PHASE 2 — CONSOLIDATION')
        print(f'Model: {model_config["name"]} ({model_config["model_id"]})')
        print(f'Scenarios: {len(scenarios)}')
        print(f'{"="*70}')
        print(f'\n--- SYSTEM PROMPT ---\n{system_prompt[:400]}...')
        print(f'\n--- USER PROMPT ---\n{user_prompt[:800]}...')
        return []

    # Single call (or small wave)
    wave_specs = [{
        'model_config': model_config,
        'user_prompt': user_prompt,
        'batch_num': 0,
    }]
    semaphore = asyncio.Semaphore(1)
    results = asyncio.run(_run_wave(wave_specs, system_prompt, semaphore))

    result = results[0]
    if isinstance(result, BaseException):
        log.error('Phase 2 consolidation failed: %s', result)
        return []

    clusters = _parse_dedup_response(result, valid_ids)
    if clusters:
        log.info('Phase 2: found %d additional duplicate clusters', len(clusters))
    else:
        log.info('Phase 2: no additional duplicates found')

    return clusters


# ── Backfill wrapper ─────────────────────────────────────────────────

def _backfill(
    domain: str,
    original_count: int,
    models_filter: list[str] | None,
    max_threads: int | None,
    verbose: bool,
) -> None:
    """Backfill by calling generate_scenarios + enrich_scenarios."""
    from datasets.data_aug_pranav.generate_scenarios import generate_scenarios
    from datasets.data_aug_pranav.enrich_scenarios import enrich_scenarios

    log.info('Backfill: generating scenarios to reach target=%d', original_count)

    generate_scenarios(
        domain=domain,
        target=original_count,
        batch_size=12,
        seed=43,
        models_filter=models_filter,
        dry_run=False,
        verbose=verbose,
        max_threads=max_threads,
    )

    log.info('Backfill: enriching new scenarios')

    enrich_scenarios(
        domain=domain,
        batch_size=8,
        seed=43,
        models_filter=models_filter,
        dry_run=False,
        verbose=verbose,
        max_threads=max_threads,
    )


# ── Orchestrator ─────────────────────────────────────────────────────

def dedup_scenarios(
    domain: str,
    batch_size: int = 60,
    seed: int = 43,
    models_filter: list[str] | None = None,
    max_threads: int = 6,
    skip_backfill: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> Path:
    """Main orchestrator: semantic dedup + optional backfill.

    Returns path to the deduped enriched JSONL file.
    """
    rng = random.Random(seed)

    # Filter models
    active_models = MODEL_CONFIGS[:]
    if models_filter:
        active_models = [m for m in active_models if m['name'] in models_filter]
        if not active_models:
            raise ValueError(f'No models match filter: {models_filter}')

    # Input/output paths
    enriched_jsonl = _SCRIPT_DIR / f'scenarios_{domain}_enriched.jsonl'
    base_jsonl = _SCRIPT_DIR / f'scenarios_{domain}.jsonl'
    deduped_jsonl = _SCRIPT_DIR / f'scenarios_{domain}_enriched_deduped.jsonl'
    meta_path = _SCRIPT_DIR / f'scenarios_{domain}_dedup_meta.json'

    # Load enriched scenarios
    enriched = _load_jsonl(enriched_jsonl)
    if not enriched:
        raise FileNotFoundError(f'No enriched scenarios found: {enriched_jsonl}')

    original_count = len(enriched)
    log.info('Loaded %d enriched scenarios for %s', original_count, domain)

    # ── Phase 1: within-batch dedup ──
    phase1_clusters = _run_phase1(
        scenarios=enriched,
        active_models=active_models,
        batch_size=batch_size,
        max_threads=max_threads,
        dry_run=dry_run,
    )

    # Compute Phase 1 removals
    phase1_remove = set()
    for cluster in phase1_clusters:
        phase1_remove.update(cluster['remove'])

    log.info('Phase 1: %d clusters, %d scenarios to remove', len(phase1_clusters), len(phase1_remove))

    # Survivors after Phase 1
    survivors = [s for s in enriched if s['scenario_id'] not in phase1_remove]

    # ── Phase 2: cross-batch consolidation ──
    phase2_clusters = _run_phase2(
        scenarios=survivors,
        active_models=active_models,
        max_threads=max_threads,
        dry_run=dry_run,
    )

    # ── Merge clusters and compute final remove set ──
    final_remove = _merge_clusters(phase1_clusters, phase2_clusters)

    log.info('Final: removing %d duplicates (Phase 1: %d clusters, Phase 2: %d clusters)',
             len(final_remove), len(phase1_clusters), len(phase2_clusters))

    if dry_run:
        print(f'\n[DRY RUN] Would remove {len(final_remove)} duplicates from {original_count} scenarios')
        print(f'Phase 1 clusters: {len(phase1_clusters)}, Phase 2 clusters: {len(phase2_clusters)}')
        return deduped_jsonl

    # ── Write deduped enriched file (survivors only) ──
    final_survivors = [s for s in enriched if s['scenario_id'] not in final_remove]
    _write_jsonl(deduped_jsonl, final_survivors)
    log.info('Wrote %d surviving scenarios to %s', len(final_survivors), deduped_jsonl)

    # ── Backfill ──
    if not skip_backfill and final_remove:
        # Remove deduped IDs from base + enriched JSONL so resume logic detects the gap
        base_removed = _remove_ids_from_jsonl(base_jsonl, final_remove)
        enriched_removed = _remove_ids_from_jsonl(enriched_jsonl, final_remove)
        log.info('Removed %d from base JSONL, %d from enriched JSONL', base_removed, enriched_removed)

        # Backfill: generate + enrich to restore count
        _backfill(
            domain=domain,
            original_count=original_count,
            models_filter=[m['name'] for m in active_models],
            max_threads=max_threads,
            verbose=verbose,
        )

        # Rebuild final deduped file: original survivors + newly enriched backfill
        enriched_after_backfill = _load_jsonl(enriched_jsonl)
        backfill_ids = {s['scenario_id'] for s in final_survivors}
        new_enriched = [s for s in enriched_after_backfill if s['scenario_id'] not in backfill_ids]

        all_final = final_survivors + new_enriched
        _write_jsonl(deduped_jsonl, all_final)
        log.info('Final deduped file: %d scenarios (%d survivors + %d backfill)',
                 len(all_final), len(final_survivors), len(new_enriched))
    elif final_remove:
        log.info('Skipping backfill (--skip-backfill). Deduped count: %d (removed %d)',
                 len(final_survivors), len(final_remove))

    # ── Write metadata ──
    model_dist: dict[str, int] = {}
    for cluster in phase1_clusters + phase2_clusters:
        # Track which models found which clusters (from batch assignment)
        pass  # Model info is implicit from batch round-robin

    meta = {
        'domain': domain,
        'original_count': original_count,
        'duplicates_removed': len(final_remove),
        'removed_ids': sorted(final_remove),
        'survivors_count': len(final_survivors),
        'final_count': len(_load_jsonl(deduped_jsonl)) if deduped_jsonl.exists() else len(final_survivors),
        'phase1_clusters': len(phase1_clusters),
        'phase2_clusters': len(phase2_clusters),
        'phase1_details': phase1_clusters,
        'phase2_details': phase2_clusters,
        'backfill': not skip_backfill and len(final_remove) > 0,
        'batch_size': batch_size,
        'seed': seed,
        'max_threads': max_threads,
        'models': [m['name'] for m in active_models],
    }

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info('Metadata written to %s', meta_path)

    return deduped_jsonl


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    # Load .env
    env_path = _PROJECT_ROOT / '.env'
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

    parser = argparse.ArgumentParser(
        description='Semantic deduplication of enriched scenarios using LLMs',
    )
    parser.add_argument(
        '--domain', required=True, choices=['hugo', 'dana'],
        help='Domain to deduplicate scenarios for',
    )
    parser.add_argument(
        '--batch-size', type=int, default=60,
        help='Scenarios per Phase 1 LLM call (default: 60)',
    )
    parser.add_argument(
        '--seed', type=int, default=43,
        help='Random seed (default: 43)',
    )
    parser.add_argument(
        '--models', type=str, default=None,
        help='Comma-separated model filter (e.g. "anthropic,openai,deepseek")',
    )
    parser.add_argument(
        '--max-threads', type=int, default=6,
        help='Max concurrent API calls (default: 6)',
    )
    parser.add_argument(
        '--skip-backfill', action='store_true',
        help='Dedup only, skip regeneration of removed scenarios',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print prompts without calling LLMs',
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable debug logging',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    # Only our logger gets DEBUG/INFO — keep httpcore/httpx silent
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    models_filter = None
    if args.models:
        models_filter = [m.strip() for m in args.models.split(',')]

    output_path = dedup_scenarios(
        domain=args.domain,
        batch_size=args.batch_size,
        seed=args.seed,
        models_filter=models_filter,
        max_threads=args.max_threads,
        skip_backfill=args.skip_backfill,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    print(f'\nOutput: {output_path}')


if __name__ == '__main__':
    main()
