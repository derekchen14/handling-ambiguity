#!/usr/bin/env python3
"""Compute all synth-vs-eval metrics and write JSON per domain.

No matplotlib — pure computation.  Output is consumed by
``analysis/analyze_synth_vs_eval.py`` for plots + markdown report.

Usage:
    .venv/bin/python3 datasets/data_aug_pranav/compute_metrics.py \
        --domain both --seed 42 --skip-llm --concurrency 10
"""

import argparse
import asyncio
import json
import math
import os
import re
import string
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare, ks_2samp

# ── Constants ────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent          # data_aug_pranav/
DATA_DIR   = SCRIPT_DIR.parent                        # datasets/
ANALYSIS_DIR = SCRIPT_DIR / "analysis"
DOMAINS = ["dana", "hugo"]
CATEGORIES = ["same_flow", "switch_flow", "ambiguous_first", "ambiguous_second"]

THRESHOLDS = {
    "flow_jsd":          (0.05, 0.15),
    "intent_jsd":        (0.05, 0.15),
    "length_ks":         (0.1,  0.3),
    "vocab_jaccard":     (0.6,  0.3),    # higher is better
    "tool_coverage":     (0.95, 0.80),   # higher is better
    "flow_pair_coverage":(0.80, 0.60),   # higher is better
    "naturalness_gap":   (0.3,  0.7),
    "ambiguity_gap":     (0.5,  1.0),
}

INTRINSIC_THRESHOLDS = {
    "flow_entropy_ratio":  (0.85, 0.70),  # uniformity: higher is better
    "tool_entropy_ratio":  (0.85, 0.70),
    "naturalness_mean":    (3.5,  2.5),    # higher is better
    "label_agreement_intent": (0.95, 0.85),  # higher is better
    "label_agreement_flow":   (0.85, 0.70),  # higher is better
    "label_agreement_tool":   (0.85, 0.70),  # higher is better
}

# ── Data Loading ─────────────────────────────────────────────────────

def load_datasets(domain: str):
    """Return (eval_convos, synth_convos) lists of dicts."""
    eval_path  = DATA_DIR / domain / "eval_set.json"
    synth_path = SCRIPT_DIR / f"conversations_{domain}.json"
    if not synth_path.exists():
        synth_path = SCRIPT_DIR / "old_data" / f"conversations_{domain}.json"
    with open(eval_path)  as f: eval_convos  = json.load(f)
    with open(synth_path) as f: synth_convos = json.load(f)
    return eval_convos, synth_convos


def _normalize_target_tools(tt) -> dict:
    """Normalize target_tools to a dict of {tool_name: params}."""
    if isinstance(tt, dict):
        return {k: v for k, v in tt.items() if isinstance(k, str)}
    if isinstance(tt, list):
        result = {}
        for item in tt:
            if isinstance(item, dict) and "name" in item:
                result[item["name"]] = item.get("args", item.get("params", {}))
        return result
    return {}


def flatten_turns(convos: list[dict], source: str) -> list[dict]:
    """Flatten conversations into one record per user turn."""
    records = []
    for c in convos:
        for t in c["turns"]:
            if t["speaker"] != "user":
                continue
            rec = {
                "convo_id":     c["convo_id"],
                "category":     c["category"],
                "flow":         t.get("flow"),
                "intent":       t.get("intent"),
                "utterance":    t["utterance"],
                "target_tools": _normalize_target_tools(t.get("target_tools", {})),
                "turn_num":     t["turn_num"],
                "source":       source,
                "_model":       c.get("_model",    "human"),
                "_provider":    c.get("_provider", "human"),
            }
            if "candidate_flows" in t:
                rec["candidate_flows"] = t["candidate_flows"]
            records.append(rec)
    return records


def flatten_agent_turns(convos: list[dict], source: str) -> list[dict]:
    """Flatten conversations into one record per agent turn."""
    records = []
    for c in convos:
        for t in c["turns"]:
            if t["speaker"] != "agent":
                continue
            records.append({
                "convo_id":  c["convo_id"],
                "category":  c["category"],
                "utterance": t["utterance"],
                "turn_num":  t["turn_num"],
                "source":    source,
            })
    return records


# ── Embedding helper ─────────────────────────────────────────────────

_EMBED_MODEL = None
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

def _encode_texts(texts: list[str]) -> np.ndarray:
    """Encode texts to dense embeddings via sentence-transformers.

    Returns L2-normalized (n_texts, dim) float32 array.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
    vecs = _EMBED_MODEL.encode(texts, show_progress_bar=False,
                                normalize_embeddings=True,
                                convert_to_numpy=True)
    return vecs


# ── Utilities ────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return [w.strip(string.punctuation).lower()
            for w in text.split()
            if w.strip(string.punctuation)]


def word_count(text: str) -> int:
    return len(text.split())


def _to_prob_pair(ca: Counter, cb: Counter):
    """Aligned probability vectors from two Counters."""
    keys = sorted(set(ca) | set(cb))
    if not keys:
        return np.zeros(1), np.zeros(1)
    p = np.array([ca.get(k, 0) for k in keys], dtype=float)
    q = np.array([cb.get(k, 0) for k in keys], dtype=float)
    ps, qs = p.sum(), q.sum()
    if ps: p /= ps
    if qs: q /= qs
    return p, q


def compute_jsd(ca: Counter, cb: Counter) -> float:
    p, q = _to_prob_pair(ca, cb)
    if p.sum() == 0 or q.sum() == 0:
        return 1.0
    return float(jensenshannon(p, q))


def compute_chisq(eval_c: Counter, synth_c: Counter):
    keys = sorted(set(eval_c) | set(synth_c))
    if not keys:
        return 0.0, 1.0
    obs = np.array([synth_c.get(k, 0) for k in keys], dtype=float)
    exp = np.array([eval_c.get(k, 0)  for k in keys], dtype=float)
    mask = exp > 0
    if mask.sum() == 0:
        return 0.0, 1.0
    obs, exp = obs[mask], exp[mask]
    total_obs, total_exp = obs.sum(), exp.sum()
    if total_exp == 0 or total_obs == 0:
        return 0.0, 1.0
    exp = exp * (total_obs / total_exp)
    stat, pval = chisquare(obs, exp)
    return float(stat), float(pval)


def _rating(value, key, higher_is_better=False):
    g, y = THRESHOLDS[key]
    if higher_is_better:
        return "green" if value >= g else ("yellow" if value >= y else "red")
    return "green" if value <= g else ("yellow" if value <= y else "red")


def _entropy(counter: Counter) -> float:
    """Shannon entropy in bits from a Counter."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


# ── Intrinsic Diversity (synth-only) ────────────────────────────────

def synth_flow_diversity(synth_recs):
    sf = Counter(r["flow"] for r in synth_recs if r["flow"] != "ambiguous")
    entropy = _entropy(sf)
    n_unique = len(sf)
    max_entropy = math.log2(n_unique) if n_unique > 1 else 1.0
    uniformity = entropy / max_entropy if max_entropy > 0 else 0.0
    return dict(entropy=entropy, n_unique=n_unique, uniformity=uniformity,
                synth_flows=dict(sf))


def synth_intent_diversity(synth_recs):
    si = Counter(r["intent"] for r in synth_recs if r.get("intent"))
    entropy = _entropy(si)
    n_unique = len(si)
    return dict(entropy=entropy, n_unique=n_unique, synth_intents=dict(si))


def synth_tool_diversity(synth_recs):
    tc = Counter()
    tools_per_turn = []
    for r in synth_recs:
        tools = list(r.get("target_tools", {}))
        tools_per_turn.append(len(tools))
        for t in tools:
            tc[t] += 1
    entropy = _entropy(tc)
    n_unique = len(tc)
    mean_tpt = float(np.mean(tools_per_turn)) if tools_per_turn else 0.0
    return dict(entropy=entropy, n_unique=n_unique, mean_tools_per_turn=mean_tpt,
                synth_tool_counts=dict(tc))


# ── 1. Flow Distribution ────────────────────────────────────────────

def flow_distribution(eval_recs, synth_recs, domain):
    ef = Counter(r["flow"] for r in eval_recs  if r["flow"] != "ambiguous")
    sf = Counter(r["flow"] for r in synth_recs if r["flow"] != "ambiguous")
    jsd = compute_jsd(ef, sf)
    chi_stat, chi_pval = compute_chisq(ef, sf)

    all_flows = sorted(set(ef) | set(sf))
    et, st = sum(ef.values()), sum(sf.values())

    flagged = []
    for f in all_flows:
        ep = ef.get(f, 0) / et if et else 0
        sp = sf.get(f, 0) / st if st else 0
        ratio = sp / ep if ep > 0 else float("inf")
        if ratio < 0.5 or ratio > 2.0:
            flagged.append((f, ratio))

    return dict(jsd=jsd, chi_stat=chi_stat, chi_pval=chi_pval,
                eval_flows=dict(ef), synth_flows=dict(sf),
                flagged_ratios=flagged)


# ── 2. Intent Distribution ──────────────────────────────────────────

def intent_distribution(eval_recs, synth_recs, domain):
    ei = Counter(r["intent"] for r in eval_recs  if r.get("intent"))
    si = Counter(r["intent"] for r in synth_recs if r.get("intent"))
    jsd = compute_jsd(ei, si)
    chi_stat, chi_pval = compute_chisq(ei, si)
    return dict(jsd=jsd, chi_stat=chi_stat, chi_pval=chi_pval,
                eval_intents=dict(ei), synth_intents=dict(si))


# ── 3. Category Distribution ────────────────────────────────────────

def category_distribution(eval_recs, synth_recs, domain):
    ec, sc = Counter(), Counter()
    seen_e, seen_s = set(), set()
    for r in eval_recs:
        if r["convo_id"] not in seen_e:
            ec[r["category"]] += 1; seen_e.add(r["convo_id"])
    for r in synth_recs:
        if r["convo_id"] not in seen_s:
            sc[r["category"]] += 1; seen_s.add(r["convo_id"])
    return dict(eval_categories=dict(ec), synth_categories=dict(sc))


# ── 4. Utterance Length ──────────────────────────────────────────────

def utterance_length(eval_recs, synth_recs, domain):
    def _lens(recs, tn=None):
        return [word_count(r["utterance"]) for r in recs if tn is None or r["turn_num"] == tn]
    et1, et3 = _lens(eval_recs, 1),  _lens(eval_recs, 3)
    st1, st3 = _lens(synth_recs, 1), _lens(synth_recs, 3)

    ks1s, ks1p = ks_2samp(et1, st1)
    ks3s, ks3p = ks_2samp(et3, st3)

    def _stats(a):
        a = np.asarray(a, dtype=float)
        return dict(mean=float(a.mean()), median=float(np.median(a)),
                    std=float(a.std()), p10=float(np.percentile(a, 10)),
                    p90=float(np.percentile(a, 90)))

    # Category-level lengths for boxplot
    cats = sorted({r["category"] for r in eval_recs})
    category_lengths = {}
    for c in cats:
        category_lengths[c] = {
            "eval":  [word_count(r["utterance"]) for r in eval_recs  if r["category"] == c],
            "synth": [word_count(r["utterance"]) for r in synth_recs if r["category"] == c],
        }

    return dict(
        ks_t1=dict(stat=float(ks1s), pval=float(ks1p)),
        ks_t3=dict(stat=float(ks3s), pval=float(ks3p)),
        eval_t1_stats=_stats(et1), eval_t3_stats=_stats(et3),
        synth_t1_stats=_stats(st1), synth_t3_stats=_stats(st3),
        eval_t1_lengths=et1, eval_t3_lengths=et3,
        synth_t1_lengths=st1, synth_t3_lengths=st3,
        category_lengths=category_lengths,
    )


# ── 5. Vocabulary ────────────────────────────────────────────────────

def vocabulary_analysis(eval_recs, synth_recs, domain):
    et, st = [], []
    for r in eval_recs:  et.extend(tokenize(r["utterance"]))
    for r in synth_recs: st.extend(tokenize(r["utterance"]))

    ev, sv = set(et), set(st)
    ettr = len(ev) / len(et) if et else 0
    sttr = len(sv) / len(st) if st else 0
    jacc = len(ev & sv) / len(ev | sv) if (ev | sv) else 0

    ef = Counter(et)
    excl = {w: ef[w] for w in ev - sv if ef[w] >= 2}

    return dict(eval_ttr=ettr, synth_ttr=sttr, jaccard=jacc,
                eval_vocab_size=len(ev), synth_vocab_size=len(sv),
                eval_exclusive_words=dict(sorted(excl.items(), key=lambda x: -x[1])[:30]))


# ── 6. Tool Usage ────────────────────────────────────────────────────

def tool_usage(eval_recs, synth_recs, domain):
    def _tc(recs):
        c = Counter()
        for r in recs:
            for t in r.get("target_tools", {}):
                c[t] += 1
        return c
    ec, sc = _tc(eval_recs), _tc(synth_recs)
    jsd = compute_jsd(ec, sc)
    chi_stat, chi_pval = compute_chisq(ec, sc)
    es, ss = set(ec), set(sc)
    gaps  = es - ss
    noise = ss - es
    cov   = len(es & ss) / len(es) if es else 1.0

    etsizes = [len(r.get("target_tools", {})) for r in eval_recs]
    stsizes = [len(r.get("target_tools", {})) for r in synth_recs]

    return dict(jsd=jsd, chi_stat=chi_stat, chi_pval=chi_pval,
                coverage_pct=cov, coverage_gaps=sorted(gaps), noise_tools=sorted(noise),
                eval_mean_tools=float(np.mean(etsizes)), synth_mean_tools=float(np.mean(stsizes)),
                eval_tool_counts=dict(ec), synth_tool_counts=dict(sc))


# ── 7. Flow Co-occurrence ────────────────────────────────────────────

def flow_cooccurrence(eval_convos, synth_convos, domain):
    def _trans(convos):
        tr = Counter()
        for c in convos:
            ut = [t for t in c["turns"] if t["speaker"] == "user"]
            if len(ut) >= 2:
                tr[(ut[0].get("flow", "?"), ut[1].get("flow", "?"))] += 1
        return tr

    et, st = _trans(eval_convos), _trans(synth_convos)
    ep, sp = set(et), set(st)
    missing = ep - sp

    # cosine similarity
    all_p = sorted(set(et) | set(st))
    ev = np.array([et.get(p, 0) for p in all_p], dtype=float)
    sv = np.array([st.get(p, 0) for p in all_p], dtype=float)
    ne, ns = np.linalg.norm(ev), np.linalg.norm(sv)
    cos = float(np.dot(ev, sv) / (ne * ns)) if ne and ns else 0.0

    cov = len(ep & sp) / len(ep) if ep else 1.0

    # top flows for heatmap reconstruction
    ff = Counter()
    for (f1, f3), n in list(et.items()) + list(st.items()):
        ff[f1] += n; ff[f3] += n
    top_flows = [f for f, _ in ff.most_common(15)]

    # Serialise transitions as "flow_a->flow_b": count
    eval_transitions = {f"{a}->{b}": n for (a, b), n in et.items()}
    synth_transitions = {f"{a}->{b}": n for (a, b), n in st.items()}

    return dict(cosine_sim=cos, coverage_pct=cov,
                missing_pairs=[(a, b) for a, b in sorted(missing)],
                num_eval_pairs=len(ep), num_synth_pairs=len(sp),
                eval_transitions=eval_transitions, synth_transitions=synth_transitions,
                top_flows=top_flows)


# ── 8. Embedding Similarity (TF-IDF, no sklearn) ────────────────────

def embedding_similarity(eval_recs, synth_recs, domain, seed=42):
    all_recs = eval_recs + synth_recs
    texts    = [r["utterance"] for r in all_recs]
    sources  = [r["source"]    for r in all_recs]
    cats     = [r["category"]  for r in all_recs]

    tfidf = _encode_texts(texts)

    # cosine similarities
    em = np.array([s == "eval"  for s in sources])
    sm = np.array([s == "synth" for s in sources])
    ev, sv = tfidf[em], tfidf[sm]

    rng = np.random.RandomState(seed)
    ns = min(100, len(ev), len(sv))
    es = ev[rng.choice(len(ev), ns, replace=False)]
    ss = sv[rng.choice(len(sv), ns, replace=False)]

    within_e = float(np.mean(es @ es.T))
    within_s = float(np.mean(ss @ ss.T))
    cross    = float(np.mean(es @ ss.T))

    # PCA via truncated SVD
    mu = tfidf.mean(axis=0)
    centered = tfidf - mu
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    pca = U[:, :2] * S[:2]

    pca_coords = [
        dict(x=float(pca[i, 0]), y=float(pca[i, 1]), source=sources[i], category=cats[i])
        for i in range(len(sources))
    ]

    return dict(within_eval=within_e, within_synth=within_s, cross_set=cross,
                well_mixed=cross > min(within_e, within_s),
                pca_coords=pca_coords)


# ── 9. Model Effects ────────────────────────────────────────────────

def model_effects(synth_recs):
    by = defaultdict(list)
    for r in synth_recs: by[r["_provider"]].append(r)

    stats = {}
    provider_lengths: dict[str, list[int]] = {}
    for prov, recs in by.items():
        lens = [word_count(r["utterance"]) for r in recs]
        toks = []
        for r in recs: toks.extend(tokenize(r["utterance"]))
        v = set(toks)
        ttr = len(v) / len(toks) if toks else 0
        fc = Counter(r["flow"] for r in recs if r["flow"] != "ambiguous")
        stats[prov] = dict(n=len(recs), mean_length=float(np.mean(lens)),
                           std_length=float(np.std(lens)), ttr=ttr,
                           top_flows=fc.most_common(5))
        provider_lengths[prov] = lens

    return dict(model_stats=stats, provider_lengths=provider_lengths)


# ── 10. Parameter Completeness ──────────────────────────────────────

def parameter_completeness(eval_recs, synth_recs, domain):
    def _nr(recs):
        total = null = 0
        for r in recs:
            for tool, params in r.get("target_tools", {}).items():
                if isinstance(params, dict):
                    for v in params.values():
                        total += 1
                        if v is None: null += 1
        return (null / total if total else 0), total, null

    er, et, en = _nr(eval_recs)
    sr, st, sn = _nr(synth_recs)
    risk = "high" if sr < er * 0.5 else "low"
    return dict(eval_null_rate=er, synth_null_rate=sr,
                eval_total_params=et, eval_null_params=en,
                synth_total_params=st, synth_null_params=sn, risk=risk)


# ── 11. Context Dependence ──────────────────────────────────────────

def context_dependence(eval_recs, synth_recs, domain):
    def _tr(recs, thresh=8):
        t3 = [r for r in recs if r["turn_num"] == 3]
        short = [r for r in t3 if word_count(r["utterance"]) < thresh]
        return (len(short) / len(t3) if t3 else 0), len(short), len(t3)
    er, en, et = _tr(eval_recs)
    sr, sn, st = _tr(synth_recs)
    return dict(eval_terse_rate=er, synth_terse_rate=sr,
                eval_terse=en, eval_total=et,
                synth_terse=sn, synth_total=st)


# ── 12. Per-Category Metrics ────────────────────────────────────────

def per_category_metrics(eval_recs, synth_recs, domain):
    results = {}
    cats = sorted({r["category"] for r in eval_recs} | {r["category"] for r in synth_recs})

    for cat in cats:
        er = [r for r in eval_recs if r["category"] == cat]
        sr = [r for r in synth_recs if r["category"] == cat]
        if not er or not sr:
            results[cat] = dict(flow_jsd=1.0, tool_jsd=1.0, length_ks=1.0,
                                terse_rate_eval=0, terse_rate_synth=0)
            continue

        ef = Counter(r["flow"] for r in er if r["flow"] != "ambiguous")
        sf = Counter(r["flow"] for r in sr if r["flow"] != "ambiguous")
        flow_jsd = compute_jsd(ef, sf)

        def _tc(recs):
            c = Counter()
            for r in recs:
                for t in r.get("target_tools", {}):
                    c[t] += 1
            return c
        tool_jsd = compute_jsd(_tc(er), _tc(sr))

        el3 = [word_count(r["utterance"]) for r in er if r["turn_num"] == 3]
        sl3 = [word_count(r["utterance"]) for r in sr if r["turn_num"] == 3]
        length_ks = float(ks_2samp(el3, sl3).statistic) if el3 and sl3 else 1.0

        def _terse(recs, thresh=8):
            t3 = [r for r in recs if r["turn_num"] == 3]
            short = [r for r in t3 if word_count(r["utterance"]) < thresh]
            return len(short) / len(t3) if t3 else 0

        results[cat] = dict(flow_jsd=flow_jsd, tool_jsd=tool_jsd,
                            length_ks=length_ks,
                            terse_rate_eval=_terse(er), terse_rate_synth=_terse(sr))

    return dict(categories=results)


# ── 13. Turn Position Analysis ──────────────────────────────────────

def turn_position_analysis(eval_recs, synth_recs, domain):
    result = {}
    for tn in (1, 3):
        er = [r for r in eval_recs if r["turn_num"] == tn]
        sr = [r for r in synth_recs if r["turn_num"] == tn]

        ef = Counter(r["flow"] for r in er if r["flow"] != "ambiguous")
        sf = Counter(r["flow"] for r in sr if r["flow"] != "ambiguous")
        flow_jsd = compute_jsd(ef, sf)

        def _tc(recs):
            c = Counter()
            for r in recs:
                for t in r.get("target_tools", {}):
                    c[t] += 1
            return c
        ec, sc = _tc(er), _tc(sr)
        tool_jsd = compute_jsd(ec, sc)

        result[f"turn_{tn}"] = dict(flow_jsd=flow_jsd, tool_jsd=tool_jsd,
                                    eval_flows=dict(ef), synth_flows=dict(sf),
                                    eval_tool_counts=dict(ec), synth_tool_counts=dict(sc))

    t1_fj = result["turn_1"]["flow_jsd"]
    t3_fj = result["turn_3"]["flow_jsd"]
    result["t3_worse"] = t3_fj > t1_fj * 1.5 if t1_fj > 0 else t3_fj > 0.1
    return result


# ── 14. Agent Response Analysis ─────────────────────────────────────

def agent_response_analysis(eval_convos, synth_convos, domain):
    eval_agents = flatten_agent_turns(eval_convos, "eval")
    synth_agents = flatten_agent_turns(synth_convos, "synth")

    el = [word_count(r["utterance"]) for r in eval_agents]
    sl = [word_count(r["utterance"]) for r in synth_agents]

    length_ks = float(ks_2samp(el, sl).statistic) if el and sl else 1.0

    ev_toks, sv_toks = set(), set()
    for r in eval_agents:
        ev_toks.update(tokenize(r["utterance"]))
    for r in synth_agents:
        sv_toks.update(tokenize(r["utterance"]))
    vocab_jaccard = len(ev_toks & sv_toks) / len(ev_toks | sv_toks) if (ev_toks | sv_toks) else 0

    return dict(length_ks=length_ks, vocab_jaccard=vocab_jaccard,
                eval_mean_length=float(np.mean(el)) if el else 0,
                synth_mean_length=float(np.mean(sl)) if sl else 0,
                eval_lengths=el, synth_lengths=sl)


# ── 15. Conditional Distributions ───────────────────────────────────

def conditional_distributions(eval_recs, synth_recs, domain):
    def _tool_given_flow(recs):
        by_flow = defaultdict(Counter)
        for r in recs:
            f = r.get("flow")
            if not f or f == "ambiguous":
                continue
            for t in r.get("target_tools", {}):
                by_flow[f][t] += 1
        return dict(by_flow)

    e_tf = _tool_given_flow(eval_recs)
    s_tf = _tool_given_flow(synth_recs)
    common_flows = sorted(set(e_tf) & set(s_tf))

    per_flow_jsd = {}
    for f in common_flows:
        per_flow_jsd[f] = compute_jsd(e_tf[f], s_tf[f])

    avg_tool_given_flow = float(np.mean(list(per_flow_jsd.values()))) if per_flow_jsd else 1.0
    worst_flows = sorted(per_flow_jsd.items(), key=lambda x: -x[1])[:5]

    def _flow_given_intent(recs):
        by_intent = defaultdict(Counter)
        for r in recs:
            i = r.get("intent")
            f = r.get("flow")
            if not i or not f or f == "ambiguous":
                continue
            by_intent[i][f] += 1
        return dict(by_intent)

    e_fi = _flow_given_intent(eval_recs)
    s_fi = _flow_given_intent(synth_recs)
    common_intents = sorted(set(e_fi) & set(s_fi))

    per_intent_jsd = {}
    for i in common_intents:
        per_intent_jsd[i] = compute_jsd(e_fi[i], s_fi[i])

    avg_flow_given_intent = float(np.mean(list(per_intent_jsd.values()))) if per_intent_jsd else 1.0

    return dict(tool_given_flow_avg_jsd=avg_tool_given_flow,
                flow_given_intent_avg_jsd=avg_flow_given_intent,
                per_flow_jsd=per_flow_jsd,
                per_intent_jsd=per_intent_jsd,
                worst_flow_conditionals=worst_flows)


# ── 16. Scenario Topic Coverage ─────────────────────────────────────

def scenario_topic_coverage(eval_convos, synth_convos, domain, seed=42, n_clusters=8):
    from scipy.cluster.vq import kmeans2

    def _scenarios(convos, src):
        return [(c.get("scenario", ""), src) for c in convos]

    all_sc = _scenarios(eval_convos, "eval") + _scenarios(synth_convos, "synth")
    texts = [s for s, _ in all_sc]
    sources = [s for _, s in all_sc]

    n_docs = len(texts)
    if n_docs < n_clusters:
        return dict(n_clusters=0, eval_only_clusters=0, synth_only_clusters=0,
                    coverage_pct=0.0)

    tfidf = _encode_texts(texts)

    rng = np.random.RandomState(seed)
    stds = tfidf.std(axis=0)
    stds[stds == 0] = 1
    whitened = tfidf / stds

    try:
        centroids, labels = kmeans2(whitened, n_clusters, minit="points", seed=rng)
    except Exception:
        return dict(n_clusters=0, eval_only_clusters=0, synth_only_clusters=0,
                    coverage_pct=0.0)

    cluster_sources = defaultdict(lambda: {"eval": 0, "synth": 0})
    for lbl, src in zip(labels, sources):
        cluster_sources[lbl][src] += 1

    eval_only = sum(1 for cs in cluster_sources.values() if cs["eval"] > 0 and cs["synth"] == 0)
    synth_only = sum(1 for cs in cluster_sources.values() if cs["synth"] > 0 and cs["eval"] == 0)
    eval_clusters = sum(1 for cs in cluster_sources.values() if cs["eval"] > 0)
    covered = sum(1 for cs in cluster_sources.values() if cs["eval"] > 0 and cs["synth"] > 0)
    coverage_pct = covered / eval_clusters if eval_clusters else 1.0

    # PCA for scatter coordinates
    mu = tfidf.mean(axis=0)
    centered = tfidf - mu
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    pca = U[:, :2] * S[:2]

    pca_coords = [
        dict(x=float(pca[i, 0]), y=float(pca[i, 1]),
             source=sources[i], cluster=int(labels[i]))
        for i in range(len(sources))
    ]

    return dict(n_clusters=n_clusters, eval_only_clusters=eval_only,
                synth_only_clusters=synth_only, coverage_pct=coverage_pct,
                cluster_details={int(k): v for k, v in cluster_sources.items()},
                pca_coords=pca_coords)


# ── LLM Helpers ──────────────────────────────────────────────────────

async def _call_llm(prompt: str, client, semaphore: asyncio.Semaphore) -> dict:
    """Single Claude Sonnet call; returns parsed JSON."""
    async with semaphore:
        try:
            resp = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = next(b.text for b in resp.content if hasattr(b, "text"))
            m = re.search(r"\{[^}]+\}", text)
            return json.loads(m.group()) if m else json.loads(text)
        except Exception as e:
            return {"score": -1, "reason": f"Error: {e}"}


# ── Full-Corpus Naturalness Judge ────────────────────────────────────

NATURALNESS_PROMPT_FULL = """Rate whether this conversation sounds like a real user or synthetic/contrived (1-5):
1 = Completely contrived — robotic, formulaic, no personality
2 = Mostly contrived — overly explicit about intent, reads like a template
3 = Mixed — some natural, some forced
4 = Mostly natural — minor tells (too polished, suspiciously complete context)
5 = Indistinguishable from real user — natural shorthand, personality, context-appropriate

Attention pointers:
- Turn 3: natural follow-up vs scripted?
- User uses pronouns/shorthand or spells everything out?
- Realistic scenario for someone actually using this tool?

Domain context: {domain}

Conversation:
Turn 1 (user): {turn1}
Turn 2 (agent): {turn2}
Turn 3 (user): {turn3}

Respond in JSON only: {{"score": <1-5>, "reason": "<brief explanation>"}}"""


async def judge_naturalness_all(eval_convos, synth_convos, domain, seed=42, concurrency=10):
    """Score ALL synth conversations + stratified sample of 30 eval (reference)."""
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    sem = asyncio.Semaphore(concurrency)
    rng = np.random.RandomState(seed)

    # Stratified sample of eval for reference
    def _sample(convos, n=30):
        by_cat = defaultdict(list)
        for c in convos: by_cat[c["category"]].append(c)
        per = max(1, n // len(by_cat))
        out = []
        for cs in by_cat.values():
            idx = rng.choice(len(cs), min(per, len(cs)), replace=False)
            out.extend(cs[i] for i in idx)
        return out[:n]

    async def _rate(c, src):
        ts = c["turns"]
        p = NATURALNESS_PROMPT_FULL.format(
            domain=domain, turn1=ts[0]["utterance"],
            turn2=ts[1]["utterance"], turn3=ts[2]["utterance"])
        r = await _call_llm(p, client, sem)
        r["source"] = src
        r["category"] = c["category"]
        r["convo_id"] = c["convo_id"]
        if "_provider" in c: r["model"] = c["_provider"]
        return r

    # ALL synth + sample of eval
    tasks = [_rate(c, "eval") for c in _sample(eval_convos)]
    tasks += [_rate(c, "synth") for c in synth_convos]
    results = await asyncio.gather(*tasks)
    await client.close()

    # Per-conversation results
    per_conversation = {}
    for r in results:
        cid = r.get("convo_id")
        if cid:
            per_conversation[cid] = {
                "score": r.get("score", -1),
                "reason": r.get("reason", ""),
                "source": r["source"],
                "category": r["category"],
            }
            if "model" in r:
                per_conversation[cid]["model"] = r["model"]

    # Summaries
    es = [r["score"] for r in results if r["source"] == "eval"  and r.get("score", -1) > 0]
    ss = [r["score"] for r in results if r["source"] == "synth" and r.get("score", -1) > 0]

    from scipy.stats import ttest_ind
    t_stat, t_pval = ttest_ind(es, ss, equal_var=False) if es and ss else (0.0, 1.0)

    # By category
    by_cat_synth = defaultdict(list)
    for r in results:
        if r["source"] == "synth" and r.get("score", -1) > 0:
            by_cat_synth[r["category"]].append(r["score"])
    by_category = {c: {"mean": float(np.mean(s)), "std": float(np.std(s)), "n": len(s)}
                   for c, s in by_cat_synth.items()}

    # By model
    by_model = defaultdict(list)
    for r in results:
        if r["source"] == "synth" and r.get("score", -1) > 0 and "model" in r:
            by_model[r["model"]].append(r["score"])
    model_breakdown = {m: {"mean": float(np.mean(s)), "std": float(np.std(s)), "n": len(s)}
                       for m, s in by_model.items()}

    # Contrived IDs (score <= 2)
    contrived_ids = sorted(
        cid for cid, info in per_conversation.items()
        if info["source"] == "synth" and info["score"] <= 2
    )

    eval_mean = float(np.mean(es)) if es else 0
    synth_mean = float(np.mean(ss)) if ss else 0

    return dict(
        per_conversation=per_conversation,
        synth_summary=dict(mean=synth_mean, std=float(np.std(ss)) if ss else 0,
                           n=len(ss), by_category=by_category, by_model=model_breakdown),
        eval_summary=dict(mean=eval_mean, std=float(np.std(es)) if es else 0, n=len(es)),
        gap=abs(eval_mean - synth_mean),
        t_stat=float(t_stat), t_pval=float(t_pval),
        contrived_ids=contrived_ids,
    )


# ── Existing LLM Judges (sample-based) ──────────────────────────────

AMBIGUITY_PROMPT = """Rate the quality of ambiguity in this user utterance (1-5 scale).

1 = Not actually ambiguous — clearly refers to one intent
2 = Weakly ambiguous — slight ambiguity but most readers would agree on intent
3 = Moderately ambiguous — could go either way but one reading is more natural
4 = Well-calibrated ambiguity — genuinely unclear between the candidates
5 = Perfect ambiguity — equally valid interpretations

Category: {category}
Candidate flows: {candidate_flows}
Utterance: {utterance}
{extra_context}

Respond in JSON only: {{"score": <1-5>, "reason": "<brief explanation>"}}"""


DIVERSITY_PROMPT = """Rate the diversity of the following batch of conversation scenarios on three dimensions (1-5 each):

1. **Topic diversity**: How varied are the subjects/domains covered?
2. **Task diversity**: How varied are the types of tasks users request?
3. **Complexity diversity**: How varied is the difficulty/complexity of requests?

Scenarios:
{scenarios}

Respond in JSON only: {{"topic": <1-5>, "task": <1-5>, "complexity": <1-5>, "reason": "<brief explanation>"}}"""


async def ambiguity_quality_analysis(eval_convos, synth_convos, domain, seed=42, concurrency=10):
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    sem = asyncio.Semaphore(concurrency)
    rng = np.random.RandomState(seed)

    def _amb_turns(convos, src):
        turns = []
        for c in convos:
            for t in c["turns"]:
                if t["speaker"] != "user":
                    continue
                if t.get("flow") == "ambiguous":
                    turns.append(dict(utterance=t["utterance"],
                                      candidate_flows=t.get("candidate_flows", []),
                                      category=c["category"], source=src,
                                      model=c.get("_provider", "human")))
                elif c["category"] == "ambiguous_second" and t["turn_num"] == 3:
                    turns.append(dict(utterance=t["utterance"],
                                      candidate_flows=t.get("candidate_flows", []),
                                      category=c["category"], source=src,
                                      model=c.get("_provider", "human")))
        return turns

    ea = _amb_turns(eval_convos, "eval")
    sa = _amb_turns(synth_convos, "synth")
    if len(sa) > 60:
        idx = rng.choice(len(sa), 60, replace=False)
        sa = [sa[i] for i in idx]

    async def _rate(turn):
        extra = ("This is a multi-request turn that should genuinely require BOTH flows."
                 if turn["category"] == "ambiguous_second" else "")
        p = AMBIGUITY_PROMPT.format(category=turn["category"],
                                     candidate_flows=turn["candidate_flows"],
                                     utterance=turn["utterance"],
                                     extra_context=extra)
        r = await _call_llm(p, client, sem)
        r.update(turn)
        return r

    results = await asyncio.gather(*[_rate(t) for t in ea + sa])
    await client.close()

    es = [r["score"] for r in results if r["source"] == "eval"  and r.get("score", -1) > 0]
    ss = [r["score"] for r in results if r["source"] == "synth" and r.get("score", -1) > 0]

    from scipy.stats import ttest_ind
    t, p = ttest_ind(es, ss, equal_var=False) if es and ss else (0.0, 1.0)

    return dict(eval_mean=float(np.mean(es)) if es else 0, eval_std=float(np.std(es)) if es else 0,
                synth_mean=float(np.mean(ss)) if ss else 0, synth_std=float(np.std(ss)) if ss else 0,
                gap=abs(float(np.mean(es)) - float(np.mean(ss))) if es and ss else 0,
                t_stat=float(t), t_pval=float(p), n_eval=len(es), n_synth=len(ss))


async def scenario_diversity_analysis(eval_convos, synth_convos, domain, concurrency=5):
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    sem = asyncio.Semaphore(concurrency)

    def _scenarios(convos):
        return [c.get("scenario", f"Category: {c['category']}") for c in convos]

    async def _batch(scenarios, src):
        results = []
        bs = 20
        for i in range(0, len(scenarios), bs):
            batch = scenarios[i:i + bs]
            numbered = "\n".join(f"{j + 1}. {s}" for j, s in enumerate(batch))
            r = await _call_llm(DIVERSITY_PROMPT.format(scenarios=numbered), client, sem)
            r["source"] = src
            results.append(r)
        return results

    er = await _batch(_scenarios(eval_convos), "eval")
    sr = await _batch(_scenarios(synth_convos), "synth")
    await client.close()

    def _avg(results):
        return {d: float(np.mean([r.get(d, 0) for r in results if r.get(d, 0) > 0]) or 0)
                for d in ("topic", "task", "complexity")}

    return dict(eval=_avg(er), synth=_avg(sr),
                n_eval_batches=len(er), n_synth_batches=len(sr))


# ── Label Agreement (Ensemble) ────────────────────────────────────────

PROJECT_ROOT = SCRIPT_DIR.parent.parent  # project root (handling-ambiguity/)


def _run_runner(runner_script: str, domain: str, config: str, seeds: list[int],
                temperature: float, eval_path: Path, output_dir: str,
                mode: str | None = None) -> list[Path]:
    """Run an experiment runner via subprocess and return result JSONL paths."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / runner_script),
        "--domain", domain,
        "--config", config,
        "--seeds", ",".join(str(s) for s in seeds),
        "--temperature", str(temperature),
        "--eval-path", str(eval_path),
        "--output-dir", output_dir,
    ]
    if mode:
        cmd += ["--mode", mode]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"    WARNING: {runner_script} failed: {result.stderr[-500:]}")
        return []

    # Find result JSONL files
    results_dir = PROJECT_ROOT / "results" / output_dir
    paths = []
    for seed in seeds:
        pattern = f"{domain}_{config}_seed{seed}.jsonl"
        matches = list(results_dir.rglob(pattern))
        if matches:
            paths.append(matches[0])
    return paths


def _parse_results_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL result file into list of conversation dicts."""
    convos = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                convos.append(json.loads(line))
    return convos


def _majority_vote_scalar(predictions: list[str | None]) -> str | None:
    """Majority vote for scalar predictions (intent, flow)."""
    valid = [p for p in predictions if p is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def _majority_vote_tools(tool_sets: list[set[str]], n_voters: int) -> set[str]:
    """Majority vote for tool sets: keep tools predicted by >= ceil(N/2) voters."""
    threshold = math.ceil(n_voters / 2)
    tool_counts: Counter = Counter()
    for ts in tool_sets:
        tool_counts.update(ts)
    return {t for t, c in tool_counts.items() if c >= threshold}


def run_label_agreement(
    synth_convos: list[dict],
    domain: str,
    seed: int,
    configs: dict[str, str],
    n_voters: int = 3,
    temperature: float = 0.3,
) -> dict:
    """Run ensemble label agreement check against synth data labels.

    For each stage (intent, flow, tool), runs the appropriate experiment runner
    N times with different seeds, majority-votes predictions, and compares to
    gold labels from the synth data.
    """
    # Write synth convos to temp file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="synth_label_check_",
        dir=str(PROJECT_ROOT), delete=False,
    )
    json.dump(synth_convos, tmp, indent=2)
    tmp.close()
    eval_path = Path(tmp.name)

    output_dir = "_label_check"
    voter_seeds = list(range(seed, seed + n_voters))

    results = {}

    # Stage definitions: (stage_name, runner_script, mode, pred_extractor, gold_extractor)
    stage_defs = {
        "intent": {
            "runner": "exp2_runner.py",
            "mode": "intent",
            "extract_pred": lambda t: t.get("detected_intent"),
            "extract_gold": lambda t: t.get("intent"),
            "vote_fn": "scalar",
        },
        "flow": {
            "runner": "exp1_runner.py",
            "mode": None,
            "extract_pred": lambda t: t["detected_flows"][0] if t.get("detected_flows") else None,
            "extract_gold": lambda t: t.get("flow"),
            "vote_fn": "scalar",
        },
        "tool": {
            "runner": "exp2_runner.py",
            "mode": "scoped_tool",
            "extract_pred": lambda t: set(t.get("predicted_tools", [])),
            "extract_gold": lambda t: set(t.get("gold_tools", [])),
            "vote_fn": "tools",
        },
    }

    try:
        for stage_name, sdef in stage_defs.items():
            config = configs.get(stage_name)
            if not config:
                continue
            print(f"    Running {stage_name} label check ({n_voters} voters, config={config})...")

            # Run the runner
            jsonl_paths = _run_runner(
                sdef["runner"], domain, config, voter_seeds, temperature,
                eval_path, output_dir, sdef["mode"],
            )
            if not jsonl_paths:
                print(f"    WARNING: No results for {stage_name}, skipping")
                continue

            # Parse results: keyed by (convo_id, turn_num) -> list of predictions
            predictions_by_turn: dict[tuple[str, int], list] = defaultdict(list)
            gold_by_turn: dict[tuple[str, int], Any] = {}
            category_by_turn: dict[tuple[str, int], str] = {}

            for path in jsonl_paths:
                convos = _parse_results_jsonl(path)
                for convo in convos:
                    cid = convo["convo_id"]
                    cat = convo.get("category", "")
                    for turn in convo.get("turns", []):
                        tn = turn["turn_num"]
                        key = (cid, tn)
                        category_by_turn[key] = cat

                        pred = sdef["extract_pred"](turn)
                        predictions_by_turn[key].append(pred)

                        if key not in gold_by_turn:
                            gold_by_turn[key] = sdef["extract_gold"](turn)

            # Majority vote and compare
            n_turns = 0
            n_agree = 0
            by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "agree": 0})

            for key, preds in predictions_by_turn.items():
                gold = gold_by_turn.get(key)
                if gold is None:
                    continue
                cat = category_by_turn.get(key, "unknown")

                if sdef["vote_fn"] == "scalar":
                    consensus = _majority_vote_scalar(preds)
                    match = consensus == gold
                else:
                    # Tool sets
                    tool_sets = [p if isinstance(p, set) else set() for p in preds]
                    consensus = _majority_vote_tools(tool_sets, n_voters)
                    gold_set = gold if isinstance(gold, set) else set()
                    # Skip trivially correct turns (ambiguous, zero-tool, etc.)
                    if not gold_set:
                        continue
                    match = consensus == gold_set

                n_turns += 1
                if match:
                    n_agree += 1
                by_category[cat]["total"] += 1
                if match:
                    by_category[cat]["agree"] += 1

            agreement = n_agree / n_turns if n_turns > 0 else 0.0
            cat_agreement = {
                cat: d["agree"] / d["total"] if d["total"] > 0 else 0.0
                for cat, d in by_category.items()
            }

            results[stage_name] = {
                "agreement": agreement,
                "n_turns": n_turns,
                "n_agree": n_agree,
                "by_category": cat_agreement,
                "config": config,
                "n_voters": n_voters,
                "temperature": temperature,
            }
            print(f"    {stage_name}: agreement={agreement:.1%} ({n_agree}/{n_turns})")

    finally:
        # Clean up temp file
        eval_path.unlink(missing_ok=True)
        # Clean up label check results
        label_check_dir = PROJECT_ROOT / "results" / output_dir
        if label_check_dir.exists():
            import shutil
            shutil.rmtree(label_check_dir, ignore_errors=True)

    return results


# ── Scorecards ───────────────────────────────────────────────────────

def compute_intrinsic_scorecard(intrinsic: dict) -> list[dict]:
    """Return scorecard rows for intrinsic (synth-only) quality."""
    rows = []

    def _add(signal, metric_label, value, key, higher_is_better=False):
        g, y = INTRINSIC_THRESHOLDS[key]
        if higher_is_better:
            rating = "green" if value >= g else ("yellow" if value >= y else "red")
        else:
            rating = "green" if value <= g else ("yellow" if value <= y else "red")
        rows.append(dict(signal=signal, metric=metric_label, value=value,
                         rating=rating, green_threshold=g, yellow_threshold=y,
                         higher_is_better=higher_is_better))

    fd = intrinsic.get("flow_diversity", {})
    if fd:
        u = fd.get("uniformity", 0)
        _add("Flow uniformity", f"ratio = {u:.2f}", u, "flow_entropy_ratio", True)

    td = intrinsic.get("tool_diversity", {})
    if td and td.get("n_unique", 0) > 1:
        te = td["entropy"]
        max_e = math.log2(td["n_unique"])
        ratio = te / max_e if max_e > 0 else 0
        _add("Tool uniformity", f"ratio = {ratio:.2f}", ratio, "tool_entropy_ratio", True)

    nat = intrinsic.get("naturalness", {})
    ss = nat.get("synth_summary", {})
    if ss.get("mean"):
        nm = ss["mean"]
        _add("Naturalness (synth)", f"mean = {nm:.2f}", nm, "naturalness_mean", True)

    la = intrinsic.get("label_agreement", {})
    for stage in ("intent", "flow", "tool"):
        if stage in la:
            a = la[stage]["agreement"]
            _add(f"{stage.title()} label quality",
                 f"agreement = {a:.2f}", a,
                 f"label_agreement_{stage}", True)

    return rows


def compute_comparative_scorecard(d: dict) -> list[dict]:
    """Return structured scorecard rows from comparative metrics."""
    rows = []

    def _add(signal, metric_label, value, key, higher_is_better=False):
        g, y = THRESHOLDS[key]
        rows.append(dict(signal=signal, metric=metric_label, value=value,
                         rating=_rating(value, key, higher_is_better),
                         green_threshold=g, yellow_threshold=y,
                         higher_is_better=higher_is_better))

    fj  = d["flow"]["jsd"]
    ij  = d["intent"]["jsd"]
    lks = max(d["length"]["ks_t1"]["stat"], d["length"]["ks_t3"]["stat"])
    vj  = d["vocab"]["jaccard"]
    tc  = d["tools"]["coverage_pct"]
    fc  = d["cooccurrence"]["coverage_pct"]

    _add("Flow match",      f"JSD = {fj:.3f}",  fj,  "flow_jsd")
    _add("Intent match",    f"JSD = {ij:.3f}",  ij,  "intent_jsd")
    _add("Length match",    f"KS = {lks:.3f}",  lks, "length_ks")
    _add("Vocab overlap",   f"Jaccard = {vj:.3f}", vj, "vocab_jaccard", True)
    _add("Tool coverage",   f"{tc:.1%}",         tc,  "tool_coverage", True)
    _add("Flow pair coverage", f"{fc:.1%}",      fc,  "flow_pair_coverage", True)

    if "turn_position" in d:
        t3fj = d["turn_position"]["turn_3"]["flow_jsd"]
        _add("Turn-3 flow match", f"JSD = {t3fj:.3f}", t3fj, "flow_jsd")

    if "per_category" in d:
        cats = d["per_category"].get("categories", {})
        for cat in ("ambiguous_first", "ambiguous_second"):
            if cat in cats:
                cfj = cats[cat]["flow_jsd"]
                _add(f"{cat} flow", f"JSD = {cfj:.3f}", cfj, "flow_jsd")

    if "naturalness" in d:
        ng = d["naturalness"]["gap"]
        _add("Naturalness gap", f"|Δ| = {ng:.2f}", ng, "naturalness_gap")

    if "ambiguity" in d:
        ag = d["ambiguity"]["gap"]
        _add("Ambiguity quality gap", f"|Δ| = {ag:.2f}", ag, "ambiguity_gap")

    return rows


# ── JSON Serialization ───────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compute synth-vs-eval metrics → JSON")
    ap.add_argument("--domain", choices=["dana", "hugo", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-llm", action="store_true", help="Skip LLM judges")
    ap.add_argument("--concurrency", type=int, default=10, help="LLM concurrency")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--check-labels", action="store_true",
                    help="Run ensemble label agreement check (requires LLM API)")
    ap.add_argument("--label-check-flow-config", default="1a_004",
                    help="exp1a config for flow label check")
    ap.add_argument("--label-check-intent-config", default="2_004",
                    help="exp2 config for intent label check")
    ap.add_argument("--label-check-tool-config", default="2_004",
                    help="exp2 config for tool label check")
    ap.add_argument("--label-check-voters", type=int, default=3,
                    help="Number of ensemble voters")
    ap.add_argument("--label-check-temp", type=float, default=0.3,
                    help="Temperature for ensemble voting")
    args = ap.parse_args()

    np.random.seed(args.seed)
    domains = DOMAINS if args.domain == "both" else [args.domain]
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Model effects needs all synth data across domains
    all_synth_recs: list[dict] = []

    for dm in domains:
        print(f"\n{'=' * 60}")
        print(f"  Computing metrics for {dm.upper()}")
        print(f"{'=' * 60}")

        eval_c, synth_c = load_datasets(dm)
        eval_r  = flatten_turns(eval_c,  "eval")
        synth_r = flatten_turns(synth_c, "synth")
        print(f"  Eval:  {len(eval_c)} convos, {len(eval_r)} user turns")
        print(f"  Synth: {len(synth_c)} convos, {len(synth_r)} user turns")

        d: dict[str, Any] = {
            "domain": dm,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "counts": {
                "eval_convos": len(eval_c), "synth_convos": len(synth_c),
                "eval_turns": len(eval_r), "synth_turns": len(synth_r),
            },
        }

        intrinsic: dict[str, Any] = {}
        comparative: dict[str, Any] = {}

        print("  [ 1/16] Flow distribution...")
        comparative["flow"] = flow_distribution(eval_r, synth_r, dm)
        intrinsic["flow_diversity"] = synth_flow_diversity(synth_r)

        print("  [ 2/16] Intent distribution...")
        comparative["intent"] = intent_distribution(eval_r, synth_r, dm)
        intrinsic["intent_diversity"] = synth_intent_diversity(synth_r)

        print("  [ 3/16] Category balance...")
        cat_result = category_distribution(eval_r, synth_r, dm)
        comparative["category"] = cat_result
        intrinsic["category"] = {"synth_categories": cat_result["synth_categories"]}

        print("  [ 4/16] Utterance length...")
        length_result = utterance_length(eval_r, synth_r, dm)
        comparative["length"] = length_result
        intrinsic["length"] = {
            "synth_t1_stats": length_result["synth_t1_stats"],
            "synth_t3_stats": length_result["synth_t3_stats"],
            "synth_t1_lengths": length_result["synth_t1_lengths"],
            "synth_t3_lengths": length_result["synth_t3_lengths"],
            "category_lengths": {c: {"synth": v["synth"]}
                                 for c, v in length_result.get("category_lengths", {}).items()},
        }

        print("  [ 5/16] Vocabulary...")
        vocab_result = vocabulary_analysis(eval_r, synth_r, dm)
        comparative["vocab"] = vocab_result
        intrinsic["vocab"] = {
            "synth_ttr": vocab_result["synth_ttr"],
            "synth_vocab_size": vocab_result["synth_vocab_size"],
        }

        print("  [ 6/16] Tool usage...")
        comparative["tools"] = tool_usage(eval_r, synth_r, dm)
        intrinsic["tool_diversity"] = synth_tool_diversity(synth_r)

        print("  [ 7/16] Flow co-occurrence...")
        comparative["cooccurrence"] = flow_cooccurrence(eval_c, synth_c, dm)

        print("  [ 8/16] Embedding similarity...")
        emb_result = embedding_similarity(eval_r, synth_r, dm, seed=args.seed)
        comparative["embedding"] = emb_result
        intrinsic["embedding_diversity"] = {
            "within_synth": emb_result["within_synth"],
            "synth_pca_coords": [c for c in emb_result.get("pca_coords", [])
                                 if c["source"] == "synth"],
        }

        print("  [ 9/16] Parameter completeness...")
        params_result = parameter_completeness(eval_r, synth_r, dm)
        comparative["params"] = params_result
        intrinsic["params"] = {
            "synth_null_rate": params_result["synth_null_rate"],
            "synth_total_params": params_result["synth_total_params"],
            "synth_null_params": params_result["synth_null_params"],
        }

        print("  [10/16] Context dependence...")
        cd_result = context_dependence(eval_r, synth_r, dm)
        comparative["context_dep"] = cd_result
        intrinsic["context_dep"] = {
            "synth_terse_rate": cd_result["synth_terse_rate"],
            "synth_terse": cd_result["synth_terse"],
            "synth_total": cd_result["synth_total"],
        }

        print("  [11/16] Per-category metrics...")
        comparative["per_category"] = per_category_metrics(eval_r, synth_r, dm)

        print("  [12/16] Turn position analysis...")
        comparative["turn_position"] = turn_position_analysis(eval_r, synth_r, dm)

        print("  [13/16] Agent response analysis...")
        ar_result = agent_response_analysis(eval_c, synth_c, dm)
        comparative["agent_response"] = ar_result
        intrinsic["agent_response"] = {
            "synth_mean_length": ar_result["synth_mean_length"],
            "synth_lengths": ar_result["synth_lengths"],
        }

        print("  [14/16] Conditional distributions...")
        comparative["conditional"] = conditional_distributions(eval_r, synth_r, dm)

        print("  [15/16] Scenario topic coverage...")
        tc_result = scenario_topic_coverage(eval_c, synth_c, dm, seed=args.seed)
        comparative["topic_coverage"] = tc_result
        intrinsic["topic_coverage"] = {
            "n_clusters": tc_result["n_clusters"],
            "synth_only_clusters": tc_result.get("synth_only_clusters", 0),
            "synth_pca_coords": [c for c in tc_result.get("pca_coords", [])
                                 if c["source"] == "synth"],
        }

        # Model effects (per-domain)
        intrinsic["model_effects"] = model_effects(synth_r)

        if not args.skip_llm:
            print("  [16a] Naturalness — full corpus (LLM)...")
            nat_result = asyncio.run(
                judge_naturalness_all(eval_c, synth_c, dm,
                                      seed=args.seed, concurrency=args.concurrency))
            print(f"         Scored {nat_result['synth_summary']['n']} synth + "
                  f"{nat_result['eval_summary']['n']} eval conversations")
            print(f"         Contrived (score<=2): {len(nat_result['contrived_ids'])}")
            comparative["naturalness"] = {
                "gap": nat_result["gap"],
                "t_stat": nat_result["t_stat"],
                "t_pval": nat_result["t_pval"],
                "eval_summary": nat_result["eval_summary"],
            }
            intrinsic["naturalness"] = {
                "per_conversation": nat_result["per_conversation"],
                "synth_summary": nat_result["synth_summary"],
                "contrived_ids": nat_result["contrived_ids"],
            }

            print("  [16b] Ambiguity quality (LLM)...")
            comparative["ambiguity"] = asyncio.run(
                ambiguity_quality_analysis(eval_c, synth_c, dm,
                                           seed=args.seed, concurrency=args.concurrency))

            print("  [16c] Scenario diversity (LLM)...")
            div_result = asyncio.run(
                scenario_diversity_analysis(eval_c, synth_c, dm, concurrency=args.concurrency))
            comparative["diversity"] = div_result
            intrinsic["diversity"] = {"synth": div_result["synth"]}
        else:
            print("  [16/16] Skipping LLM judges (--skip-llm)")

        if args.check_labels:
            print("  [17/17] Label agreement (ensemble)...")
            intrinsic["label_agreement"] = run_label_agreement(
                synth_c, dm, args.seed,
                configs={
                    "flow": args.label_check_flow_config,
                    "intent": args.label_check_intent_config,
                    "tool": args.label_check_tool_config,
                },
                n_voters=args.label_check_voters,
                temperature=args.label_check_temp,
            )

        d["intrinsic"] = intrinsic
        d["comparative"] = comparative

        # Scorecards
        d["intrinsic_scorecard"] = compute_intrinsic_scorecard(intrinsic)
        d["comparative_scorecard"] = compute_comparative_scorecard(comparative)

        # Write JSON
        out_path = ANALYSIS_DIR / f"metrics_{dm}.json"
        with open(out_path, "w") as f:
            json.dump(d, f, indent=2, cls=_NumpyEncoder)
        size_kb = out_path.stat().st_size / 1024
        print(f"\n  Wrote {out_path} ({size_kb:.0f} KB)")

        # Accumulate synth for model effects
        all_synth_recs.extend(synth_r)

    # Model effects (cross-domain)
    if all_synth_recs:
        print("\n  Computing model effects (cross-domain)...")
        me = model_effects(all_synth_recs)
        me_path = ANALYSIS_DIR / "metrics_model_effects.json"
        with open(me_path, "w") as f:
            json.dump(me, f, indent=2, cls=_NumpyEncoder)
        print(f"  Wrote {me_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
