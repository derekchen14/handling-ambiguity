#!/usr/bin/env python3
"""Analyze synthetic vs eval dataset distribution.

Produces quantitative plots + stats and (optionally) qualitative LLM-based
ratings, then assembles a concise markdown report.

Usage:
    python datasets/data_aug_pranav/analyze_synth_vs_eval.py \
        --domain both --seed 42 --skip-llm
"""

import argparse
import asyncio
import json
import math
import os
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare, ks_2samp

# ── Constants ────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent          # analysis/
AUG_DIR    = SCRIPT_DIR.parent                        # data_aug_pranav/
DATA_DIR   = AUG_DIR.parent                           # datasets/
DOMAINS = ["dana", "hugo"]
CATEGORIES = ["same_flow", "switch_flow", "ambiguous_first", "ambiguous_second"]

EVAL_COLOR = "#2196F3"
SYNTH_COLOR = "#FF9800"

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

# ── Data Loading ─────────────────────────────────────────────────────

def load_datasets(domain: str):
    """Return (eval_convos, synth_convos) lists of dicts."""
    eval_path  = DATA_DIR / domain / "eval_set.json"
    synth_path = AUG_DIR / f"conversations_{domain}.json"
    with open(eval_path)  as f: eval_convos  = json.load(f)
    with open(synth_path) as f: synth_convos = json.load(f)
    return eval_convos, synth_convos


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
                "target_tools": t.get("target_tools", {}),
                "turn_num":     t["turn_num"],
                "source":       source,
                "_model":       c.get("_model",    "human"),
                "_provider":    c.get("_provider", "human"),
            }
            if "candidate_flows" in t:
                rec["candidate_flows"] = t["candidate_flows"]
            records.append(rec)
    return records

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
    # Keep only bins where expected > 0
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
        return "🟢" if value >= g else ("🟡" if value >= y else "🔴")
    return "🟢" if value <= g else ("🟡" if value <= y else "🔴")


# ── 1. Flow Distribution ────────────────────────────────────────────

def flow_distribution(eval_recs, synth_recs, domain, out):
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

    # horizontal paired bar chart sorted by eval freq
    flows_sorted = sorted(all_flows, key=lambda f: ef.get(f, 0), reverse=True)
    fig, ax = plt.subplots(figsize=(10, max(8, len(flows_sorted) * 0.35)))
    y = np.arange(len(flows_sorted))
    h = 0.35
    ev = [ef.get(f, 0) / et * 100 for f in flows_sorted]
    sv = [sf.get(f, 0) / st * 100 for f in flows_sorted]
    ax.barh(y + h / 2, ev, h, label="Eval",  color=EVAL_COLOR,  alpha=0.8)
    ax.barh(y - h / 2, sv, h, label="Synth", color=SYNTH_COLOR, alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(flows_sorted, fontsize=8)
    ax.set_xlabel("Percentage (%)"); ax.set_title(f"Flow Distribution — {domain.title()} (JSD={jsd:.3f})")
    ax.legend(); ax.invert_yaxis(); plt.tight_layout()
    fname = f"flow_distribution_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)

    return dict(jsd=jsd, chi_stat=chi_stat, chi_pval=chi_pval,
                eval_flows=dict(ef), synth_flows=dict(sf),
                flagged_ratios=flagged, plot=fname)


# ── 2. Intent Distribution ──────────────────────────────────────────

def intent_distribution(eval_recs, synth_recs, domain, out):
    ei = Counter(r["intent"] for r in eval_recs  if r.get("intent"))
    si = Counter(r["intent"] for r in synth_recs if r.get("intent"))
    jsd = compute_jsd(ei, si)
    chi_stat, chi_pval = compute_chisq(ei, si)

    intents = sorted(set(ei) | set(si))
    et, st = sum(ei.values()), sum(si.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(intents)); w = 0.35
    ax.bar(x - w / 2, [ei.get(i, 0) / et * 100 for i in intents], w, label="Eval",  color=EVAL_COLOR,  alpha=0.8)
    ax.bar(x + w / 2, [si.get(i, 0) / st * 100 for i in intents], w, label="Synth", color=SYNTH_COLOR, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(intents)
    ax.set_ylabel("Percentage (%)"); ax.set_title(f"Intent Distribution — {domain.title()} (JSD={jsd:.3f})")
    ax.legend(); plt.tight_layout()
    fname = f"intent_distribution_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)

    return dict(jsd=jsd, chi_stat=chi_stat, chi_pval=chi_pval,
                eval_intents=dict(ei), synth_intents=dict(si), plot=fname)


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

def utterance_length(eval_recs, synth_recs, domain, out):
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    mx = max(max(et1 + st1, default=0), max(et3 + st3, default=0))
    bins = np.linspace(0, min(mx, 80), 40)

    axes[0, 0].hist(et1, bins, alpha=.6, color=EVAL_COLOR,  label="Eval",  density=True)
    axes[0, 0].hist(st1, bins, alpha=.6, color=SYNTH_COLOR, label="Synth", density=True)
    axes[0, 0].set_title(f"Turn 1 (KS={ks1s:.3f}, p={ks1p:.3f})"); axes[0, 0].set_xlabel("Words"); axes[0, 0].legend()

    axes[0, 1].hist(et3, bins, alpha=.6, color=EVAL_COLOR,  label="Eval",  density=True)
    axes[0, 1].hist(st3, bins, alpha=.6, color=SYNTH_COLOR, label="Synth", density=True)
    axes[0, 1].set_title(f"Turn 3 (KS={ks3s:.3f}, p={ks3p:.3f})"); axes[0, 1].set_xlabel("Words"); axes[0, 1].legend()

    cats = sorted({r["category"] for r in eval_recs})
    bd, bl, bc = [], [], []
    for c in cats:
        bd.append([word_count(r["utterance"]) for r in eval_recs  if r["category"] == c])
        bd.append([word_count(r["utterance"]) for r in synth_recs if r["category"] == c])
        bl += [f"{c}\n(eval)", f"{c}\n(synth)"]
        bc += [EVAL_COLOR, SYNTH_COLOR]
    bp = axes[1, 0].boxplot(bd, tick_labels=bl, patch_artist=True)
    for patch, col in zip(bp["boxes"], bc):
        patch.set_facecolor(col); patch.set_alpha(.6)
    axes[1, 0].set_title("Length by Category × Source"); axes[1, 0].set_ylabel("Words")
    axes[1, 0].tick_params(axis="x", rotation=45, labelsize=7)
    axes[1, 1].axis("off")
    plt.suptitle(f"Utterance Length — {domain.title()}", fontsize=14); plt.tight_layout()
    fname = f"utterance_length_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)

    return dict(ks_t1=dict(stat=ks1s, pval=ks1p), ks_t3=dict(stat=ks3s, pval=ks3p),
                eval_t1_stats=_stats(et1), eval_t3_stats=_stats(et3),
                synth_t1_stats=_stats(st1), synth_t3_stats=_stats(st3), plot=fname)


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

def tool_usage(eval_recs, synth_recs, domain, out):
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
    gaps  = es - ss          # eval-only (critical)
    noise = ss - es          # synth-only
    cov   = len(es & ss) / len(es) if es else 1.0

    etsizes = [len(r.get("target_tools", {})) for r in eval_recs]
    stsizes = [len(r.get("target_tools", {})) for r in synth_recs]

    # dot plot (top 30 by eval freq)
    all_t = sorted(set(ec) | set(sc), key=lambda t: ec.get(t, 0), reverse=True)[:30]
    et_tot, st_tot = sum(ec.values()), sum(sc.values())

    fig, ax = plt.subplots(figsize=(10, max(8, len(all_t) * 0.3)))
    y = np.arange(len(all_t))
    ep = [ec.get(t, 0) / et_tot * 100 if et_tot else 0 for t in all_t]
    sp = [sc.get(t, 0) / st_tot * 100 if st_tot else 0 for t in all_t]
    ax.scatter(ep, y, color=EVAL_COLOR,  label="Eval",  s=60, zorder=3)
    ax.scatter(sp, y, color=SYNTH_COLOR, label="Synth", s=60, marker="D", zorder=3)
    for i in range(len(all_t)):
        ax.plot([ep[i], sp[i]], [y[i], y[i]], color="gray", alpha=.3, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(all_t, fontsize=7)
    ax.set_xlabel("Percentage (%)"); ax.set_title(f"Tool Usage — {domain.title()} (JSD={jsd:.3f})")
    ax.legend(); ax.invert_yaxis(); plt.tight_layout()
    fname = f"tool_usage_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)

    return dict(jsd=jsd, chi_stat=chi_stat, chi_pval=chi_pval,
                coverage_pct=cov, coverage_gaps=sorted(gaps), noise_tools=sorted(noise),
                eval_mean_tools=float(np.mean(etsizes)), synth_mean_tools=float(np.mean(stsizes)),
                plot=fname)


# ── 7. Flow Co-occurrence ────────────────────────────────────────────

def flow_cooccurrence(eval_convos, synth_convos, domain, out):
    def _trans(convos):
        tr = Counter()
        for c in convos:
            ut = [t for t in c["turns"] if t["speaker"] == "user"]
            if len(ut) >= 2:
                tr[(ut[0].get("flow", "?"), ut[1].get("flow", "?"))] += 1
        return tr

    et, st = _trans(eval_convos), _trans(synth_convos)
    ep, sp = set(et), set(st)
    cov = len(ep & sp) / len(ep) if ep else 1.0
    missing = ep - sp

    # cosine similarity
    all_p = sorted(set(et) | set(st))
    ev = np.array([et.get(p, 0) for p in all_p], dtype=float)
    sv = np.array([st.get(p, 0) for p in all_p], dtype=float)
    ne, ns = np.linalg.norm(ev), np.linalg.norm(sv)
    cos = float(np.dot(ev, sv) / (ne * ns)) if ne and ns else 0.0

    # heatmaps of top-15 flows
    ff = Counter()
    for (f1, f3), n in list(et.items()) + list(st.items()):
        ff[f1] += n; ff[f3] += n
    top = [f for f, _ in ff.most_common(15)]

    def _mat(tr, flows):
        idx = {f: i for i, f in enumerate(flows)}
        m = np.zeros((len(flows), len(flows)))
        for (f1, f3), n in tr.items():
            if f1 in idx and f3 in idx:
                m[idx[f1]][idx[f3]] += n
        rs = m.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        return m / rs

    em, sm = _mat(et, top), _mat(st, top)
    vmax = max(em.max(), sm.max(), 1e-9)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7))
    for ax, mat, lbl in [(a1, em, "Eval"), (a2, sm, "Synth")]:
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(top))); ax.set_yticks(range(len(top)))
        ax.set_xticklabels(top, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(top, fontsize=7)
        ax.set_title(f"{lbl} (Turn 1 → Turn 3)")
        ax.set_xlabel("Turn 3 flow"); ax.set_ylabel("Turn 1 flow")
        plt.colorbar(im, ax=ax, shrink=0.7)
    plt.suptitle(f"Flow Co-occurrence — {domain.title()} (cosine={cos:.3f})", fontsize=14)
    plt.tight_layout()
    fname = f"flow_cooccurrence_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)

    return dict(cosine_sim=cos, coverage_pct=cov,
                missing_pairs=[(a, b) for a, b in sorted(missing)],
                num_eval_pairs=len(ep), num_synth_pairs=len(sp), plot=fname)


# ── 8. Embedding Similarity (TF-IDF, no sklearn) ────────────────────

def embedding_similarity(eval_recs, synth_recs, domain, out, seed=42):
    all_recs = eval_recs + synth_recs
    texts    = [r["utterance"] for r in all_recs]
    sources  = [r["source"]    for r in all_recs]
    cats     = [r["category"]  for r in all_recs]

    # build vocab & TF-IDF
    df = Counter()
    dt = []
    for t in texts:
        toks = tokenize(t)
        c = Counter(toks); dt.append(c)
        for w in set(toks): df[w] += 1

    vocab = [w for w, f in df.most_common(2000) if f >= 2]
    vi = {w: i for i, w in enumerate(vocab)}
    n_docs, n_v = len(texts), len(vocab)
    idf = np.array([math.log(n_docs / (1 + df[w])) for w in vocab])

    tfidf = np.zeros((n_docs, n_v))
    for i, c in enumerate(dt):
        tot = sum(c.values())
        if not tot: continue
        for w, cnt in c.items():
            if w in vi: tfidf[i, vi[w]] = (cnt / tot) * idf[vi[w]]

    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tfidf /= norms

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
    # For efficiency, only keep top-2 components via SVD
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    pca = U[:, :2] * S[:2]

    fig, ax = plt.subplots(figsize=(10, 8))
    markers = {"same_flow": "o", "switch_flow": "s",
               "ambiguous_first": "^", "ambiguous_second": "D"}
    for src, col, al in [("eval", EVAL_COLOR, .6), ("synth", SYNTH_COLOR, .4)]:
        mk = np.array([s == src for s in sources])
        for cat, m in markers.items():
            cm = np.array([c == cat for c in cats]) & mk
            if cm.any():
                ax.scatter(pca[cm, 0], pca[cm, 1], c=col, marker=m, alpha=al,
                           s=20, label=f"{src}/{cat}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"TF-IDF PCA — {domain.title()} (cross-sim={cross:.3f})")
    ax.legend(fontsize=7, loc="upper right", ncol=2); plt.tight_layout()
    fname = f"embedding_pca_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)

    return dict(within_eval=within_e, within_synth=within_s, cross_set=cross,
                well_mixed=cross > min(within_e, within_s), plot=fname)


# ── 9. Model Effects ────────────────────────────────────────────────

def model_effects(synth_recs, out):
    by = defaultdict(list)
    for r in synth_recs: by[r["_provider"]].append(r)

    stats = {}
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

    provs = sorted(by.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [[word_count(r["utterance"]) for r in by[p]] for p in provs]
    bp = ax.boxplot(data, tick_labels=provs, patch_artist=True)
    cols = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cols[i % len(cols)]); patch.set_alpha(.6)
    ax.set_ylabel("Word count"); ax.set_title("Utterance Length by Provider (Synthetic)")
    plt.tight_layout()
    fname = "model_effects.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)
    return dict(model_stats=stats, plot=fname)


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


# ── Qualitative Analyses (LLM-gated) ────────────────────────────────

NATURALNESS_PROMPT = """Rate the following 3-turn conversation on naturalness (1-5 scale).

1 = Completely unnatural, robotic
2 = Mostly unnatural with occasional natural phrasing
3 = Mixed — some natural, some stilted
4 = Mostly natural with minor issues
5 = Completely natural, indistinguishable from real user

Domain context: {domain}

Conversation:
Turn 1 (user): {turn1}
Turn 2 (agent): {turn2}
Turn 3 (user): {turn3}

Respond in JSON only: {{"score": <1-5>, "reason": "<brief explanation>"}}"""

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


async def naturalness_analysis(eval_convos, synth_convos, domain, seed=42):
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    sem = asyncio.Semaphore(10)
    rng = np.random.RandomState(seed)

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
        p = NATURALNESS_PROMPT.format(domain=domain, turn1=ts[0]["utterance"],
                                      turn2=ts[1]["utterance"], turn3=ts[2]["utterance"])
        r = await _call_llm(p, client, sem)
        r["source"] = src; r["category"] = c["category"]
        if "_provider" in c: r["model"] = c["_provider"]
        return r

    tasks = [_rate(c, "eval") for c in _sample(eval_convos)]
    tasks += [_rate(c, "synth") for c in _sample(synth_convos)]
    results = await asyncio.gather(*tasks)
    await client.close()

    es = [r["score"] for r in results if r["source"] == "eval"  and r.get("score", -1) > 0]
    ss = [r["score"] for r in results if r["source"] == "synth" and r.get("score", -1) > 0]

    from scipy.stats import ttest_ind
    t, p = ttest_ind(es, ss, equal_var=False) if es and ss else (0.0, 1.0)

    by_m = defaultdict(list)
    for r in results:
        if r["source"] == "synth" and r.get("score", -1) > 0 and "model" in r:
            by_m[r["model"]].append(r["score"])
    mb = {m: dict(mean=float(np.mean(s)), std=float(np.std(s)), n=len(s)) for m, s in by_m.items()}

    return dict(eval_mean=float(np.mean(es)) if es else 0, eval_std=float(np.std(es)) if es else 0,
                synth_mean=float(np.mean(ss)) if ss else 0, synth_std=float(np.std(ss)) if ss else 0,
                gap=abs(float(np.mean(es)) - float(np.mean(ss))) if es and ss else 0,
                t_stat=float(t), t_pval=float(p), model_breakdown=mb,
                n_eval=len(es), n_synth=len(ss))


async def ambiguity_quality_analysis(eval_convos, synth_convos, domain, seed=42):
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    sem = asyncio.Semaphore(10)
    rng = np.random.RandomState(seed)

    def _amb_turns(convos, src):
        turns = []
        for c in convos:
            for t in c["turns"]:
                if t["speaker"] != "user":
                    continue
                # ambiguous_first: turn 1 has flow="ambiguous"
                if t.get("flow") == "ambiguous":
                    turns.append(dict(utterance=t["utterance"],
                                      candidate_flows=t.get("candidate_flows", []),
                                      category=c["category"], source=src,
                                      model=c.get("_provider", "human")))
                # ambiguous_second: turn 3 is the multi-request
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


async def scenario_diversity_analysis(eval_convos, synth_convos, domain):
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    sem = asyncio.Semaphore(5)

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


# ── Report ───────────────────────────────────────────────────────────

def _scorecard(results):
    header = ("| Signal | Metric | Rating | Green / Yellow | Red |\n"
              "|--------|--------|--------|----------------|-----|")
    rows = []
    for domain in results["domains"]:
        d = results[domain]
        fj  = d["flow"]["jsd"]
        ij  = d["intent"]["jsd"]
        lks = max(d["length"]["ks_t1"]["stat"], d["length"]["ks_t3"]["stat"])
        vj  = d["vocab"]["jaccard"]
        tc  = d["tools"]["coverage_pct"]
        fc  = d["cooccurrence"]["coverage_pct"]

        rows.append(f"| **{domain.title()}** | | | | |")
        rows.append(f"| Flow match | JSD = {fj:.3f} | {_rating(fj, 'flow_jsd')} | < 0.05 / < 0.15 | >= 0.15 |")
        rows.append(f"| Intent match | JSD = {ij:.3f} | {_rating(ij, 'intent_jsd')} | < 0.05 / < 0.15 | >= 0.15 |")
        rows.append(f"| Length match | KS = {lks:.3f} | {_rating(lks, 'length_ks')} | < 0.1 / < 0.3 | >= 0.3 |")
        rows.append(f"| Vocab overlap | Jaccard = {vj:.3f} | {_rating(vj, 'vocab_jaccard', True)} | > 0.6 / > 0.3 | <= 0.3 |")
        rows.append(f"| Tool coverage | {tc:.1%} | {_rating(tc, 'tool_coverage', True)} | > 95% / > 80% | <= 80% |")
        rows.append(f"| Flow pair coverage | {fc:.1%} | {_rating(fc, 'flow_pair_coverage', True)} | > 80% / > 60% | <= 60% |")
        if "naturalness" in d:
            ng = d["naturalness"]["gap"]
            rows.append(f"| Naturalness gap | |Δ| = {ng:.2f} | {_rating(ng, 'naturalness_gap')} | < 0.3 / < 0.7 | >= 0.7 |")
        if "ambiguity" in d:
            ag = d["ambiguity"]["gap"]
            rows.append(f"| Ambiguity quality gap | |Δ| = {ag:.2f} | {_rating(ag, 'ambiguity_gap')} | < 0.5 / < 1.0 | >= 1.0 |")
    return header + "\n" + "\n".join(rows)


def write_report(results, out):
    L = []
    domains = results["domains"]

    L.append("# Synthetic vs Eval: Distribution Analysis\n")

    # Executive Summary
    L.append("## Executive Summary\n")
    for dm in domains:
        d = results[dm]
        L.append(
            f"**{dm.title()}**: Flow JSD={d['flow']['jsd']:.3f}, "
            f"Intent JSD={d['intent']['jsd']:.3f}, "
            f"Tool coverage={d['tools']['coverage_pct']:.1%}. "
            f"Vocab Jaccard={d['vocab']['jaccard']:.3f}. "
            f"{len(d['tools']['coverage_gaps'])} eval tools missing from synth.\n")
    L.append("")

    # Scorecard
    L.append("## Transfer Risk Scorecard\n")
    L.append(_scorecard(results))
    L.append("")

    # ── Section 1: Distribution Analyses ──
    L.append("## 1. Distribution Analyses\n")
    for dm in domains:
        d = results[dm]
        L.append(f"### {dm.title()}\n")

        # Flow
        L.append("#### Flow Distribution\n")
        L.append(f"JSD = {d['flow']['jsd']:.4f}, "
                 f"χ² = {d['flow']['chi_stat']:.1f} (p = {d['flow']['chi_pval']:.3e})\n")
        if d["flow"]["flagged_ratios"]:
            L.append("**Flagged flows** (ratio < 0.5 or > 2.0):\n")
            for f, r in d["flow"]["flagged_ratios"]:
                L.append(f"- `{f}`: {r:.2f}x")
            L.append("")
        L.append(f"![Flow Distribution]({d['flow']['plot']})\n")

        # Intent
        L.append("#### Intent Distribution\n")
        L.append(f"JSD = {d['intent']['jsd']:.4f}, "
                 f"χ² = {d['intent']['chi_stat']:.1f} (p = {d['intent']['chi_pval']:.3e})\n")
        intents = sorted(set(d["intent"]["eval_intents"]) | set(d["intent"]["synth_intents"]))
        L.append("| Intent | Eval | Synth |")
        L.append("|--------|------|-------|")
        for i in intents:
            L.append(f"| {i} | {d['intent']['eval_intents'].get(i, 0)} | "
                     f"{d['intent']['synth_intents'].get(i, 0)} |")
        L.append("")
        L.append(f"![Intent Distribution]({d['intent']['plot']})\n")

        # Category
        L.append("#### Category Balance\n")
        L.append("| Category | Eval | Synth |")
        L.append("|----------|------|-------|")
        for c in CATEGORIES:
            L.append(f"| {c} | {d['category']['eval_categories'].get(c, 0)} | "
                     f"{d['category']['synth_categories'].get(c, 0)} |")
        L.append("")

        # Length
        L.append("#### Utterance Length\n")
        L.append(f"Turn 1 KS = {d['length']['ks_t1']['stat']:.3f} "
                 f"(p = {d['length']['ks_t1']['pval']:.3e}), "
                 f"Turn 3 KS = {d['length']['ks_t3']['stat']:.3f} "
                 f"(p = {d['length']['ks_t3']['pval']:.3e})\n")
        L.append("| Stat | Eval T1 | Synth T1 | Eval T3 | Synth T3 |")
        L.append("|------|---------|----------|---------|----------|")
        for s in ("mean", "median", "std", "p10", "p90"):
            L.append(f"| {s} | {d['length']['eval_t1_stats'][s]:.1f} | "
                     f"{d['length']['synth_t1_stats'][s]:.1f} | "
                     f"{d['length']['eval_t3_stats'][s]:.1f} | "
                     f"{d['length']['synth_t3_stats'][s]:.1f} |")
        L.append("")
        L.append(f"![Utterance Length]({d['length']['plot']})\n")

        # Vocab
        L.append("#### Vocabulary\n")
        L.append(f"Eval TTR = {d['vocab']['eval_ttr']:.3f} ({d['vocab']['eval_vocab_size']} types), "
                 f"Synth TTR = {d['vocab']['synth_ttr']:.3f} ({d['vocab']['synth_vocab_size']} types), "
                 f"Jaccard = {d['vocab']['jaccard']:.3f}\n")
        if d["vocab"]["eval_exclusive_words"]:
            L.append("**Eval-exclusive words** (freq >= 2, top 20):\n")
            words = list(d["vocab"]["eval_exclusive_words"].items())[:20]
            L.append(", ".join(f"`{w}` ({c})" for w, c in words))
            L.append("")

        # Tools
        L.append("#### Tool Usage\n")
        L.append(f"JSD = {d['tools']['jsd']:.4f}, Coverage = {d['tools']['coverage_pct']:.1%}\n")
        if d["tools"]["coverage_gaps"]:
            L.append(f"**Missing from synth** (critical): "
                     f"{', '.join(f'`{t}`' for t in d['tools']['coverage_gaps'])}\n")
        if d["tools"]["noise_tools"]:
            L.append(f"**Synth-only** (noise): "
                     f"{', '.join(f'`{t}`' for t in d['tools']['noise_tools'])}\n")
        L.append(f"Mean tools/turn: eval={d['tools']['eval_mean_tools']:.2f}, "
                 f"synth={d['tools']['synth_mean_tools']:.2f}\n")
        L.append(f"![Tool Usage]({d['tools']['plot']})\n")

        # Co-occurrence
        L.append("#### Flow Co-occurrence\n")
        L.append(f"Cosine similarity = {d['cooccurrence']['cosine_sim']:.3f}, "
                 f"Pair coverage = {d['cooccurrence']['coverage_pct']:.1%} "
                 f"({d['cooccurrence']['num_eval_pairs']} eval, "
                 f"{d['cooccurrence']['num_synth_pairs']} synth)\n")
        if d["cooccurrence"]["missing_pairs"]:
            mp = d["cooccurrence"]["missing_pairs"]
            L.append(f"**Missing transitions** ({len(mp)} pairs):")
            for a, b in mp[:10]:
                L.append(f"- `{a}` -> `{b}`")
            L.append("")
        L.append(f"![Flow Co-occurrence]({d['cooccurrence']['plot']})\n")

        # Embeddings
        L.append("#### Embedding Similarity\n")
        e = d["embedding"]
        L.append(f"Within-eval = {e['within_eval']:.3f}, "
                 f"Within-synth = {e['within_synth']:.3f}, "
                 f"Cross-set = {e['cross_set']:.3f} "
                 f"({'well-mixed' if e['well_mixed'] else 'NOT well-mixed'})\n")
        L.append(f"![Embedding PCA]({e['plot']})\n")

        # Param completeness
        L.append("#### Parameter Completeness\n")
        pc = d["params"]
        L.append(f"Eval null rate = {pc['eval_null_rate']:.1%} "
                 f"({pc['eval_null_params']}/{pc['eval_total_params']}), "
                 f"Synth null rate = {pc['synth_null_rate']:.1%} "
                 f"({pc['synth_null_params']}/{pc['synth_total_params']})\n")
        if pc["risk"] == "high":
            L.append("**Warning**: Synth has significantly fewer null params — "
                     "risk of hallucinated parameter values.\n")

        # Context dependence
        L.append("#### Context Dependence\n")
        cd = d["context_dep"]
        L.append(f"Terse turn-3 rate (< 8 words): eval={cd['eval_terse_rate']:.1%} "
                 f"({cd['eval_terse']}/{cd['eval_total']}), "
                 f"synth={cd['synth_terse_rate']:.1%} "
                 f"({cd['synth_terse']}/{cd['synth_total']})\n")

    # ── Section 2: Qualitative ──
    has_qual = any("naturalness" in results.get(dm, {}) for dm in domains)
    if has_qual:
        L.append("## 2. Qualitative Analyses\n")
        for dm in domains:
            d = results[dm]
            L.append(f"### {dm.title()}\n")

            if "naturalness" in d:
                n = d["naturalness"]
                L.append("#### Naturalness\n")
                L.append(f"Eval: {n['eval_mean']:.2f} +/- {n['eval_std']:.2f} (n={n['n_eval']}), "
                         f"Synth: {n['synth_mean']:.2f} +/- {n['synth_std']:.2f} (n={n['n_synth']})\n")
                L.append(f"Gap: {n['gap']:.2f}, Welch's t={n['t_stat']:.2f}, p={n['t_pval']:.3f}\n")
                if n.get("model_breakdown"):
                    L.append("| Provider | Mean | Std | n |")
                    L.append("|----------|------|-----|---|")
                    for m, s in sorted(n["model_breakdown"].items()):
                        L.append(f"| {m} | {s['mean']:.2f} | {s['std']:.2f} | {s['n']} |")
                    L.append("")

            if "ambiguity" in d:
                a = d["ambiguity"]
                L.append("#### Ambiguity Quality\n")
                L.append(f"Eval: {a['eval_mean']:.2f} +/- {a['eval_std']:.2f} (n={a['n_eval']}), "
                         f"Synth: {a['synth_mean']:.2f} +/- {a['synth_std']:.2f} (n={a['n_synth']})\n")
                L.append(f"Gap: {a['gap']:.2f}, Welch's t={a['t_stat']:.2f}, p={a['t_pval']:.3f}\n")

            if "diversity" in d:
                dv = d["diversity"]
                L.append("#### Scenario Diversity\n")
                L.append("| Dimension | Eval | Synth |")
                L.append("|-----------|------|-------|")
                for dim in ("topic", "task", "complexity"):
                    L.append(f"| {dim.title()} | {dv['eval'].get(dim, 0):.2f} | "
                             f"{dv['synth'].get(dim, 0):.2f} |")
                L.append("")

    # ── Section 3: Model Effects ──
    if "model_effects" in results:
        me = results["model_effects"]
        L.append("## 3. Model-Specific Effects\n")
        L.append("| Provider | n | Mean Length | Std Length | TTR |")
        L.append("|----------|---|------------|-----------|-----|")
        for m, s in sorted(me["model_stats"].items()):
            L.append(f"| {m} | {s['n']} | {s['mean_length']:.1f} | "
                     f"{s['std_length']:.1f} | {s['ttr']:.3f} |")
        L.append("")
        L.append(f"![Model Effects]({me['plot']})\n")

    # ── Section 4: Recommendations ──
    L.append("## 4. Recommendations\n")
    for dm in domains:
        d = results[dm]
        L.append(f"### {dm.title()}\n")
        risks = []
        if d["flow"]["jsd"] >= 0.15:
            risks.append("- **High flow JSD**: Significant flow distribution mismatch. "
                         "Consider rebalancing generation.")
        if d["tools"]["coverage_gaps"]:
            gaps = d["tools"]["coverage_gaps"]
            risks.append(f"- **Tool coverage gaps**: {len(gaps)} eval tools missing from synth: "
                         f"{', '.join(gaps[:5])}")
        if d["vocab"]["jaccard"] < 0.3:
            risks.append("- **Low vocabulary overlap**: Synthetic language diverges from eval. "
                         "Review generation prompts.")
        if d["params"]["risk"] == "high":
            risks.append("- **Parameter hallucination risk**: Synth has fewer null params. "
                         "Add null-param examples.")
        if d["context_dep"]["synth_terse_rate"] < d["context_dep"]["eval_terse_rate"] * 0.5:
            risks.append("- **Under-represented terse follow-ups**: Synth lacks short "
                         "context-dependent turn-3 utterances.")
        if d["cooccurrence"]["coverage_pct"] < 0.6:
            risks.append("- **Low flow-pair coverage**: Many eval transitions missing from synth.")
        L.append("\n".join(risks) if risks else
                 "No critical risks identified. Synthetic data appears well-aligned.")
        L.append("")

    path = out / "synth_vs_eval_report.md"
    path.write_text("\n".join(L))
    return path


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Analyze synth vs eval distribution")
    ap.add_argument("--domain", choices=["dana", "hugo", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-llm", action="store_true", help="Skip qualitative analyses")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    out = SCRIPT_DIR
    domains = DOMAINS if args.domain == "both" else [args.domain]
    results = {"domains": domains}

    for dm in domains:
        print(f"\n{'=' * 60}")
        print(f"  Analyzing {dm.upper()}")
        print(f"{'=' * 60}")

        eval_c, synth_c = load_datasets(dm)
        eval_r  = flatten_turns(eval_c,  "eval")
        synth_r = flatten_turns(synth_c, "synth")
        print(f"  Eval:  {len(eval_c)} convos, {len(eval_r)} user turns")
        print(f"  Synth: {len(synth_c)} convos, {len(synth_r)} user turns")

        d: dict[str, Any] = {}

        print("  [ 1/11] Flow distribution...")
        d["flow"] = flow_distribution(eval_r, synth_r, dm, out)

        print("  [ 2/11] Intent distribution...")
        d["intent"] = intent_distribution(eval_r, synth_r, dm, out)

        print("  [ 3/11] Category balance...")
        d["category"] = category_distribution(eval_r, synth_r, dm)

        print("  [ 4/11] Utterance length...")
        d["length"] = utterance_length(eval_r, synth_r, dm, out)

        print("  [ 5/11] Vocabulary...")
        d["vocab"] = vocabulary_analysis(eval_r, synth_r, dm)

        print("  [ 6/11] Tool usage...")
        d["tools"] = tool_usage(eval_r, synth_r, dm, out)

        print("  [ 7/11] Flow co-occurrence...")
        d["cooccurrence"] = flow_cooccurrence(eval_c, synth_c, dm, out)

        print("  [ 8/11] Embedding similarity...")
        d["embedding"] = embedding_similarity(eval_r, synth_r, dm, out, seed=args.seed)

        print("  [ 9/11] Parameter completeness...")
        d["params"] = parameter_completeness(eval_r, synth_r, dm)

        print("  [10/11] Context dependence...")
        d["context_dep"] = context_dependence(eval_r, synth_r, dm)

        if not args.skip_llm:
            print("  [11a] Naturalness (LLM)...")
            d["naturalness"] = asyncio.run(
                naturalness_analysis(eval_c, synth_c, dm, seed=args.seed))
            print("  [11b] Ambiguity quality (LLM)...")
            d["ambiguity"] = asyncio.run(
                ambiguity_quality_analysis(eval_c, synth_c, dm, seed=args.seed))
            print("  [11c] Scenario diversity (LLM)...")
            d["diversity"] = asyncio.run(
                scenario_diversity_analysis(eval_c, synth_c, dm))
        else:
            print("  [11/11] Skipping qualitative (--skip-llm)")

        results[dm] = d

    # Model effects across all synth data
    print("\n  Model effects...")
    all_synth = []
    for dm in domains:
        _, sc = load_datasets(dm)
        all_synth.extend(flatten_turns(sc, "synth"))
    results["model_effects"] = model_effects(all_synth, out)

    # Write report
    print("  Writing report...")
    rp = write_report(results, out)
    print(f"\n  Report: {rp}")
    pngs = sorted(out.glob("*.png"))
    print(f"  Generated {len(pngs)} plots:")
    for p in pngs:
        print(f"    {p.name}")


if __name__ == "__main__":
    main()
