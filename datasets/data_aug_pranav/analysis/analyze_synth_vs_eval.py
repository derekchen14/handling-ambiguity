#!/usr/bin/env python3
"""Generate plots and markdown report from pre-computed metrics JSON.

Reads ``metrics_{domain}.json`` (produced by ``compute_metrics.py``) and
generates all visualisations + the markdown report.

Usage:
    .venv/bin/python3 datasets/data_aug_pranav/analysis/analyze_synth_vs_eval.py \
        --domain both
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Constants ────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent          # analysis/
AUG_DIR    = SCRIPT_DIR.parent                        # data_aug_pranav/
DOMAINS = ["dana", "hugo"]
CATEGORIES = ["same_flow", "switch_flow", "ambiguous_first", "ambiguous_second"]

EVAL_COLOR = "#2196F3"
SYNTH_COLOR = "#FF9800"

RATING_EMOJI = {"green": "\U0001f7e2", "yellow": "\U0001f7e1", "red": "\U0001f534"}


# ── Metrics Loading ──────────────────────────────────────────────────

def load_metrics(domain: str, metrics_dir: Path | None = None) -> dict:
    """Load metrics_{domain}.json from the given directory."""
    d = metrics_dir or SCRIPT_DIR
    path = d / f"metrics_{domain}.json"
    with open(path) as f:
        return json.load(f)


def load_model_effects(metrics_dir: Path | None = None) -> dict | None:
    d = metrics_dir or SCRIPT_DIR
    path = d / "metrics_model_effects.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Plot Functions ───────────────────────────────────────────────────

def plot_flow_distribution(d: dict, domain: str, out: Path):
    flow = d["comparative"]["flow"]
    ef = Counter(flow["eval_flows"])
    sf = Counter(flow["synth_flows"])
    jsd = flow["jsd"]

    all_flows = sorted(set(ef) | set(sf))
    et, st = sum(ef.values()), sum(sf.values())
    flows_sorted = sorted(all_flows, key=lambda f: ef.get(f, 0), reverse=True)

    fig, ax = plt.subplots(figsize=(10, max(8, len(flows_sorted) * 0.35)))
    y = np.arange(len(flows_sorted))
    h = 0.35
    ev = [ef.get(f, 0) / et * 100 if et else 0 for f in flows_sorted]
    sv = [sf.get(f, 0) / st * 100 if st else 0 for f in flows_sorted]
    ax.barh(y + h / 2, ev, h, label="Eval",  color=EVAL_COLOR,  alpha=0.8)
    ax.barh(y - h / 2, sv, h, label="Synth", color=SYNTH_COLOR, alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(flows_sorted, fontsize=8)
    ax.set_xlabel("Percentage (%)"); ax.set_title(f"Flow Distribution \u2014 {domain.title()} (JSD={jsd:.3f})")
    ax.legend(); ax.invert_yaxis(); plt.tight_layout()
    fname = f"flow_distribution_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)
    return fname


def plot_intent_distribution(d: dict, domain: str, out: Path):
    intent = d["comparative"]["intent"]
    ei = Counter(intent["eval_intents"])
    si = Counter(intent["synth_intents"])
    jsd = intent["jsd"]

    intents = sorted(set(ei) | set(si))
    et, st = sum(ei.values()), sum(si.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(intents)); w = 0.35
    ax.bar(x - w / 2, [ei.get(i, 0) / et * 100 if et else 0 for i in intents], w, label="Eval",  color=EVAL_COLOR,  alpha=0.8)
    ax.bar(x + w / 2, [si.get(i, 0) / st * 100 if st else 0 for i in intents], w, label="Synth", color=SYNTH_COLOR, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(intents)
    ax.set_ylabel("Percentage (%)"); ax.set_title(f"Intent Distribution \u2014 {domain.title()} (JSD={jsd:.3f})")
    ax.legend(); plt.tight_layout()
    fname = f"intent_distribution_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)
    return fname


def plot_utterance_length(d: dict, domain: str, out: Path):
    length = d["comparative"]["length"]
    et1 = length["eval_t1_lengths"]
    et3 = length["eval_t3_lengths"]
    st1 = length["synth_t1_lengths"]
    st3 = length["synth_t3_lengths"]

    ks1s = length["ks_t1"]["stat"]
    ks1p = length["ks_t1"]["pval"]
    ks3s = length["ks_t3"]["stat"]
    ks3p = length["ks_t3"]["pval"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    mx = max(max(et1 + st1, default=0), max(et3 + st3, default=0))
    bins = np.linspace(0, min(mx, 80), 40)

    axes[0, 0].hist(et1, bins, alpha=.6, color=EVAL_COLOR,  label="Eval",  density=True)
    axes[0, 0].hist(st1, bins, alpha=.6, color=SYNTH_COLOR, label="Synth", density=True)
    axes[0, 0].set_title(f"Turn 1 (KS={ks1s:.3f}, p={ks1p:.3f})"); axes[0, 0].set_xlabel("Words"); axes[0, 0].legend()

    axes[0, 1].hist(et3, bins, alpha=.6, color=EVAL_COLOR,  label="Eval",  density=True)
    axes[0, 1].hist(st3, bins, alpha=.6, color=SYNTH_COLOR, label="Synth", density=True)
    axes[0, 1].set_title(f"Turn 3 (KS={ks3s:.3f}, p={ks3p:.3f})"); axes[0, 1].set_xlabel("Words"); axes[0, 1].legend()

    # Category boxplot
    cat_lengths = length.get("category_lengths", {})
    cats = sorted(cat_lengths.keys())
    bd, bl, bc = [], [], []
    for c in cats:
        bd.append(cat_lengths[c].get("eval", []))
        bd.append(cat_lengths[c].get("synth", []))
        bl += [f"{c}\n(eval)", f"{c}\n(synth)"]
        bc += [EVAL_COLOR, SYNTH_COLOR]
    if bd:
        bp = axes[1, 0].boxplot(bd, tick_labels=bl, patch_artist=True)
        for patch, col in zip(bp["boxes"], bc):
            patch.set_facecolor(col); patch.set_alpha(.6)
        axes[1, 0].set_title("Length by Category \u00d7 Source"); axes[1, 0].set_ylabel("Words")
        axes[1, 0].tick_params(axis="x", rotation=45, labelsize=7)
    axes[1, 1].axis("off")
    plt.suptitle(f"Utterance Length \u2014 {domain.title()}", fontsize=14); plt.tight_layout()
    fname = f"utterance_length_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)
    return fname


def plot_tool_usage(d: dict, domain: str, out: Path):
    tools = d["comparative"]["tools"]
    ec = Counter(tools["eval_tool_counts"])
    sc = Counter(tools["synth_tool_counts"])
    jsd = tools["jsd"]

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
    ax.set_xlabel("Percentage (%)"); ax.set_title(f"Tool Usage \u2014 {domain.title()} (JSD={jsd:.3f})")
    ax.legend(); ax.invert_yaxis(); plt.tight_layout()
    fname = f"tool_usage_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)
    return fname


def plot_flow_cooccurrence(d: dict, domain: str, out: Path):
    cooc = d["comparative"]["cooccurrence"]
    cos = cooc["cosine_sim"]
    top = cooc.get("top_flows", [])[:15]
    if not top:
        return ""

    # Parse transitions back to Counter of tuples
    def _parse_trans(trans_dict):
        tr = Counter()
        for key, n in trans_dict.items():
            parts = key.split("->")
            if len(parts) == 2:
                tr[(parts[0], parts[1])] += n
        return tr

    et = _parse_trans(cooc.get("eval_transitions", {}))
    st = _parse_trans(cooc.get("synth_transitions", {}))

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
        ax.set_title(f"{lbl} (Turn 1 \u2192 Turn 3)")
        ax.set_xlabel("Turn 3 flow"); ax.set_ylabel("Turn 1 flow")
        plt.colorbar(im, ax=ax, shrink=0.7)
    plt.suptitle(f"Flow Co-occurrence \u2014 {domain.title()} (cosine={cos:.3f})", fontsize=14)
    plt.tight_layout()
    fname = f"flow_cooccurrence_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)
    return fname


def plot_embedding_pca(d: dict, domain: str, out: Path):
    emb = d["comparative"]["embedding"]
    coords = emb.get("pca_coords", [])
    if not coords:
        return ""
    cross = emb["cross_set"]

    pca_x = np.array([c["x"] for c in coords])
    pca_y = np.array([c["y"] for c in coords])
    sources = [c["source"] for c in coords]
    cats = [c["category"] for c in coords]

    fig, ax = plt.subplots(figsize=(10, 8))
    markers = {"same_flow": "o", "switch_flow": "s",
               "ambiguous_first": "^", "ambiguous_second": "D"}
    for src, col, al in [("eval", EVAL_COLOR, .6), ("synth", SYNTH_COLOR, .4)]:
        mk = np.array([s == src for s in sources])
        for cat, m in markers.items():
            cm = np.array([c == cat for c in cats]) & mk
            if cm.any():
                ax.scatter(pca_x[cm], pca_y[cm], c=col, marker=m, alpha=al,
                           s=20, label=f"{src}/{cat}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"TF-IDF PCA \u2014 {domain.title()} (cross-sim={cross:.3f})")
    ax.legend(fontsize=7, loc="upper right", ncol=2); plt.tight_layout()
    fname = f"embedding_pca_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)
    return fname


def plot_model_effects(me: dict, out: Path):
    provider_lengths = me.get("provider_lengths", {})
    if not provider_lengths:
        return ""

    provs = sorted(provider_lengths.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [provider_lengths[p] for p in provs]
    bp = ax.boxplot(data, tick_labels=provs, patch_artist=True)
    cols = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cols[i % len(cols)]); patch.set_alpha(.6)
    ax.set_ylabel("Word count"); ax.set_title("Utterance Length by Provider (Synthetic)")
    plt.tight_layout()
    fname = "model_effects.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)
    return fname


def plot_per_category_metrics(d: dict, domain: str, out: Path):
    cats_data = d["comparative"]["per_category"].get("categories", {})
    cats = sorted(cats_data.keys())
    if not cats:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cats))
    w = 0.25
    flow_vals = [cats_data[c]["flow_jsd"] for c in cats]
    tool_vals = [cats_data[c]["tool_jsd"] for c in cats]
    len_vals  = [cats_data[c]["length_ks"] for c in cats]
    ax.bar(x - w, flow_vals, w, label="Flow JSD", color="#2196F3", alpha=0.8)
    ax.bar(x,     tool_vals, w, label="Tool JSD", color="#FF9800", alpha=0.8)
    ax.bar(x + w, len_vals,  w, label="Length KS", color="#4CAF50", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8, rotation=15)
    ax.set_ylabel("Metric Value"); ax.set_title(f"Per-Category Metrics \u2014 {domain.title()}")
    ax.legend(); plt.tight_layout()
    fname = f"per_category_metrics_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)
    return fname


def plot_turn_position(d: dict, domain: str, out: Path):
    tp = d["comparative"]["turn_position"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, tn in enumerate([1, 3]):
        key = f"turn_{tn}"
        ef = Counter(tp[key].get("eval_flows", {}))
        sf = Counter(tp[key].get("synth_flows", {}))

        flows = sorted(set(ef) | set(sf), key=lambda f: ef.get(f, 0), reverse=True)[:15]
        et_tot, st_tot = sum(ef.values()), sum(sf.values())
        y = np.arange(len(flows)); h = 0.35
        ev = [ef.get(f, 0) / et_tot * 100 if et_tot else 0 for f in flows]
        sv = [sf.get(f, 0) / st_tot * 100 if st_tot else 0 for f in flows]
        ax = axes[0, col]
        ax.barh(y + h/2, ev, h, label="Eval", color=EVAL_COLOR, alpha=0.8)
        ax.barh(y - h/2, sv, h, label="Synth", color=SYNTH_COLOR, alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(flows, fontsize=7)
        jsd = tp[key]["flow_jsd"]
        ax.set_title(f"Turn {tn} Flows (JSD={jsd:.3f})"); ax.legend()
        ax.invert_yaxis()

        # Tools
        ec = Counter(tp[key].get("eval_tool_counts", {}))
        sc = Counter(tp[key].get("synth_tool_counts", {}))
        tools = sorted(set(ec) | set(sc), key=lambda t: ec.get(t, 0), reverse=True)[:15]
        et_tot, st_tot = sum(ec.values()), sum(sc.values())
        y = np.arange(len(tools)); h = 0.35
        ev = [ec.get(t, 0) / et_tot * 100 if et_tot else 0 for t in tools]
        sv = [sc.get(t, 0) / st_tot * 100 if st_tot else 0 for t in tools]
        ax = axes[1, col]
        ax.barh(y + h/2, ev, h, label="Eval", color=EVAL_COLOR, alpha=0.8)
        ax.barh(y - h/2, sv, h, label="Synth", color=SYNTH_COLOR, alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(tools, fontsize=7)
        tjsd = tp[key]["tool_jsd"]
        ax.set_title(f"Turn {tn} Tools (JSD={tjsd:.3f})"); ax.legend()
        ax.invert_yaxis()

    plt.suptitle(f"Turn Position Analysis \u2014 {domain.title()}", fontsize=14)
    plt.tight_layout()
    fname = f"turn_position_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)
    return fname


def plot_agent_response(d: dict, domain: str, out: Path):
    ar = d["comparative"]["agent_response"]
    el = ar.get("eval_lengths", [])
    sl = ar.get("synth_lengths", [])
    if not el and not sl:
        return ""
    length_ks = ar["length_ks"]

    fig, ax = plt.subplots(figsize=(10, 5))
    mx = max(max(el, default=0), max(sl, default=0))
    bins = np.linspace(0, min(mx, 120), 40)
    ax.hist(el, bins, alpha=0.6, color=EVAL_COLOR, label="Eval", density=True)
    ax.hist(sl, bins, alpha=0.6, color=SYNTH_COLOR, label="Synth", density=True)
    ax.set_xlabel("Words"); ax.set_ylabel("Density")
    ax.set_title(f"Agent Response Lengths \u2014 {domain.title()} (KS={length_ks:.3f})")
    ax.legend(); plt.tight_layout()
    fname = f"agent_response_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)
    return fname


def plot_conditional_dist(d: dict, domain: str, out: Path):
    cd = d["comparative"]["conditional"]
    per_flow_jsd = cd.get("per_flow_jsd", {})

    if per_flow_jsd:
        flows_sorted = sorted(per_flow_jsd.keys(), key=lambda f: -per_flow_jsd[f])
        fig, ax = plt.subplots(figsize=(8, max(4, len(flows_sorted) * 0.35)))
        y = np.arange(len(flows_sorted))
        vals = [per_flow_jsd[f] for f in flows_sorted]
        colors = ["#F44336" if v > 0.3 else "#FF9800" if v > 0.15 else "#4CAF50" for v in vals]
        ax.barh(y, vals, color=colors, alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(flows_sorted, fontsize=8)
        ax.set_xlabel("JSD"); ax.set_title(f"P(tool|flow) JSD \u2014 {domain.title()}")
        ax.axvline(0.15, color="gray", ls="--", alpha=0.5, label="Yellow threshold")
        ax.axvline(0.3, color="red", ls="--", alpha=0.5, label="Red threshold")
        ax.legend(fontsize=7); ax.invert_yaxis(); plt.tight_layout()
    else:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No common flows", ha="center", va="center")
    fname = f"conditional_dist_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)
    return fname


def plot_topic_coverage(d: dict, domain: str, out: Path):
    tc = d["comparative"]["topic_coverage"]
    coords = tc.get("pca_coords", [])
    if not coords:
        return ""

    pca_x = np.array([c["x"] for c in coords])
    pca_y = np.array([c["y"] for c in coords])
    sources = [c["source"] for c in coords]
    clusters = [c["cluster"] for c in coords]

    cluster_sources = {}
    for src, cl in zip(sources, clusters):
        if cl not in cluster_sources:
            cluster_sources[cl] = {"eval": 0, "synth": 0}
        cluster_sources[cl][src] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    for src, col, marker in [("eval", EVAL_COLOR, "o"), ("synth", SYNTH_COLOR, "D")]:
        mask = np.array([s == src for s in sources])
        ax.scatter(pca_x[mask], pca_y[mask], c=col, marker=marker,
                   alpha=0.5, s=25, label=src)

    n_clusters = tc.get("n_clusters", 0)
    for cl_id in range(n_clusters):
        cl_mask = np.array([c == cl_id for c in clusters])
        if cl_mask.any():
            cx, cy = pca_x[cl_mask].mean(), pca_y[cl_mask].mean()
            cs = cluster_sources.get(cl_id, {"eval": 0, "synth": 0})
            tag = "E" if cs["synth"] == 0 else ("S" if cs["eval"] == 0 else "B")
            color = "#F44336" if tag == "E" else ("#FF9800" if tag == "S" else "#4CAF50")
            ax.annotate(f"C{cl_id}({tag})", (cx, cy), fontsize=7, fontweight="bold",
                        color=color, ha="center")

    eval_only = tc.get("eval_only_clusters", 0)
    synth_only = tc.get("synth_only_clusters", 0)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Scenario Topics \u2014 {domain.title()} "
                 f"(eval-only={eval_only}, synth-only={synth_only})")
    ax.legend(); plt.tight_layout()
    fname = f"topic_coverage_{domain}.png"
    fig.savefig(out / fname, dpi=150); plt.close(fig)
    return fname


# ── Report ───────────────────────────────────────────────────────────

def _scorecard(results, scorecard_key="comparative_scorecard"):
    header = ("| Signal | Metric | Rating | Green / Yellow | Red |\n"
              "|--------|--------|--------|----------------|-----|")
    rows = []
    for domain in results["domains"]:
        d = results[domain]
        scorecard = d.get(scorecard_key, [])
        if not scorecard:
            continue
        rows.append(f"| **{domain.title()}** | | | | |")
        for entry in scorecard:
            emoji = RATING_EMOJI.get(entry["rating"], entry["rating"])
            g = entry["green_threshold"]
            y = entry["yellow_threshold"]
            hib = entry.get("higher_is_better", False)
            if hib:
                bounds = f"> {g} / > {y}"
                red_label = f"<= {y}"
            else:
                bounds = f"< {g} / < {y}"
                red_label = f">= {y}"
            rows.append(f"| {entry['signal']} | {entry['metric']} | {emoji} | {bounds} | {red_label} |")
    return header + "\n" + "\n".join(rows)


def write_report(results, out):
    L = []
    domains = results["domains"]
    plots = results.get("_plots", {})

    L.append("# Synthetic vs Eval: Distribution Analysis\n")

    # ── Executive Summary ──
    L.append("## Executive Summary\n")
    for dm in domains:
        d = results[dm]
        comp = d.get("comparative", {})
        L.append(
            f"**{dm.title()}**: Flow JSD={comp['flow']['jsd']:.3f}, "
            f"Intent JSD={comp['intent']['jsd']:.3f}, "
            f"Tool coverage={comp['tools']['coverage_pct']:.1%}. "
            f"Vocab Jaccard={comp['vocab']['jaccard']:.3f}. "
            f"{len(comp['tools']['coverage_gaps'])} eval tools missing from synth.\n")
    L.append("")

    # ── Intrinsic Scorecard ──
    L.append("## Intrinsic Quality Scorecard\n")
    L.append(_scorecard(results, "intrinsic_scorecard"))
    L.append("")

    # ── Comparative Scorecard ──
    L.append("## Comparative Scorecard\n")
    L.append(_scorecard(results, "comparative_scorecard"))
    L.append("")

    # ── Section 1: Intrinsic Quality ──
    L.append("## 1. Intrinsic Quality\n")
    for dm in domains:
        d = results[dm]
        intr = d.get("intrinsic", {})
        dm_plots = plots.get(dm, {})
        L.append(f"### {dm.title()}\n")

        # Flow/Intent/Tool diversity
        fd = intr.get("flow_diversity", {})
        if fd:
            L.append("#### Flow Diversity\n")
            L.append(f"Entropy = {fd['entropy']:.3f}, "
                     f"Unique flows = {fd['n_unique']}, "
                     f"Uniformity = {fd['uniformity']:.3f}\n")

        id_ = intr.get("intent_diversity", {})
        if id_:
            L.append("#### Intent Diversity\n")
            L.append(f"Entropy = {id_['entropy']:.3f}, "
                     f"Unique intents = {id_['n_unique']}\n")

        td = intr.get("tool_diversity", {})
        if td:
            L.append("#### Tool Diversity\n")
            L.append(f"Entropy = {td['entropy']:.3f}, "
                     f"Unique tools = {td['n_unique']}, "
                     f"Mean tools/turn = {td['mean_tools_per_turn']:.2f}\n")

        la = intr.get("label_agreement", {})
        if la:
            L.append("#### Label Agreement (Ensemble)\n")
            L.append("| Stage | Agreement | Turns | Config | Voters |")
            L.append("|-------|-----------|-------|--------|--------|")
            for stage in ("intent", "flow", "tool"):
                if stage in la:
                    s = la[stage]
                    L.append(f"| {stage.title()} | {s['agreement']:.1%} | "
                             f"{s['n_turns']} | {s['config']} | {s['n_voters']} |")
            L.append("")

        # Naturalness (synth-only)
        nat = intr.get("naturalness", {})
        if nat:
            ss = nat.get("synth_summary", {})
            L.append("#### Naturalness (Synth)\n")
            L.append(f"Mean = {ss.get('mean', 0):.2f} +/- {ss.get('std', 0):.2f} "
                     f"(n={ss.get('n', 0)})\n")
            mb = ss.get("by_model", {})
            if mb:
                L.append("| Provider | Mean | Std | n |")
                L.append("|----------|------|-----|---|")
                for m, s in sorted(mb.items()):
                    L.append(f"| {m} | {s['mean']:.2f} | {s['std']:.2f} | {s['n']} |")
                L.append("")
            contrived = nat.get("contrived_ids", [])
            if contrived:
                L.append(f"**Contrived conversations** (score <= 2): {len(contrived)} total\n")
                pc = nat.get("per_conversation", {})
                for cid in contrived[:10]:
                    info = pc.get(cid, {})
                    reason = info.get("reason", "")
                    L.append(f"- `{cid}`: {reason}")
                if len(contrived) > 10:
                    L.append(f"- ... and {len(contrived) - 10} more")
                L.append("")

        # Diversity (synth-only)
        dv = intr.get("diversity", {})
        if dv and dv.get("synth"):
            L.append("#### Scenario Diversity (Synth)\n")
            L.append("| Dimension | Score |")
            L.append("|-----------|-------|")
            for dim in ("topic", "task", "complexity"):
                L.append(f"| {dim.title()} | {dv['synth'].get(dim, 0):.2f} |")
            L.append("")

    # Model effects (cross-domain)
    me = results.get("model_effects")
    if me:
        L.append("### Model-Specific Effects (Cross-Domain)\n")
        L.append("| Provider | n | Mean Length | Std Length | TTR |")
        L.append("|----------|---|------------|-----------|-----|")
        for m, s in sorted(me["model_stats"].items()):
            L.append(f"| {m} | {s['n']} | {s['mean_length']:.1f} | "
                     f"{s['std_length']:.1f} | {s['ttr']:.3f} |")
        L.append("")
        me_plot = plots.get("_model_effects")
        if me_plot:
            L.append(f"![Model Effects]({me_plot})\n")

    # ── Section 2: Distribution Match ──
    L.append("## 2. Distribution Match\n")
    for dm in domains:
        d = results[dm]
        comp = d.get("comparative", {})
        dm_plots = plots.get(dm, {})
        L.append(f"### {dm.title()}\n")

        # Flow
        L.append("#### Flow Distribution\n")
        L.append(f"JSD = {comp['flow']['jsd']:.4f}, "
                 f"\u03c7\u00b2 = {comp['flow']['chi_stat']:.1f} (p = {comp['flow']['chi_pval']:.3e})\n")
        if comp["flow"]["flagged_ratios"]:
            L.append("**Flagged flows** (ratio < 0.5 or > 2.0):\n")
            for f, r in comp["flow"]["flagged_ratios"]:
                L.append(f"- `{f}`: {r:.2f}x")
            L.append("")
        if dm_plots.get("flow"):
            L.append(f"![Flow Distribution]({dm_plots['flow']})\n")

        # Intent
        L.append("#### Intent Distribution\n")
        L.append(f"JSD = {comp['intent']['jsd']:.4f}, "
                 f"\u03c7\u00b2 = {comp['intent']['chi_stat']:.1f} (p = {comp['intent']['chi_pval']:.3e})\n")
        intents = sorted(set(comp["intent"]["eval_intents"]) | set(comp["intent"]["synth_intents"]))
        L.append("| Intent | Eval | Synth |")
        L.append("|--------|------|-------|")
        for i in intents:
            L.append(f"| {i} | {comp['intent']['eval_intents'].get(i, 0)} | "
                     f"{comp['intent']['synth_intents'].get(i, 0)} |")
        L.append("")
        if dm_plots.get("intent"):
            L.append(f"![Intent Distribution]({dm_plots['intent']})\n")

        # Category
        L.append("#### Category Balance\n")
        L.append("| Category | Eval | Synth |")
        L.append("|----------|------|-------|")
        for c in CATEGORIES:
            L.append(f"| {c} | {comp['category']['eval_categories'].get(c, 0)} | "
                     f"{comp['category']['synth_categories'].get(c, 0)} |")
        L.append("")

        # Length
        L.append("#### Utterance Length\n")
        L.append(f"Turn 1 KS = {comp['length']['ks_t1']['stat']:.3f} "
                 f"(p = {comp['length']['ks_t1']['pval']:.3e}), "
                 f"Turn 3 KS = {comp['length']['ks_t3']['stat']:.3f} "
                 f"(p = {comp['length']['ks_t3']['pval']:.3e})\n")
        L.append("| Stat | Eval T1 | Synth T1 | Eval T3 | Synth T3 |")
        L.append("|------|---------|----------|---------|----------|")
        for s in ("mean", "median", "std", "p10", "p90"):
            L.append(f"| {s} | {comp['length']['eval_t1_stats'][s]:.1f} | "
                     f"{comp['length']['synth_t1_stats'][s]:.1f} | "
                     f"{comp['length']['eval_t3_stats'][s]:.1f} | "
                     f"{comp['length']['synth_t3_stats'][s]:.1f} |")
        L.append("")
        if dm_plots.get("length"):
            L.append(f"![Utterance Length]({dm_plots['length']})\n")

        # Vocab
        L.append("#### Vocabulary\n")
        L.append(f"Eval TTR = {comp['vocab']['eval_ttr']:.3f} ({comp['vocab']['eval_vocab_size']} types), "
                 f"Synth TTR = {comp['vocab']['synth_ttr']:.3f} ({comp['vocab']['synth_vocab_size']} types), "
                 f"Jaccard = {comp['vocab']['jaccard']:.3f}\n")
        if comp["vocab"]["eval_exclusive_words"]:
            L.append("**Eval-exclusive words** (freq >= 2, top 20):\n")
            words = list(comp["vocab"]["eval_exclusive_words"].items())[:20]
            L.append(", ".join(f"`{w}` ({c})" for w, c in words))
            L.append("")

        # Tools
        L.append("#### Tool Usage\n")
        L.append(f"JSD = {comp['tools']['jsd']:.4f}, Coverage = {comp['tools']['coverage_pct']:.1%}\n")
        if comp["tools"]["coverage_gaps"]:
            L.append(f"**Missing from synth** (critical): "
                     f"{', '.join(f'`{t}`' for t in comp['tools']['coverage_gaps'])}\n")
        if comp["tools"]["noise_tools"]:
            L.append(f"**Synth-only** (noise): "
                     f"{', '.join(f'`{t}`' for t in comp['tools']['noise_tools'])}\n")
        L.append(f"Mean tools/turn: eval={comp['tools']['eval_mean_tools']:.2f}, "
                 f"synth={comp['tools']['synth_mean_tools']:.2f}\n")
        if dm_plots.get("tools"):
            L.append(f"![Tool Usage]({dm_plots['tools']})\n")

        # Co-occurrence
        L.append("#### Flow Co-occurrence\n")
        L.append(f"Cosine similarity = {comp['cooccurrence']['cosine_sim']:.3f}, "
                 f"Pair coverage = {comp['cooccurrence']['coverage_pct']:.1%} "
                 f"({comp['cooccurrence']['num_eval_pairs']} eval, "
                 f"{comp['cooccurrence']['num_synth_pairs']} synth)\n")
        if comp["cooccurrence"]["missing_pairs"]:
            mp = comp["cooccurrence"]["missing_pairs"]
            L.append(f"**Missing transitions** ({len(mp)} pairs):")
            for a, b in mp[:10]:
                L.append(f"- `{a}` -> `{b}`")
            L.append("")
        if dm_plots.get("cooccurrence"):
            L.append(f"![Flow Co-occurrence]({dm_plots['cooccurrence']})\n")

        # Embeddings
        L.append("#### Embedding Similarity\n")
        e = comp["embedding"]
        L.append(f"Within-eval = {e['within_eval']:.3f}, "
                 f"Within-synth = {e['within_synth']:.3f}, "
                 f"Cross-set = {e['cross_set']:.3f} "
                 f"({'well-mixed' if e['well_mixed'] else 'NOT well-mixed'})\n")
        if dm_plots.get("embedding"):
            L.append(f"![Embedding PCA]({dm_plots['embedding']})\n")

        # Param completeness
        L.append("#### Parameter Completeness\n")
        pc = comp["params"]
        L.append(f"Eval null rate = {pc['eval_null_rate']:.1%} "
                 f"({pc['eval_null_params']}/{pc['eval_total_params']}), "
                 f"Synth null rate = {pc['synth_null_rate']:.1%} "
                 f"({pc['synth_null_params']}/{pc['synth_total_params']})\n")
        if pc["risk"] == "high":
            L.append("**Warning**: Synth has significantly fewer null params \u2014 "
                     "risk of hallucinated parameter values.\n")

        # Context dependence
        L.append("#### Context Dependence\n")
        cd = comp["context_dep"]
        L.append(f"Terse turn-3 rate (< 8 words): eval={cd['eval_terse_rate']:.1%} "
                 f"({cd['eval_terse']}/{cd['eval_total']}), "
                 f"synth={cd['synth_terse_rate']:.1%} "
                 f"({cd['synth_terse']}/{cd['synth_total']})\n")

    # ── Section 3: Transfer Gap Deep-Dives ──
    L.append("## 3. Transfer Gap Deep-Dives\n")
    for dm in domains:
        d = results[dm]
        comp = d.get("comparative", {})
        dm_plots = plots.get(dm, {})
        L.append(f"### {dm.title()}\n")

        if "per_category" in comp:
            L.append("#### Per-Category Metrics\n")
            pc = comp["per_category"]["categories"]
            L.append("| Category | Flow JSD | Tool JSD | Length KS | Terse % (eval) | Terse % (synth) |")
            L.append("|----------|----------|----------|-----------|----------------|-----------------|")
            for cat in CATEGORIES:
                if cat in pc:
                    c = pc[cat]
                    L.append(f"| {cat} | {c['flow_jsd']:.3f} | {c['tool_jsd']:.3f} | "
                             f"{c['length_ks']:.3f} | {c['terse_rate_eval']:.1%} | "
                             f"{c['terse_rate_synth']:.1%} |")
            L.append("")
            if dm_plots.get("per_category"):
                L.append(f"![Per-Category Metrics]({dm_plots['per_category']})\n")

        if "turn_position" in comp:
            tp = comp["turn_position"]
            L.append("#### Turn Position Analysis\n")
            L.append("| Turn | Flow JSD | Tool JSD |")
            L.append("|------|----------|----------|")
            for tn in ("turn_1", "turn_3"):
                L.append(f"| {tn.replace('_', ' ').title()} | "
                         f"{tp[tn]['flow_jsd']:.3f} | {tp[tn]['tool_jsd']:.3f} |")
            L.append("")
            if tp.get("t3_worse"):
                ratio = tp["turn_3"]["flow_jsd"] / tp["turn_1"]["flow_jsd"] if tp["turn_1"]["flow_jsd"] > 0 else float("inf")
                L.append(f"**Warning**: Turn-3 flow gap is {ratio:.1f}x worse than turn-1.\n")
            if dm_plots.get("turn_position"):
                L.append(f"![Turn Position]({dm_plots['turn_position']})\n")

        if "agent_response" in comp:
            ar = comp["agent_response"]
            L.append("#### Agent Response Comparison\n")
            L.append(f"Length KS = {ar['length_ks']:.3f}, "
                     f"Vocab Jaccard = {ar['vocab_jaccard']:.3f}\n")
            L.append(f"Mean length: eval = {ar['eval_mean_length']:.1f}, "
                     f"synth = {ar['synth_mean_length']:.1f}\n")
            if dm_plots.get("agent_response"):
                L.append(f"![Agent Response]({dm_plots['agent_response']})\n")

        if "conditional" in comp:
            cd = comp["conditional"]
            L.append("#### Conditional Distributions\n")
            L.append(f"Avg P(tool|flow) JSD = {cd['tool_given_flow_avg_jsd']:.3f}, "
                     f"Avg P(flow|intent) JSD = {cd['flow_given_intent_avg_jsd']:.3f}\n")
            if cd.get("per_flow_jsd"):
                L.append("| Flow | P(tool&#124;flow) JSD |")
                L.append("|------|---------------------|")
                for f, j in sorted(cd["per_flow_jsd"].items(), key=lambda x: -x[1])[:10]:
                    flag = " **!**" if j > 0.3 else ""
                    L.append(f"| {f} | {j:.3f}{flag} |")
                L.append("")
            if cd.get("per_intent_jsd"):
                L.append("| Intent | P(flow&#124;intent) JSD |")
                L.append("|--------|----------------------|")
                for i, j in sorted(cd["per_intent_jsd"].items(), key=lambda x: -x[1]):
                    L.append(f"| {i} | {j:.3f} |")
                L.append("")
            if cd.get("worst_flow_conditionals"):
                worst = cd["worst_flow_conditionals"]
                L.append(f"Worst conditional gaps: "
                         f"{', '.join(f'`{f}` ({j:.3f})' for f, j in worst)}\n")
            if dm_plots.get("conditional"):
                L.append(f"![Conditional Distributions]({dm_plots['conditional']})\n")

        if "topic_coverage" in comp:
            tc = comp["topic_coverage"]
            L.append("#### Scenario Topic Coverage\n")
            L.append(f"{tc['n_clusters']} clusters: "
                     f"{tc['eval_only_clusters']} eval-only, "
                     f"{tc['synth_only_clusters']} synth-only, "
                     f"coverage = {tc['coverage_pct']:.1%}\n")
            if tc.get("cluster_details"):
                L.append("| Cluster | Eval | Synth | Status |")
                L.append("|---------|------|-------|--------|")
                for cl_id in sorted(tc["cluster_details"].keys(), key=lambda x: int(x)):
                    cs = tc["cluster_details"][cl_id]
                    status = ("eval-only" if cs["synth"] == 0
                              else ("synth-only" if cs["eval"] == 0 else "both"))
                    L.append(f"| {cl_id} | {cs['eval']} | {cs['synth']} | {status} |")
                L.append("")
            if dm_plots.get("topic_coverage"):
                L.append(f"![Topic Coverage]({dm_plots['topic_coverage']})\n")

        # Qualitative comparisons
        if "naturalness" in comp:
            n = comp["naturalness"]
            intr = d.get("intrinsic", {})
            intr_nat = intr.get("naturalness", {})
            ss = intr_nat.get("synth_summary", {})
            L.append("#### Naturalness (Comparison)\n")
            es = n.get("eval_summary", {})
            L.append(f"Eval: {es.get('mean', 0):.2f} +/- {es.get('std', 0):.2f} (n={es.get('n', 0)}), "
                     f"Synth: {ss.get('mean', 0):.2f} +/- {ss.get('std', 0):.2f} (n={ss.get('n', 0)})\n")
            L.append(f"Gap: {n.get('gap', 0):.2f}, Welch's t={n.get('t_stat', 0):.2f}, "
                     f"p={n.get('t_pval', 1):.3f}\n")

        if "ambiguity" in comp:
            a = comp["ambiguity"]
            L.append("#### Ambiguity Quality\n")
            L.append(f"Eval: {a['eval_mean']:.2f} +/- {a['eval_std']:.2f} (n={a['n_eval']}), "
                     f"Synth: {a['synth_mean']:.2f} +/- {a['synth_std']:.2f} (n={a['n_synth']})\n")
            L.append(f"Gap: {a['gap']:.2f}, Welch's t={a['t_stat']:.2f}, p={a['t_pval']:.3f}\n")

        if "diversity" in comp:
            dv = comp["diversity"]
            L.append("#### Scenario Diversity (Comparison)\n")
            L.append("| Dimension | Eval | Synth |")
            L.append("|-----------|------|-------|")
            for dim in ("topic", "task", "complexity"):
                L.append(f"| {dim.title()} | {dv['eval'].get(dim, 0):.2f} | "
                         f"{dv['synth'].get(dim, 0):.2f} |")
            L.append("")

    # ── Section 4: Recommendations ──
    L.append("## 4. Recommendations\n")
    for dm in domains:
        d = results[dm]
        comp = d.get("comparative", {})
        L.append(f"### {dm.title()}\n")
        risks = []
        if comp["flow"]["jsd"] >= 0.15:
            risks.append("- **High flow JSD**: Significant flow distribution mismatch. "
                         "Consider rebalancing generation.")
        if comp["tools"]["coverage_gaps"]:
            gaps = comp["tools"]["coverage_gaps"]
            risks.append(f"- **Tool coverage gaps**: {len(gaps)} eval tools missing from synth: "
                         f"{', '.join(gaps[:5])}")
        if comp["vocab"]["jaccard"] < 0.3:
            risks.append("- **Low vocabulary overlap**: Synthetic language diverges from eval. "
                         "Review generation prompts.")
        if comp["params"]["risk"] == "high":
            risks.append("- **Parameter hallucination risk**: Synth has fewer null params. "
                         "Add null-param examples.")
        if comp["context_dep"]["synth_terse_rate"] < comp["context_dep"]["eval_terse_rate"] * 0.5:
            risks.append("- **Under-represented terse follow-ups**: Synth lacks short "
                         "context-dependent turn-3 utterances.")
        if comp["cooccurrence"]["coverage_pct"] < 0.6:
            risks.append("- **Low flow-pair coverage**: Many eval transitions missing from synth.")
        L.append("\n".join(risks) if risks else
                 "No critical risks identified. Synthetic data appears well-aligned.")
        L.append("")

    path = out / "synth_vs_eval_report.md"
    path.write_text("\n".join(L))
    return path


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate plots + report from metrics JSON")
    ap.add_argument("--domain", choices=["dana", "hugo", "both"], default="both")
    ap.add_argument("--metrics-dir", type=Path, default=None,
                    help="Directory containing metrics_*.json (default: analysis/)")
    args = ap.parse_args()

    out = SCRIPT_DIR
    metrics_dir = args.metrics_dir or SCRIPT_DIR
    domains = DOMAINS if args.domain == "both" else [args.domain]

    # Check if metrics JSON exists; if not, try running compute_metrics.py
    for dm in domains:
        mpath = metrics_dir / f"metrics_{dm}.json"
        if not mpath.exists():
            print(f"  metrics_{dm}.json not found \u2014 running compute_metrics.py...")
            compute_script = AUG_DIR / "compute_metrics.py"
            subprocess.run([sys.executable, str(compute_script),
                            "--domain", dm, "--skip-llm"], check=True)

    results = {"domains": domains, "_plots": {}}

    for dm in domains:
        print(f"\n{'=' * 60}")
        print(f"  Generating plots for {dm.upper()}")
        print(f"{'=' * 60}")

        d = load_metrics(dm, metrics_dir)
        results[dm] = d

        dm_plots = {}

        print("  [1/12] Flow distribution...")
        dm_plots["flow"] = plot_flow_distribution(d, dm, out)

        print("  [2/12] Intent distribution...")
        dm_plots["intent"] = plot_intent_distribution(d, dm, out)

        print("  [3/12] Utterance length...")
        dm_plots["length"] = plot_utterance_length(d, dm, out)

        print("  [4/12] Tool usage...")
        dm_plots["tools"] = plot_tool_usage(d, dm, out)

        print("  [5/12] Flow co-occurrence...")
        dm_plots["cooccurrence"] = plot_flow_cooccurrence(d, dm, out)

        print("  [6/12] Embedding PCA...")
        dm_plots["embedding"] = plot_embedding_pca(d, dm, out)

        print("  [7/12] Per-category metrics...")
        dm_plots["per_category"] = plot_per_category_metrics(d, dm, out)

        print("  [8/12] Turn position...")
        dm_plots["turn_position"] = plot_turn_position(d, dm, out)

        print("  [9/12] Agent response...")
        dm_plots["agent_response"] = plot_agent_response(d, dm, out)

        print("  [10/12] Conditional distributions...")
        dm_plots["conditional"] = plot_conditional_dist(d, dm, out)

        print("  [11/12] Topic coverage...")
        dm_plots["topic_coverage"] = plot_topic_coverage(d, dm, out)

        results["_plots"][dm] = dm_plots

    # Model effects
    me = load_model_effects(metrics_dir)
    if me:
        print("\n  [12/12] Model effects...")
        me_plot = plot_model_effects(me, out)
        results["model_effects"] = me
        results["_plots"]["_model_effects"] = me_plot

    # Write report
    print("\n  Writing report...")
    rp = write_report(results, out)
    print(f"\n  Report: {rp}")
    pngs = sorted(out.glob("*.png"))
    print(f"  Generated {len(pngs)} plots:")
    for p in pngs:
        print(f"    {p.name}")


if __name__ == "__main__":
    main()
