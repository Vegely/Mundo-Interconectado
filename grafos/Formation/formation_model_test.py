"""
formation_model_test.py
=======================
Tests whether the PyPI dependency network was likely formed by a
Barabási–Albert (preferential attachment / "rich-get-richer") process.

Methodology mirrors the teacher's R scripts:
  - p-valor_redes_reales.R   → p-values against an ER null model
  - p-valores_entre_modelos.R → p-values for BA / WS synthetic graphs

Strategy
--------
1. Load the real network (pypi_multiseed_10k.graphml) – same file used in
   communities.py.
2. Compute two fingerprint statistics on the REAL network:
      • Global clustering coefficient (transitivity)
      • Degree-distribution entropy   H = -Σ p(k) log2 p(k)
3. Build a Monte-Carlo null distribution using Erdős–Rényi (ER) graphs
   with the same n and m  →  two-sided p-values for both statistics.
4. Repeat step 3 but the "observed" graphs are themselves Barabási–Albert
   (BA) synthetic networks (Nr realisations), exactly like the teacher's
   p-valores_entre_modelos.R.  This tells us whether BA graphs look like
   ER or whether they are distinguishable.
5. Compare the real network's statistics directly against BA synthetic
   distributions to see how compatible the real net is with BA.
6. Plot everything: histograms, p-value distributions, degree distributions
   on log-log axes.

Usage
-----
    python formation_model_test.py

Requirements: igraph, numpy, matplotlib, scipy
    pip install igraph numpy matplotlib scipy
"""

import random
import math
import warnings
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
ig.set_random_number_generator(random.Random(SEED))

# ── configuration ─────────────────────────────────────────────────────────────
GRAPH_FILE   = "pypi_multiseed_10k.graphml"   # same file as communities.py
N_MONTE      = 1000    # ER Monte-Carlo samples per test  (teacher used 300–10 000)
NR           = 100     # number of BA realisations for the p-value distribution
BA_M         = 2       # edges added per new node in BA (teacher used m=2)
ALPHA        = 0.05    # significance level


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def entropy_degree(g: ig.Graph) -> float:
    """Shannon entropy (bits) of the degree distribution – undirected view."""
    deg = np.array(g.degree())          # undirected degree (as in teacher's R)
    counts = np.bincount(deg)
    probs  = counts[counts > 0] / len(deg)
    return float(-np.sum(probs * np.log2(probs)))


def clustering(g: ig.Graph) -> float:
    """Global (transitivity) clustering coefficient."""
    return g.transitivity_undirected(mode="zero")


def two_sided_pval(sample: np.ndarray, observed: float) -> float:
    """
    Two-sided Monte-Carlo p-value, exactly matching the teacher's formula:
        2 * min(#{sample <= obs}+1,  #{sample > obs}+1) / (N+1)
    """
    n   = len(sample)
    leq = int(np.sum(sample <= observed))
    return 2 * min(leq + 1, n + 1 - leq + 1) / (n + 1)


def er_null_samples(n: int, m: int, n_monte: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate N_MONTE Erdős–Rényi G(n,m) graphs and return (cc_array, ent_array)."""
    cc_samples  = np.empty(n_monte)
    ent_samples = np.empty(n_monte)
    for i in range(n_monte):
        g = ig.Graph.Erdos_Renyi(n=n, m=m, directed=False, loops=False)
        cc_samples[i]  = clustering(g)
        ent_samples[i] = entropy_degree(g)
    return cc_samples, ent_samples


def ba_graph(n: int, m: int) -> ig.Graph:
    """Barabási–Albert preferential-attachment graph (undirected)."""
    return ig.Graph.Barabasi(n=n, m=m, directed=False)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load the real network
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  PyPI Dependency Network – Formation Model Test")
print("=" * 65)

print(f"\n[1] Loading '{GRAPH_FILE}' …")
real_g = ig.Graph.Read_GraphML(GRAPH_FILE)
print(f"    Directed graph: {real_g.vcount():,} nodes, {real_g.ecount():,} edges")

# Work with the undirected version for the clustering / entropy tests
# (mirrors the teacher's approach which always uses undirected igraph metrics)
real_ug = real_g.as_undirected(combine_edges="first")
n_real  = real_ug.vcount()
m_real  = real_ug.ecount()
print(f"    Undirected view: {n_real:,} nodes, {m_real:,} edges")

cc_real  = clustering(real_ug)
ent_real = entropy_degree(real_ug)
print(f"\n    Clustering coefficient : {cc_real:.6f}")
print(f"    Degree entropy (bits)  : {ent_real:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ER null model  →  p-values for the REAL network  (p-valor_redes_reales.R)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[2] Building ER null distribution ({N_MONTE} samples, n={n_real}, m={m_real}) …")
print("    (this may take a few minutes for a 6 k-node graph)")
cc_er, ent_er = er_null_samples(n_real, m_real, N_MONTE)

pval_cc_real  = two_sided_pval(cc_er,  cc_real)
pval_ent_real = two_sided_pval(ent_er, ent_real)

print(f"\n    ── Real network vs ER null ──")
print(f"    p-value (clustering)  : {pval_cc_real:.4f}  {'← significant' if pval_cc_real < ALPHA else ''}")
print(f"    p-value (entropy)     : {pval_ent_real:.4f}  {'← significant' if pval_ent_real < ALPHA else ''}")
if pval_cc_real < ALPHA or pval_ent_real < ALPHA:
    print("    → Real network is DISTINGUISHABLE from ER  (expected for BA / WS).")
else:
    print("    → Cannot distinguish from ER at this sample size.")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BA p-value distribution  (p-valores_entre_modelos.R – BA section)
#     Each of Nr BA graphs is tested against its own ER null.
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[3] BA self-test: {NR} BA graphs vs their ER null ({N_MONTE} each) …")
pvals_ba_cc  = np.empty(NR)
pvals_ba_ent = np.empty(NR)

ba_cc_vals  = np.empty(NR)   # keep observed stats for direct comparison later
ba_ent_vals = np.empty(NR)

for j in range(NR):
    g_ba = ba_graph(n_real, BA_M)
    m_ba = g_ba.ecount()

    cc_ba_j  = clustering(g_ba)
    ent_ba_j = entropy_degree(g_ba)
    ba_cc_vals[j]  = cc_ba_j
    ba_ent_vals[j] = ent_ba_j

    # ER null with same (n, m) as this BA realisation
    cc_s, ent_s = er_null_samples(n_real, m_ba, N_MONTE)
    pvals_ba_cc[j]  = two_sided_pval(cc_s,  cc_ba_j)
    pvals_ba_ent[j] = two_sided_pval(ent_s, ent_ba_j)

    if (j + 1) % 10 == 0:
        print(f"    {j+1}/{NR} done …")

frac_sig_ba_cc  = np.mean(pvals_ba_cc  < ALPHA)
frac_sig_ba_ent = np.mean(pvals_ba_ent < ALPHA)
print(f"\n    ── BA graphs vs ER null ──")
print(f"    Fraction significant (clustering) : {frac_sig_ba_cc:.2%}")
print(f"    Fraction significant (entropy)    : {frac_sig_ba_ent:.2%}")
print("    (If BA is truly different from ER, most p-values should be < 0.05)")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Direct comparison: real network statistics vs BA distribution
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4] Comparing real network directly against BA distribution …")
pval_real_vs_ba_cc  = two_sided_pval(ba_cc_vals,  cc_real)
pval_real_vs_ba_ent = two_sided_pval(ba_ent_vals, ent_real)

print(f"\n    ── Real network vs BA distribution ──")
print(f"    Mean BA clustering : {np.mean(ba_cc_vals):.6f}   (real: {cc_real:.6f})")
print(f"    Mean BA entropy    : {np.mean(ba_ent_vals):.6f}   (real: {ent_real:.6f})")
print(f"    p-value (clustering)  : {pval_real_vs_ba_cc:.4f}")
print(f"    p-value (entropy)     : {pval_real_vs_ba_ent:.4f}")

if pval_real_vs_ba_cc >= ALPHA and pval_real_vs_ba_ent >= ALPHA:
    verdict = "COMPATIBLE with Barabási–Albert  ✓"
elif pval_real_vs_ba_cc < ALPHA and pval_real_vs_ba_ent < ALPHA:
    verdict = "INCOMPATIBLE with Barabási–Albert on BOTH statistics  ✗"
else:
    stat_names = []
    if pval_real_vs_ba_cc  < ALPHA: stat_names.append("clustering")
    if pval_real_vs_ba_ent < ALPHA: stat_names.append("entropy")
    verdict = f"PARTIALLY compatible – differs on: {', '.join(stat_names)}"

print(f"\n    Verdict: {verdict}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Degree-distribution power-law check (visual + KS)
#     BA produces scale-free networks  →  log-log degree distribution is linear
# ══════════════════════════════════════════════════════════════════════════════

print("\n[5] Degree-distribution analysis (log-log) …")
in_deg  = np.array(real_g.indegree())
out_deg = np.array(real_g.outdegree())

def ccdf(degrees: np.ndarray):
    """Complementary CDF  P(K >= k) for plotting on log-log axes."""
    k_vals = np.sort(np.unique(degrees[degrees > 0]))
    p_vals = np.array([np.mean(degrees >= k) for k in k_vals])
    return k_vals, p_vals

k_in,  p_in  = ccdf(in_deg)
k_out, p_out = ccdf(out_deg)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Plots
# ══════════════════════════════════════════════════════════════════════════════

print("\n[6] Generating plots …")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("PyPI Dependency Network – Formation Model Test\n"
             "Barabási–Albert (rich-get-richer) hypothesis", fontsize=14, y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Row 0: ER null vs REAL ─────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
ax0.hist(cc_er, bins=40, color="steelblue", edgecolor="white", alpha=0.8, label="ER null")
ax0.axvline(cc_real, color="crimson", lw=2, label=f"Real ({cc_real:.4f})")
ax0.set_title(f"ER null – Clustering\np = {pval_cc_real:.4f}")
ax0.set_xlabel("Clustering coefficient"); ax0.set_ylabel("Count"); ax0.legend(fontsize=8)

ax1 = fig.add_subplot(gs[0, 1])
ax1.hist(ent_er, bins=40, color="steelblue", edgecolor="white", alpha=0.8, label="ER null")
ax1.axvline(ent_real, color="crimson", lw=2, label=f"Real ({ent_real:.3f} bits)")
ax1.set_title(f"ER null – Entropy\np = {pval_ent_real:.4f}")
ax1.set_xlabel("Degree entropy (bits)"); ax0.set_ylabel("Count"); ax1.legend(fontsize=8)

# ── Row 0 col 2: summary table ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis("off")
rows = [
    ["Statistic",         "Real value",   "ER mean",               "p-value (vs ER)"],
    ["Clustering",        f"{cc_real:.4f}",  f"{np.mean(cc_er):.4f}", f"{pval_cc_real:.4f}"],
    ["Degree entropy",    f"{ent_real:.4f}", f"{np.mean(ent_er):.4f}",f"{pval_ent_real:.4f}"],
    ["",                  "",             "",                      ""],
    ["Statistic",         "Real value",   "BA mean",               "p-value (vs BA)"],
    ["Clustering",        f"{cc_real:.4f}",  f"{np.mean(ba_cc_vals):.4f}", f"{pval_real_vs_ba_cc:.4f}"],
    ["Degree entropy",    f"{ent_real:.4f}", f"{np.mean(ba_ent_vals):.4f}",f"{pval_real_vs_ba_ent:.4f}"],
]
table = ax2.table(cellText=rows, loc="center", cellLoc="center")
table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.6)
ax2.set_title("Summary table", pad=10)

# ── Row 1: BA p-value histograms ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(pvals_ba_cc, bins=20, color="darkorange", edgecolor="white", alpha=0.8)
ax3.axvline(ALPHA, color="black", lw=1.5, ls="--", label=f"α={ALPHA}")
ax3.set_title(f"BA vs ER: p-values (clustering)\n"
              f"{frac_sig_ba_cc:.0%} significant")
ax3.set_xlabel("p-value"); ax3.set_ylabel("Count"); ax3.legend(fontsize=8)

ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(pvals_ba_ent, bins=20, color="darkorange", edgecolor="white", alpha=0.8)
ax4.axvline(ALPHA, color="black", lw=1.5, ls="--", label=f"α={ALPHA}")
ax4.set_title(f"BA vs ER: p-values (entropy)\n"
              f"{frac_sig_ba_ent:.0%} significant")
ax4.set_xlabel("p-value"); ax4.set_ylabel("Count"); ax4.legend(fontsize=8)

# ── Row 1 col 2: real vs BA scatter ───────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.scatter(ba_cc_vals, ba_ent_vals, alpha=0.5, color="darkorange", s=20, label="BA realisations")
ax5.scatter([cc_real], [ent_real], color="crimson", s=100, zorder=5,
            marker="*", label="Real network")
ax5.set_xlabel("Clustering coefficient")
ax5.set_ylabel("Degree entropy (bits)")
ax5.set_title("Real network vs BA realisations")
ax5.legend(fontsize=8)

# ── Row 2: degree distributions log-log ───────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
ax6.loglog(k_in, p_in, "o", ms=3, color="steelblue", alpha=0.7, label="Empirical CCDF")
# Fit a power law to the tail (k >= 3, as found in the report)
mask = k_in >= 3
if mask.sum() > 5:
    slope, intercept, r, *_ = stats.linregress(np.log(k_in[mask]), np.log(p_in[mask]))
    ax6.loglog(k_in[mask],
               np.exp(intercept) * k_in[mask] ** slope,
               "--", color="crimson", lw=1.5,
               label=f"Power-law fit γ≈{-slope:.2f}")
ax6.set_title("In-degree CCDF (log-log)\nLinear → power law")
ax6.set_xlabel("In-degree k"); ax6.set_ylabel("P(K ≥ k)")
ax6.legend(fontsize=8)

ax7 = fig.add_subplot(gs[2, 1])
ax7.loglog(k_out, p_out, "o", ms=3, color="seagreen", alpha=0.7, label="Empirical CCDF")
mask2 = k_out >= 11
if mask2.sum() > 5:
    slope2, intercept2, *_ = stats.linregress(np.log(k_out[mask2]), np.log(p_out[mask2]))
    ax7.loglog(k_out[mask2],
               np.exp(intercept2) * k_out[mask2] ** slope2,
               "--", color="crimson", lw=1.5,
               label=f"Power-law fit γ≈{-slope2:.2f}")
ax7.set_title("Out-degree CCDF (log-log)\nLinear → power law")
ax7.set_xlabel("Out-degree k"); ax7.set_ylabel("P(K ≥ k)")
ax7.legend(fontsize=8)

# ── Row 2 col 2: verdict text ─────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis("off")
summary_lines = [
    "INTERPRETATION",
    "─" * 38,
    "",
    f"Real net vs ER null:",
    f"  clustering p = {pval_cc_real:.4f}",
    f"  entropy    p = {pval_ent_real:.4f}",
    f"  → Real net is {'NOT ' if pval_cc_real >= ALPHA and pval_ent_real >= ALPHA else ''}distinguishable",
    f"    from ER.",
    "",
    f"Real net vs BA distribution:",
    f"  clustering p = {pval_real_vs_ba_cc:.4f}",
    f"  entropy    p = {pval_real_vs_ba_ent:.4f}",
    "",
    f"  Verdict:",
    f"  {verdict}",
    "",
    "A power-law in-degree distribution",
    "and large entropy relative to ER",
    "are hallmarks of preferential",
    "attachment (Barabási–Albert).",
]
ax8.text(0.05, 0.95, "\n".join(summary_lines),
         transform=ax8.transAxes, fontsize=8.5,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6))

plt.savefig("formation_model_test.png", dpi=150, bbox_inches="tight")
print("    Saved → formation_model_test.png")
plt.show()

print("\n[Done] All tests complete.")