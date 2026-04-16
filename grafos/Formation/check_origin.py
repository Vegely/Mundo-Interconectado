"""
formation_model_test.py  (v3 – directed, multi-model, parallelized & optimized)
======================================================
Tests whether the PyPI dependency network is compatible with three
candidate formation models, working with the DIRECTED graph throughout.
"""

import random
import warnings
import multiprocessing
import concurrent.futures
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")

# ── configuration ─────────────────────────────────────────────────────────────
GRAPH_FILE = "pypi_multiseed_10k.graphml"
N_MC       = 200    # synthetic graphs per model
ALPHA      = 0.05   # significance level

# Bianconi–Barabási params
BB_M       = 2      # edges added per step
BB_ETA_LOW = 0.1    # fitness drawn from Uniform(BB_ETA_LOW, 1.0)

# Copying model params
CP_BETA    = 0.6    # probability of copying a neighbour's edge (vs random)

# LFR params – will be derived from real network
LFR_MU     = 0.3   # mixing parameter (fraction of inter-community edges)

# ══════════════════════════════════════════════════════════════════════════════
# Statistics (all directed-aware)
# ══════════════════════════════════════════════════════════════════════════════

def indegree_entropy(g: ig.Graph) -> float:
    deg = np.array(g.indegree())
    counts = np.bincount(deg)
    probs = counts[counts > 0] / len(deg)
    return float(-np.sum(probs * np.log2(probs)))

def outdegree_entropy(g: ig.Graph) -> float:
    deg = np.array(g.outdegree())
    counts = np.bincount(deg)
    probs = counts[counts > 0] / len(deg)
    return float(-np.sum(probs * np.log2(probs)))

def clustering_directed(g: ig.Graph) -> float:
    return g.as_undirected(combine_edges="first").transitivity_undirected(mode="zero")

def reciprocity(g: ig.Graph) -> float:
    return g.reciprocity()

def compute_stats(g: ig.Graph) -> np.ndarray:
    return np.array([
        clustering_directed(g),
        indegree_entropy(g),
        outdegree_entropy(g),
        reciprocity(g),
    ])

STAT_NAMES = ["Clustering", "In-deg entropy", "Out-deg entropy", "Reciprocity"]

def two_sided_pval(sample: np.ndarray, observed: float) -> float:
    n   = len(sample)
    leq = int(np.sum(sample <= observed))
    return 2 * min(leq + 1, n + 1 - leq + 1) / (n + 1)

def ccdf(degrees: np.ndarray):
    degrees = degrees[degrees > 0]
    if len(degrees) == 0:
        return np.array([]), np.array([])
    k_vals  = np.sort(np.unique(degrees))
    p_vals  = np.array([np.mean(degrees >= k) for k in k_vals])
    return k_vals, p_vals

# ══════════════════════════════════════════════════════════════════════════════
# Fast Model Generators
# ══════════════════════════════════════════════════════════════════════════════

def bianconi_barabasi(n: int, m: int = 2) -> ig.Graph:
    """Optimized Bianconi-Barabasi using cached in-degrees."""
    rng = np.random.default_rng()
    etas = rng.uniform(BB_ETA_LOW, 1.0, size=n)
    edges = []
    
    in_deg = np.zeros(n, dtype=float)
    
    # Seed graph
    for i in range(1, m + 1):
        edges.append((i, 0))
        in_deg[0] += 1

    weights = etas.copy()
    weights[:m+1] = etas[:m+1] * (in_deg[:m+1] + 1e-9)

    for t in range(m + 1, n):
        w_t = weights[:t]
        prob = w_t / w_t.sum()
        
        # Select targets
        targets = set(rng.choice(t, size=m, p=prob, replace=True))
        
        for tgt in targets:
            edges.append((t, tgt))
            in_deg[tgt] += 1
            # Update cache locally, $O(1)$ instead of scanning graph
            weights[tgt] = etas[tgt] * (in_deg[tgt] + 1e-9)

    g = ig.Graph(n=n, edges=edges, directed=True)
    g.vs["eta"] = etas.tolist()
    return g


def copying_model(n: int, beta: float = 0.6, m_init: int = 3) -> ig.Graph:
    """Optimized Copying Model using an adjacency list."""
    rng = np.random.default_rng()
    edges = [(i, i + 1) for i in range(m_init - 1)]

    # Caching out-edges avoids an $O(E)$ search at every step
    adj_out = [[] for _ in range(n)]
    for u, v in edges:
        adj_out[u].append(v)

    for t in range(m_init, n):
        proto = rng.integers(0, t)
        targets = set()

        for w in adj_out[proto]:
            if w != t and rng.random() < beta:
                targets.add(w)

        rand_tgt = rng.integers(0, t)
        targets.add(rand_tgt)

        for w in targets:
            edges.append((t, w))
            adj_out[t].append(w)

    return ig.Graph(n=n, edges=edges, directed=True)


def lfr_benchmark(n: int, avg_deg: float, max_deg: int, mu: float = 0.3) -> ig.Graph:
    """NetworkX fallback to bypass the igraph C-API matrix bug completely."""
    import networkx as nx
    
    n_comm = 20
    sizes  = [n // n_comm] * n_comm
    sizes[-1] += n - sum(sizes)
    
    comm_size = n // n_comm
    # Force pure Python floats just to be safe
    p_in   = float(min(1.0, (avg_deg * (1 - mu)) / comm_size))
    p_out  = float(min(1.0, (avg_deg * mu) / (n - comm_size)))
    
    pref_matrix = [
        [p_in if i == j else p_out for j in range(n_comm)]
        for i in range(n_comm)
    ]
    
    # Generate the graph using NetworkX's stable block model generator
    gnx = nx.stochastic_block_model(sizes, pref_matrix, directed=True, selfloops=False)
    
    # Convert the edges instantly back to igraph
    return ig.Graph(n=n, edges=list(gnx.edges()), directed=True)

# ══════════════════════════════════════════════════════════════════════════════
# Parallel Worker
# ══════════════════════════════════════════════════════════════════════════════

def worker_run(args):
    """Worker function for parallel graph generation and stat computation."""
    model_key, seed, n_real, avg_deg, max_deg = args
    
    # Isolate randomness per worker
    np.random.seed(seed)
    random.seed(seed)
    ig.set_random_number_generator(random.Random(seed))

    if model_key == "BB":
        g = bianconi_barabasi(n_real, m=BB_M)
    elif model_key == "CP":
        g = copying_model(n_real, beta=CP_BETA)
    else:
        g = lfr_benchmark(n_real, avg_deg=avg_deg, max_deg=max_deg, mu=LFR_MU)

    return compute_stats(g)


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  PyPI Dependency Network – Formation Model Test  (v3, fast)")
    print("=" * 65)

    print(f"\n[1] Loading '{GRAPH_FILE}' …")
    try:
        real_g = ig.Graph.Read_GraphML(GRAPH_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find '{GRAPH_FILE}'. Please check the path.")
        return

    if not real_g.is_directed():
        real_g = real_g.as_directed(mode="mutual")
        
    n_real   = real_g.vcount()
    m_real   = real_g.ecount()
    avg_deg  = m_real / n_real
    max_deg  = max(real_g.indegree())
    
    print(f"    Directed: {n_real:,} nodes, {m_real:,} edges")
    print(f"    Avg degree: {avg_deg:.2f}   Max in-degree: {max_deg}")

    real_stats = compute_stats(real_g)
    for name, val in zip(STAT_NAMES, real_stats):
        print(f"    {name:20s}: {val:.6f}")

    in_deg_real  = np.array(real_g.indegree())
    out_deg_real = np.array(real_g.outdegree())

    model_configs = [
        ("Bianconi–Barabási\n(fitness)",   "BB",  "#e07b39"),
        ("Copying / Duplication",          "CP",  "#3a86b4"),
        ("LFR Benchmark\n(community)",     "LFR", "#5cb85c"),
    ]

    synth_stats = {}

    print(f"\n[2] Generating synthetic distributions (Parallel Execution on {multiprocessing.cpu_count()} cores) …")
    
    base_seed = 42
    
    for label, key, color in model_configs:
        print(f"    -> Spawning {N_MC} instances of {key} ... ", end="", flush=True)
        
        # Assign a deterministic, unique seed per iteration
        seed_offset = 1000 if key == "BB" else 2000 if key == "CP" else 3000
        worker_args = [(key, base_seed + i + seed_offset, n_real, avg_deg, max_deg) for i in range(N_MC)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(executor.map(worker_run, worker_args))
            
        synth_stats[key] = np.array(results)
        print("Done!")

    print("\n[3] Computing p-values …")
    pvals = {}
    for _, key, _ in model_configs:
        pvals[key] = np.array([
            two_sided_pval(synth_stats[key][:, i], real_stats[i])
            for i in range(len(STAT_NAMES))
        ])

    print(f"\n{'Model':<22} {'Clustering':>12} {'In-ent':>10} {'Out-ent':>10} {'Recipr':>10}  Verdict")
    print("-" * 75)
    for label, key, _ in model_configs:
        ps   = pvals[key]
        sig  = [("*" if p < ALPHA else " ") for p in ps]
        n_sig = sum(p < ALPHA for p in ps)
        if n_sig == 0:
            verdict = "COMPATIBLE  ✓"
        elif n_sig == len(STAT_NAMES):
            verdict = "INCOMPATIBLE ✗"
        else:
            failed = [STAT_NAMES[i] for i, p in enumerate(ps) if p < ALPHA]
            verdict = f"PARTIAL – fails on: {', '.join(failed)}"
        label_clean = label.replace("\n", " ")
        print(f"{label_clean:<22} {ps[0]:>10.4f}{sig[0]} {ps[1]:>9.4f}{sig[1]} "
              f"{ps[2]:>9.4f}{sig[2]} {ps[3]:>9.4f}{sig[3]}   {verdict}")
    print("  (* = significant at α=0.05 → real network differs from model on this stat)")

    print("\n[4] Generating plots …")

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "PyPI Dependency Network – Directed Formation Model Comparison\n"
        "Bianconi–Barabási  |  Copying/Duplication  |  LFR Benchmark",
        fontsize=13, y=0.99
    )

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.52, wspace=0.38)
    stat_idx = {"Clustering": 0, "In-deg entropy": 1, "Out-deg entropy": 2, "Reciprocity": 3}

    def hist_panel(ax, synth, real_val, pval, stat_name, color, model_name):
        ax.hist(synth, bins=30, color=color, edgecolor="white", alpha=0.75)
        ax.axvline(real_val, color="crimson", lw=2, ls="--", label=f"Real ({real_val:.3f})")
        sig = "✗ differs" if pval < ALPHA else "✓ compatible"
        ax.set_title(f"{model_name}\n{stat_name}  p={pval:.3f}  {sig}", fontsize=8)
        ax.set_xlabel(stat_name, fontsize=7)
        ax.legend(fontsize=7)

    # Row 0: In-degree entropy
    for col, (label, key, color) in enumerate(model_configs):
        ax = fig.add_subplot(gs[0, col])
        si = stat_idx["In-deg entropy"]
        hist_panel(ax, synth_stats[key][:, si], real_stats[si],
                   pvals[key][si], "In-deg entropy (bits)", color,
                   label.replace("\n", " "))

    # Row 0 col 3: in-degree CCDF
    ax_indeg = fig.add_subplot(gs[0, 3])
    k_in, p_in = ccdf(in_deg_real)
    if len(k_in) > 0:
        ax_indeg.loglog(k_in, p_in, "o", ms=3, color="steelblue", alpha=0.7, label="Real")
        mask = k_in >= 3
        if mask.sum() > 5:
            slope, intercept, *_ = stats.linregress(np.log(k_in[mask]), np.log(p_in[mask]))
            ax_indeg.loglog(k_in[mask], np.exp(intercept) * k_in[mask] ** slope,
                            "--", color="crimson", lw=1.5, label=f"PL fit γ≈{-slope:.2f}")
            
    for _, key, color in model_configs:
        sample_g = (bianconi_barabasi(n_real, m=BB_M) if key == "BB"
                    else copying_model(n_real, beta=CP_BETA) if key == "CP"
                    else lfr_benchmark(n_real, avg_deg=avg_deg, max_deg=max_deg))
        ki, pi = ccdf(np.array(sample_g.indegree()))
        if len(ki) > 0:
            ax_indeg.loglog(ki, pi, "-", lw=1, alpha=0.5, color=color, label=key)
            
    ax_indeg.set_title("In-degree CCDF\n(log-log)", fontsize=8)
    ax_indeg.set_xlabel("In-degree k", fontsize=7)
    ax_indeg.set_ylabel("P(K ≥ k)", fontsize=7)
    ax_indeg.legend(fontsize=7)

    # Row 1: Clustering
    for col, (label, key, color) in enumerate(model_configs):
        ax = fig.add_subplot(gs[1, col])
        si = stat_idx["Clustering"]
        hist_panel(ax, synth_stats[key][:, si], real_stats[si],
                   pvals[key][si], "Clustering coefficient", color,
                   label.replace("\n", " "))

    # Row 1 col 3: out-degree CCDF
    ax_outdeg = fig.add_subplot(gs[1, 3])
    k_out, p_out = ccdf(out_deg_real)
    if len(k_out) > 0:
        ax_outdeg.loglog(k_out, p_out, "o", ms=3, color="seagreen", alpha=0.7, label="Real")
        mask2 = k_out >= 5
        if mask2.sum() > 5:
            slope2, intercept2, *_ = stats.linregress(np.log(k_out[mask2]), np.log(p_out[mask2]))
            ax_outdeg.loglog(k_out[mask2], np.exp(intercept2) * k_out[mask2] ** slope2,
                             "--", color="crimson", lw=1.5, label=f"PL fit γ≈{-slope2:.2f}")
    ax_outdeg.set_title("Out-degree CCDF\n(log-log)", fontsize=8)
    ax_outdeg.set_xlabel("Out-degree k", fontsize=7)
    ax_outdeg.set_ylabel("P(K ≥ k)", fontsize=7)
    ax_outdeg.legend(fontsize=7)

    # Row 2: Out-degree entropy
    for col, (label, key, color) in enumerate(model_configs):
        ax = fig.add_subplot(gs[2, col])
        si = stat_idx["Out-deg entropy"]
        hist_panel(ax, synth_stats[key][:, si], real_stats[si],
                   pvals[key][si], "Out-deg entropy (bits)", color,
                   label.replace("\n", " "))

    # Row 2 col 3: Reciprocity
    ax_recip = fig.add_subplot(gs[2, 3])
    for _, key, color in model_configs:
        si = stat_idx["Reciprocity"]
        ax_recip.hist(synth_stats[key][:, si], bins=25, alpha=0.5,
                      color=color, label=key, edgecolor="white")
    ax_recip.axvline(real_stats[stat_idx["Reciprocity"]], color="crimson", lw=2, ls="--", label="Real")
    ax_recip.set_title("Reciprocity – all models", fontsize=8)
    ax_recip.set_xlabel("Reciprocity", fontsize=7)
    ax_recip.legend(fontsize=7)

    # Row 3: Scatter
    for col, (label, key, color) in enumerate(model_configs):
        ax = fig.add_subplot(gs[3, col])
        ax.scatter(synth_stats[key][:, 0], synth_stats[key][:, 1],
                   alpha=0.4, color=color, s=18, label=f"{key} samples")
        ax.scatter([real_stats[0]], [real_stats[1]],
                   color="crimson", s=100, zorder=5, marker="*", label="Real")
        ax.set_xlabel("Clustering", fontsize=7)
        ax.set_ylabel("In-deg entropy", fontsize=7)
        ax.set_title(f"{label.replace(chr(10),' ')} – scatter", fontsize=8)
        ax.legend(fontsize=7)

    # Row 3 col 3: p-value table
    ax_tbl = fig.add_subplot(gs[3, 3])
    ax_tbl.axis("off")
    header = ["Model", "Clust", "In-H", "Out-H", "Recip", "Verdict"]
    rows   = [header]
    for label, key, _ in model_configs:
        ps    = pvals[key]
        n_sig = sum(p < ALPHA for p in ps)
        v     = "COMPAT ✓" if n_sig == 0 else ("INCOMPAT ✗" if n_sig == 4 else "PARTIAL")
        rows.append([
            label.replace("\n", " "),
            f"{ps[0]:.3f}{'*' if ps[0]<ALPHA else ''}",
            f"{ps[1]:.3f}{'*' if ps[1]<ALPHA else ''}",
            f"{ps[2]:.3f}{'*' if ps[2]<ALPHA else ''}",
            f"{ps[3]:.3f}{'*' if ps[3]<ALPHA else ''}",
            v,
        ])
    tbl = ax_tbl.table(cellText=rows, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.1, 1.55)
    ax_tbl.set_title("p-value summary\n(* = p < 0.05)", pad=8, fontsize=8)

    plt.savefig("formation_model_test.png", dpi=150, bbox_inches="tight")
    print("    Saved → formation_model_test.png")
    
    # plt.show() # Uncomment if you want the window to pop up natively

    print("\n[Done] All tests complete.")

# Required wrapper for multiprocessing in Windows/Python
if __name__ == '__main__':
    # Important: Freeze support ensures multiprocess works correctly if wrapped in exes
    multiprocessing.freeze_support()
    main()