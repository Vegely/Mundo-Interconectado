"""
sbm_pa_test.py
======================================================
Tests a "Stochastic Block Model + Preferential Attachment" 
to simulate ecosystem communities with rich-get-richer dynamics.
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
N_MC       = 100    # Synthetic graphs to generate
ALPHA      = 0.05   

# Advanced Model Parameters
SBM_COMMUNITIES = 11    # Number of sub-ecosystems in PyPI (e.g., ML, Web, Crypto)
SBM_M_EDGES     = 5     # Average edges a new package creates
SBM_MU          = 0.0513  # Mixing parameter: 15% chance to link outside your community
SBM_RECIPROCITY =  0.0079  # 2% chance to create a mutual back-link
# ──────────────────────────────────────────────────────────────────────────────

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
# THE ADVANCED SBM + PA MODEL
# ══════════════════════════════════════════════════════════════════════════════

def sbm_pa_model(n: int, m: int, n_comm: int, mu: float, p_recip: float) -> ig.Graph:
    """
    Nodes belong to communities. They prefer to attach to high-degree hubs 
    *within* their own community, driving up both Clustering and Power-Laws.
    """
    rng = np.random.default_rng()
    edges = []
    in_deg = np.zeros(n, dtype=float)
    
    # Pre-assign packages to communities evenly
    comms = np.arange(n) % n_comm
    rng.shuffle(comms)
    
    # Seed each community with a tiny fully-connected core so it has gravity
    core_size = max(3, m)
    for c in range(n_comm):
        core_nodes = np.where(comms == c)[0][:core_size]
        for u in core_nodes:
            for v in core_nodes:
                if u != v:
                    edges.append((int(u), int(v)))
                    in_deg[v] += 1
                    
    start_idx = core_size * n_comm
    
    # Grow network
    for t in range(start_idx, n):
        c_t = comms[t]
        
        # Determine how many edges this node gets (adds entropy to out-degree)
        actual_m = max(1, int(rng.normal(m, 2))) 
        
        for _ in range(actual_m):
            # Decide if edge stays in-network or goes external
            if rng.random() > mu:
                mask = (comms[:t] == c_t) # Same ecosystem
            else:
                mask = (comms[:t] != c_t) # Different ecosystem
                
            candidates = np.where(mask)[0]
            if len(candidates) == 0:
                candidates = np.arange(t) # Fallback if ecosystem is empty
                
            # Preferential attachment restricted to the chosen candidate pool
            w = in_deg[candidates] + 1.0
            prob = w / w.sum()
            target = rng.choice(candidates, p=prob)
            
            edges.append((t, int(target)))
            in_deg[target] += 1
            
            # Reciprocity fix: Occasionally developers link back
            if rng.random() < p_recip:
                edges.append((int(target), t))
                in_deg[t] += 1

    return ig.Graph(n=n, edges=edges, directed=True)

def worker_run(args):
    seed, n_real = args
    np.random.seed(seed)
    random.seed(seed)
    g = sbm_pa_model(n_real, SBM_M_EDGES, SBM_COMMUNITIES, SBM_MU, SBM_RECIPROCITY)
    return compute_stats(g)

# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  PyPI Advanced Model: Communities + Preferential Attachment")
    print("=" * 65)

    try:
        real_g = ig.Graph.Read_GraphML(GRAPH_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find '{GRAPH_FILE}'.")
        return

    if not real_g.is_directed():
        real_g = real_g.as_directed(mode="mutual")
        
    n_real = real_g.vcount()
    real_stats = compute_stats(real_g)
    
    print(f"    Loaded network with {n_real:,} nodes.")
    for name, val in zip(STAT_NAMES, real_stats):
        print(f"    Real {name:15s}: {val:.6f}")

    print(f"\n[Generating {N_MC} instances of Advanced Model in parallel...]")
    base_seed = 777
    worker_args = [(base_seed + i, n_real) for i in range(N_MC)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(worker_run, worker_args))
        
    synth_stats = np.array(results)
    
    print("\n[Computing p-values...]")
    pvals = np.array([two_sided_pval(synth_stats[:, i], real_stats[i]) for i in range(len(STAT_NAMES))])

    print(f"\n{'Metric':<20} {'Real Value':>12} {'SBM+PA Mean':>15} {'p-value':>10}")
    print("-" * 60)
    for i, name in enumerate(STAT_NAMES):
        sig = "*" if pvals[i] < ALPHA else "✓"
        print(f"{name:<20} {real_stats[i]:>12.4f} {synth_stats[:, i].mean():>15.4f} {pvals[i]:>9.4f} {sig}")

    # Plotting
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("PyPI vs. SBM+PA (Communities + Rich-Get-Richer)", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    for i, name in enumerate(STAT_NAMES):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.hist(synth_stats[:, i], bins=20, color="#2ecc71", edgecolor="white", alpha=0.8, label="SBM+PA")
        ax.axvline(real_stats[i], color="crimson", lw=2, ls="--", label="Real PyPI")
        ax.set_title(f"{name}\n(p = {pvals[i]:.3f})", fontsize=10)
        ax.legend()

    ax_scat = fig.add_subplot(gs[0, 2])
    ax_scat.scatter(synth_stats[:, 0], synth_stats[:, 1], color="#2ecc71", alpha=0.5, label="SBM+PA Samples")
    ax_scat.scatter([real_stats[0]], [real_stats[1]], color="crimson", s=150, marker="*", zorder=5, label="Real PyPI")
    ax_scat.set_xlabel("Clustering", fontsize=9)
    ax_scat.set_ylabel("In-degree Entropy", fontsize=9)
    ax_scat.set_title("Clustering vs In-Entropy", fontsize=10)
    ax_scat.legend()

    ax_deg = fig.add_subplot(gs[1, 2])
    k_in, p_in = ccdf(np.array(real_g.indegree()))
    if len(k_in) > 0:
        ax_deg.loglog(k_in, p_in, "o", ms=4, color="steelblue", label="Real In-deg")
    
    sample_model = sbm_pa_model(n_real, SBM_M_EDGES, SBM_COMMUNITIES, SBM_MU, SBM_RECIPROCITY)
    hk_in, hp_in = ccdf(np.array(sample_model.indegree()))
    if len(hk_in) > 0:
        ax_deg.loglog(hk_in, hp_in, "-", lw=2, color="#2ecc71", label="SBM+PA In-deg")
        
    ax_deg.set_title("In-degree CCDF (log-log)", fontsize=10)
    ax_deg.legend()

    plt.savefig("sbm_pa_model_test.png", dpi=150, bbox_inches="tight")
    print("\n[Done] Saved → sbm_pa_model_test.png")
    
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()