"""
test_models.py
======================================================
Reads the 6 optimized models from train.py, runs the validation,
computes p-values, and plots the massive 24-panel comparison.
"""

import json
import warnings
import multiprocessing
import concurrent.futures
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

GRAPH_FILE = "pypi_multiseed_10k.graphml"
INPUT_FILE = "best_parameters.json"
N_MC       = 200   # Validation samples
ALPHA      = 0.05

STAT_NAMES = ["Clustering", "In-Entropy", "Out-Entropy", "Reciprocity"]
MODEL_ORDER = ["Copying", "Hybrid", "SBM_PA", "ERGM", "BTER", "Kronecker"]

def compute_stats(g: ig.Graph) -> np.ndarray:
    clust = g.as_undirected(combine_edges="first").transitivity_undirected(mode="zero")
    in_deg, out_deg = np.array(g.indegree()), np.array(g.outdegree())
    
    in_c = np.bincount(in_deg); in_p = in_c[in_c > 0] / len(in_deg)
    out_c = np.bincount(out_deg); out_p = out_c[out_c > 0] / len(out_deg)
    
    in_ent = float(-np.sum(in_p * np.log2(in_p)))
    out_ent = float(-np.sum(out_p * np.log2(out_p)))
    
    return np.array([clust, in_ent, out_ent, g.reciprocity()])

def two_sided_pval(sample: np.ndarray, observed: float) -> float:
    n = len(sample)
    leq = int(np.sum(sample <= observed))
    return 2 * min(leq + 1, n + 1 - leq + 1) / (n + 1)

# Generators (Identical logic to train.py)
def copying_model(n, m, beta, m_init):
    rng = np.random.default_rng()
    edges = [(i, i + 1) for i in range(m_init - 1)]
    adj_out = [[] for _ in range(n)]
    for u, v in edges: adj_out[u].append(v)
    for t in range(m_init, n):
        proto = rng.integers(0, t)
        targets = set([w for w in adj_out[proto] if w != t and rng.random() < beta] + [rng.integers(0, t)])
        for w in targets: edges.append((t, w)); adj_out[t].append(w)
    return ig.Graph(n=n, edges=edges, directed=True)

def hybrid_model(n, m, m_init, p_copy):
    rng = np.random.default_rng()
    edges, in_deg, adj_out = [], np.zeros(n, dtype=float), [[] for _ in range(n)]
    for i in range(m_init):
        for j in range(m_init):
            if i != j: edges.append((i, j)); in_deg[j] += 1; adj_out[i].append(j)
    for t in range(m_init, n):
        w = in_deg[:t] + 1.0 
        proto = rng.choice(t, p=w/w.sum())
        targets = set([proto] + [w for w in adj_out[proto] if w != t and rng.random() < p_copy])
        if rng.random() < 0.2: targets.add(rng.integers(0, t))
        for tgt in targets: edges.append((t, tgt)); in_deg[tgt] += 1; adj_out[t].append(tgt)
    return ig.Graph(n=n, edges=edges, directed=True)

def sbm_pa_model(n, m, m_e, n_comm, mu, p_recip):
    rng = np.random.default_rng()
    edges, in_deg = [], np.zeros(n, dtype=float)
    comms = np.arange(n) % n_comm
    rng.shuffle(comms)
    cs = max(3, m_e)
    for c in range(n_comm):
        cn = np.where(comms == c)[0][:cs]
        for u in cn:
            for v in cn:
                if u != v: edges.append((int(u), int(v))); in_deg[v] += 1
    for t in range(cs * n_comm, n):
        c_t, actual_m = comms[t], max(1, int(rng.normal(m_e, 2))) 
        for _ in range(actual_m):
            mask = (comms[:t] == c_t) if rng.random() > mu else (comms[:t] != c_t)
            cands = np.where(mask)[0]
            if len(cands) == 0: cands = np.arange(t)
            tgt = rng.choice(cands, p=(in_deg[cands] + 1.0)/sum(in_deg[cands] + 1.0))
            edges.append((t, int(tgt))); in_deg[tgt] += 1
            if rng.random() < p_recip: edges.append((int(tgt), t)); in_deg[t] += 1
    return ig.Graph(n=n, edges=edges, directed=True)

def ergm_model(n, m, theta_mut, theta_star):
    rng = np.random.default_rng()
    edges = set((s, t) for s, t in zip(rng.integers(0, n, m), rng.integers(0, n, m)) if s != t)
    in_deg = np.zeros(n, dtype=int)
    for u, v in edges: in_deg[v] += 1
    el = list(edges)
    for _ in range(30000):
        if not el: break
        idx = rng.integers(0, len(el))
        u, v = el[idx]
        nu, nv = int(rng.integers(0, n)), int(rng.integers(0, n))
        if nu == nv or (nu, nv) in edges: continue
        delta = (theta_mut * (((nv, nu) in edges) - ((v, u) in edges))) + (theta_star * (in_deg[nv] - (in_deg[v] - 1)))
        if delta >= 0 or rng.random() < np.exp(delta):
            edges.remove((u, v)); edges.add((nu, nv)); el[idx] = (nu, nv)
            in_deg[v] -= 1; in_deg[nv] += 1
    return ig.Graph(n=n, edges=list(edges), directed=True)

def bter_model(n, m, alpha, block_density):
    rng, sizes, edges, offset = np.random.default_rng(), [], [], 0
    while sum(sizes) < n:
        s = int(rng.pareto(alpha) + 1)
        sizes.append(s if sum(sizes) + s <= n else n - sum(sizes))
    for s in sizes:
        if s > 1:
            for u in range(s):
                for v in range(s):
                    if u != v and rng.random() < block_density: edges.append((offset+u, offset+v))
        offset += s
    if len(edges) < m: edges.extend(list(zip(rng.integers(0, n, m - len(edges)), rng.integers(0, n, m - len(edges)))))
    return ig.Graph(n=n, edges=edges, directed=True)

def kronecker_model(n, m_real, a, b, c):
    """Stochastic Kronecker Graph (Fractal matrix multiplication)."""
    k = int(np.ceil(np.log2(n)))
    rng = np.random.default_rng()
    
    # Ensure d doesn't go negative, and normalize the array so it strictly sums to 1.0
    d = max(0.01, 1.0 - (a + b + c)) 
    p = np.array([a, b, c, d])
    p = p / p.sum()  # <--- THIS FIXES THE CRASH
    
    edges = []
    for _ in range(m_real):
        u, v = 0, 0
        for i in range(k):
            step = 2**(k - 1 - i)
            quad = rng.choice(4, p=p)
            if quad == 1: v += step
            elif quad == 2: u += step
            elif quad == 3: u += step; v += step
        if u < n and v < n and u != v: edges.append((u, v))
        
    return ig.Graph(n=n, edges=edges, directed=True)

def worker_run(args):
    m_name, params, seed, n_real, m_real = args
    np.random.seed(seed)
    if m_name == "Copying": g = copying_model(n_real, m_real, params["beta"], params["m_init"])
    elif m_name == "Hybrid": g = hybrid_model(n_real, m_real, params["m_init"], params["p_copy"])
    elif m_name == "SBM_PA": g = sbm_pa_model(n_real, m_real, params["m"], params["n_comm"], params["mu"], params["p_recip"])
    elif m_name == "ERGM": g = ergm_model(n_real, m_real, params["theta_mut"], params["theta_star"])
    elif m_name == "BTER": g = bter_model(n_real, m_real, params["alpha"], params["density"])
    else: g = kronecker_model(n_real, m_real, params["a"], params["b"], params["c"])
    return m_name, compute_stats(g)

def main():
    print("=" * 60)
    print("  PyPI Network: The 6-Model Validation")
    print("=" * 60)

    try:
        with open(INPUT_FILE, 'r') as f: best_configs = json.load(f)
    except FileNotFoundError:
        print(f"Error: Run train.py first.")
        return

    real_g = ig.Graph.Read_GraphML(GRAPH_FILE)
    if not real_g.is_directed(): real_g = real_g.as_directed(mode="mutual")
    n_real, m_real = real_g.vcount(), real_g.ecount()
    real_stats = compute_stats(real_g)

    tasks = [(m_name, best_configs[m_name]["params"], 1000 + i, n_real, m_real) for m_name in MODEL_ORDER for i in range(N_MC)]
    results = {m: [] for m in MODEL_ORDER}
    
    print(f"[+] Running parallel validation...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for m_name, stat_arr in executor.map(worker_run, tasks):
            results[m_name].append(stat_arr)

    for k in results: results[k] = np.array(results[k])

    print("\n[+] Final P-Value Results:")
    print("-" * 75)
    for m_name in MODEL_ORDER:
        ps = [two_sided_pval(results[m_name][:, i], real_stats[i]) for i in range(4)]
        print(f" {m_name:<10} | Clust: {ps[0]:.3f} | In-Ent: {ps[1]:.3f} | Out-Ent: {ps[2]:.3f} | Recip: {ps[3]:.3f}")

    print("\n[+] Plotting 24-Panel Grid...")
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("The 6-Model Comparison vs. Real PyPI Network", fontsize=20, y=0.92)
    gs = gridspec.GridSpec(len(MODEL_ORDER), 4, figure=fig, hspace=0.4, wspace=0.3)
    
    colors = {"Copying": "#3498db", "Hybrid": "#9b59b6", "SBM_PA": "#2ecc71", 
              "ERGM": "#e74c3c", "BTER": "#f39c12", "Kronecker": "#34495e"}
    
    for stat_idx, stat_name in enumerate(STAT_NAMES):
        for row, m_name in enumerate(MODEL_ORDER):
            ax = fig.add_subplot(gs[row, stat_idx])
            ax.hist(results[m_name][:, stat_idx], bins=20, color=colors[m_name], alpha=0.7)
            ax.axvline(real_stats[stat_idx], color="crimson", lw=2, ls="--")
            if row == 0: ax.set_title(stat_name, fontsize=14)
            if stat_idx == 0: ax.set_ylabel(m_name, fontsize=12, fontweight='bold')

    plt.savefig("final_6model_validation.png", dpi=150, bbox_inches="tight")
    print("[+] Saved final_6model_validation.png!")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()