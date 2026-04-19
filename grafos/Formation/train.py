import warnings
import json
import multiprocessing
import concurrent.futures
import numpy as np
import igraph as ig

warnings.filterwarnings("ignore")

GRAPH_FILE = "pypi_multiseed_10k.graphml"
SEARCH_ITERATIONS = 250    # Random parameter combos per model
GRAPHS_PER_TEST   = 8      # Graphs to average per test
OUTPUT_FILE       = "best_parameters.json"

# ══════════════════════════════════════════════════════════════════════════════
# STATS & LOSS
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(g: ig.Graph) -> np.ndarray:
    clust = g.as_undirected(combine_edges="first").transitivity_undirected(mode="zero")
    
    in_deg = np.array(g.indegree())
    in_counts = np.bincount(in_deg)
    in_probs = in_counts[in_counts > 0] / len(in_deg)
    in_ent = float(-np.sum(in_probs * np.log2(in_probs)))
    
    out_deg = np.array(g.outdegree())
    out_counts = np.bincount(out_deg)
    out_probs = out_counts[out_counts > 0] / len(out_deg)
    out_ent = float(-np.sum(out_probs * np.log2(out_probs)))
    
    return np.array([clust, in_ent, out_ent, g.reciprocity()])

def copying_model(n, m_real, beta, m_init):
    rng = np.random.default_rng()
    edges = [(i, i + 1) for i in range(m_init - 1)]
    adj_out = [[] for _ in range(n)]
    for u, v in edges: adj_out[u].append(v)
    for t in range(m_init, n):
        proto = rng.integers(0, t)
        targets = set()
        for w in adj_out[proto]:
            if w != t and rng.random() < beta: targets.add(w)
        targets.add(rng.integers(0, t))
        for w in targets:
            edges.append((t, w))
            adj_out[t].append(w)
    return ig.Graph(n=n, edges=edges, directed=True)

def sbm_pa_model(n, m_real, m, n_comm, mu, p_recip):
    rng = np.random.default_rng()
    edges, in_deg = [], np.zeros(n, dtype=float)
    comms = np.arange(n) % n_comm
    rng.shuffle(comms)
    core_size = max(3, m)
    for c in range(n_comm):
        core_nodes = np.where(comms == c)[0][:core_size]
        for u in core_nodes:
            for v in core_nodes:
                if u != v:
                    edges.append((int(u), int(v))); in_deg[v] += 1
    for t in range(core_size * n_comm, n):
        c_t, actual_m = comms[t], max(1, int(rng.normal(m, 2))) 
        for _ in range(actual_m):
            mask = (comms[:t] == c_t) if rng.random() > mu else (comms[:t] != c_t)
            candidates = np.where(mask)[0]
            if len(candidates) == 0: candidates = np.arange(t)
            target = rng.choice(candidates, p=(in_deg[candidates] + 1.0)/sum(in_deg[candidates] + 1.0))
            edges.append((t, int(target))); in_deg[target] += 1
            if rng.random() < p_recip: edges.append((int(target), t)); in_deg[t] += 1
    return ig.Graph(n=n, edges=edges, directed=True)

def ergm_model(n, m_real, theta_mut, theta_in_star):
    """Fast MCMC ERGM approximation optimizing for Reciprocity and Preferential Attachment."""
    rng = np.random.default_rng()
    sources, targets = rng.integers(0, n, m_real), rng.integers(0, n, m_real)
    edges = set((s, t) for s, t in zip(sources, targets) if s != t)
    
    in_deg = np.zeros(n, dtype=int)
    for u, v in edges: in_deg[v] += 1
    
    edge_list = list(edges)
    for _ in range(30000): # MCMC Burn-in
        if not edge_list: break
        idx = rng.integers(0, len(edge_list))
        u, v = edge_list[idx]
        new_u, new_v = int(rng.integers(0, n)), int(rng.integers(0, n))
        
        if new_u == new_v or (new_u, new_v) in edges: continue
        
        # Calculate log-odds change for network statistics
        mut_diff = ((new_v, new_u) in edges) - ((v, u) in edges)
        star_diff = in_deg[new_v] - (in_deg[v] - 1)
        
        delta = (theta_mut * mut_diff) + (theta_in_star * star_diff)
        
        # Metropolis-Hastings acceptance step
        if delta >= 0 or rng.random() < np.exp(delta):
            edges.remove((u, v))
            edges.add((new_u, new_v))
            edge_list[idx] = (new_u, new_v)
            in_deg[v] -= 1; in_deg[new_v] += 1
            
    return ig.Graph(n=n, edges=list(edges), directed=True)

def bter_model(n, m_real, alpha, block_density):
    """Approximation of the Block Two-Level Erdős-Rényi (BTER) Model."""
    rng = np.random.default_rng()
    sizes = []
    while sum(sizes) < n:
        s = int(rng.pareto(alpha) + 1)
        if sum(sizes) + s > n: s = n - sum(sizes)
        sizes.append(s)
    
    edges, offset = [], 0
    # Phase 1: High-density power-law blocks
    for s in sizes:
        if s > 1:
            for u in range(s):
                for v in range(s):
                    if u != v and rng.random() < block_density:
                        edges.append((offset+u, offset+v))
        offset += s
        
    # Phase 2: Fill remaining edges to meet target M
    current_m = len(edges)
    if current_m < m_real:
        edges.extend(list(zip(rng.integers(0, n, m_real - current_m), rng.integers(0, n, m_real - current_m))))
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

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION CORE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_params(args):
    model_name, params, n_real, m_real, real_stats = args
    synth_results = []
    
    for _ in range(GRAPHS_PER_TEST):
        if model_name == "Copying":
            g = copying_model(n_real, m_real, params["beta"], params["m_init"])
        elif model_name == "Hybrid":
            g = hybrid_model(n_real, m_real, params["m_init"], params["p_copy"])
        elif model_name == "SBM_PA":
            g = sbm_pa_model(n_real, m_real, params["m"], params["n_comm"], params["mu"], params["p_recip"])
        elif model_name == "ERGM":
            g = ergm_model(n_real, m_real, params["theta_mut"], params["theta_star"])
        elif model_name == "BTER":
            g = bter_model(n_real, m_real, params["alpha"], params["density"])
        elif model_name == "Kronecker":
            g = kronecker_model(n_real, m_real, params["a"], params["b"], params["c"])
            
        synth_results.append(compute_stats(g))
        
    avg_stats = np.mean(synth_results, axis=0)
    loss = np.sum(((avg_stats - real_stats) / real_stats) ** 2)
    return {"model": model_name, "params": params, "loss": loss, "stats": avg_stats.tolist()}

def main():
    print("=" * 60)
    print("  PyPI Network: The 6-Model Master Optimizer")
    print("=" * 60)

    try:
        real_g = ig.Graph.Read_GraphML(GRAPH_FILE)
        if not real_g.is_directed(): real_g = real_g.as_directed(mode="mutual")
    except Exception as e:
        print(f"Failed to load {GRAPH_FILE}: {e}")
        return

    n_real, m_real = real_g.vcount(), real_g.ecount()
    real_stats = compute_stats(real_g)
    print(f"[+] Loaded PyPI: {n_real} nodes, {m_real} edges. Building search space...")

    rng = np.random.default_rng(42)
    tasks = []
    
    for _ in range(SEARCH_ITERATIONS):
        tasks.append(("Copying", {"beta": float(rng.uniform(0.1, 0.9)), "m_init": int(rng.integers(2, 10))}, n_real, m_real, real_stats))
        tasks.append(("Hybrid", {"p_copy": float(rng.uniform(0.1, 0.9)), "m_init": int(rng.integers(2, 10))}, n_real, m_real, real_stats))
        tasks.append(("SBM_PA", {"m": int(rng.integers(2, 10)), "n_comm": int(rng.integers(5, 40)), "mu": float(rng.uniform(0.01, 0.3)), "p_recip": float(rng.uniform(0.001, 0.05))}, n_real, m_real, real_stats))
        # ERGM: Thetas control the log-odds multipliers
        tasks.append(("ERGM", {"theta_mut": float(rng.uniform(0.1, 5.0)), "theta_star": float(rng.uniform(0.001, 0.1))}, n_real, m_real, real_stats))
        # BTER: Alpha controls power-law slope, density controls triangle clusters
        tasks.append(("BTER", {"alpha": float(rng.uniform(1.1, 3.0)), "density": float(rng.uniform(0.1, 0.9))}, n_real, m_real, real_stats))
        # Kronecker: Matrix probabilities must sum <= 1.0. 
        a, b, c = float(rng.uniform(0.4, 0.8)), float(rng.uniform(0.05, 0.3)), float(rng.uniform(0.05, 0.3))
        tasks.append(("Kronecker", {"a": a, "b": b, "c": c}, n_real, m_real, real_stats))

    best_results = {m: {"loss": float('inf')} for m in ["Copying", "Hybrid", "SBM_PA", "ERGM", "BTER", "Kronecker"]}

    print(f"[+] Training on {multiprocessing.cpu_count()} cores ({len(tasks)} configs total)...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        completed = 0
        for result in executor.map(evaluate_params, tasks):
            completed += 1
            m_name = result["model"]
            if result["loss"] < best_results[m_name]["loss"]:
                best_results[m_name] = result
                print(f"  🌟 New Best [{m_name:<9}] Loss: {result['loss']:.4f}")
            if completed % 50 == 0:
                print(f"  ... {completed}/{len(tasks)} tested")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(best_results, f, indent=4)
        
    print(f"\n[+] Done! Best parameters saved to {OUTPUT_FILE}.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()