"""
train_network_model.py
======================================================
Trains the SBM+PA model on the real PyPI network using 
Randomized Search to find the optimal hyperparameters.
"""

import random
import warnings
import multiprocessing
import concurrent.futures
import numpy as np
import igraph as ig

warnings.filterwarnings("ignore")

# ── configuration ─────────────────────────────────────────────────────────────
GRAPH_FILE = "pypi_multiseed_10k.graphml"
SEARCH_ITERATIONS = 200   # Number of different parameter combinations to try
GRAPHS_PER_TEST   = 5     # Graphs to generate per parameter test (keep low for speed)

# Define the bounds for our search space
PARAM_BOUNDS = {
    "n_comm":  (5, 50),         # Number of communities
    "m_edges": (2, 8),          # Average edges added per new node
    "mu":      (0.01, 0.40),    # Probability to link outside community
    "p_recip": (0.001, 0.05)    # Probability of mutual back-link
}
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(g: ig.Graph) -> np.ndarray:
    """Computes [Clustering, In-Entropy, Out-Entropy, Reciprocity]"""
    # Clustering
    clust = g.as_undirected(combine_edges="first").transitivity_undirected(mode="zero")
    
    # In-Entropy
    in_deg = np.array(g.indegree())
    in_counts = np.bincount(in_deg)
    in_probs = in_counts[in_counts > 0] / len(in_deg)
    in_ent = float(-np.sum(in_probs * np.log2(in_probs)))
    
    # Out-Entropy
    out_deg = np.array(g.outdegree())
    out_counts = np.bincount(out_deg)
    out_probs = out_counts[out_counts > 0] / len(out_deg)
    out_ent = float(-np.sum(out_probs * np.log2(out_probs)))
    
    # Reciprocity
    recip = g.reciprocity()
    
    return np.array([clust, in_ent, out_ent, recip])

def sbm_pa_model(n: int, m: int, n_comm: int, mu: float, p_recip: float) -> ig.Graph:
    """The fast generative model."""
    rng = np.random.default_rng()
    edges = []
    in_deg = np.zeros(n, dtype=float)
    
    comms = np.arange(n) % n_comm
    rng.shuffle(comms)
    
    core_size = max(3, m)
    for c in range(n_comm):
        core_nodes = np.where(comms == c)[0][:core_size]
        for u in core_nodes:
            for v in core_nodes:
                if u != v:
                    edges.append((int(u), int(v)))
                    in_deg[v] += 1
                    
    start_idx = core_size * n_comm
    
    for t in range(start_idx, n):
        c_t = comms[t]
        actual_m = max(1, int(rng.normal(m, 2))) 
        
        for _ in range(actual_m):
            if rng.random() > mu:
                mask = (comms[:t] == c_t)
            else:
                mask = (comms[:t] != c_t)
                
            candidates = np.where(mask)[0]
            if len(candidates) == 0:
                candidates = np.arange(t)
                
            w = in_deg[candidates] + 1.0
            prob = w / w.sum()
            target = rng.choice(candidates, p=prob)
            
            edges.append((t, int(target)))
            in_deg[target] += 1
            
            if rng.random() < p_recip:
                edges.append((int(target), t))
                in_deg[t] += 1

    return ig.Graph(n=n, edges=edges, directed=True)

def evaluate_parameters(args):
    """Generates graphs for a specific parameter set and returns the Loss."""
    n_comm, m_edges, mu, p_recip, n_real, real_stats = args
    
    synth_results = []
    for _ in range(GRAPHS_PER_TEST):
        g = sbm_pa_model(n_real, m_edges, n_comm, mu, p_recip)
        synth_results.append(compute_stats(g))
        
    avg_synth_stats = np.mean(synth_results, axis=0)
    
    # Calculate the Loss Function (Normalized Mean Squared Error)
    # We normalize by the real stats so large numbers (entropy) don't overpower small numbers (reciprocity)
    loss = np.sum(((avg_synth_stats - real_stats) / real_stats) ** 2)
    
    return {
        "params": {"n_comm": n_comm, "m_edges": m_edges, "mu": mu, "p_recip": p_recip},
        "stats": avg_synth_stats,
        "loss": loss
    }

def main():
    print("=" * 65)
    print("  PyPI Network: SBM+PA Parameter Training Optimizer")
    print("=" * 65)

    print(f"\n[1] Loading '{GRAPH_FILE}' to extract target metrics…")
    try:
        real_g = ig.Graph.Read_GraphML(GRAPH_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find '{GRAPH_FILE}'.")
        return

    if not real_g.is_directed():
        real_g = real_g.as_directed(mode="mutual")
        
    n_real = real_g.vcount()
    real_stats = compute_stats(real_g)
    
    print(f"    Target Clustering : {real_stats[0]:.4f}")
    print(f"    Target In-Entropy : {real_stats[1]:.4f}")
    print(f"    Target Out-Entropy: {real_stats[2]:.4f}")
    print(f"    Target Reciprocity: {real_stats[3]:.4f}")

    print(f"\n[2] Generating {SEARCH_ITERATIONS} random parameter combinations...")
    rng = np.random.default_rng(42)
    search_space = []
    for _ in range(SEARCH_ITERATIONS):
        search_space.append((
            int(rng.integers(*PARAM_BOUNDS["n_comm"])),
            int(rng.integers(*PARAM_BOUNDS["m_edges"])),
            float(rng.uniform(*PARAM_BOUNDS["mu"])),
            float(rng.uniform(*PARAM_BOUNDS["p_recip"])),
            n_real,
            real_stats
        ))

    print(f"[3] Training! Running on {multiprocessing.cpu_count()} CPU cores...")
    print("    (This will take some time depending on SEARCH_ITERATIONS)\n")
    
    best_loss = float('inf')
    best_params = None
    best_stats = None

    # Run the parameter evaluation in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {executor.submit(evaluate_parameters, args): args for args in search_space}
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            result = future.result()
            
            if result["loss"] < best_loss:
                best_loss = result["loss"]
                best_params = result["params"]
                best_stats = result["stats"]
                print(f"🌟 NEW BEST! (Iteration {completed}/{SEARCH_ITERATIONS}) Loss: {best_loss:.4f}")
                print(f"   Params: Comm={best_params['n_comm']}, M={best_params['m_edges']}, "
                      f"Mu={best_params['mu']:.3f}, Recip={best_params['p_recip']:.3f}")
            elif completed % 20 == 0:
                print(f"    ... {completed}/{SEARCH_ITERATIONS} configurations tested.")

    print("\n" + "=" * 65)
    print("🎉 TRAINING COMPLETE! BEST PARAMETERS FOUND:")
    print("=" * 65)
    print(f"SBM_COMMUNITIES = {best_params['n_comm']}")
    print(f"SBM_M_EDGES     = {best_params['m_edges']}")
    print(f"SBM_MU          = {best_params['mu']:.4f}")
    print(f"SBM_RECIPROCITY = {best_params['p_recip']:.4f}")
    print("-" * 65)
    print("How closely they matched reality:")
    print(f"    Clustering : Real {real_stats[0]:.4f} -> Model {best_stats[0]:.4f}")
    print(f"    In-Entropy : Real {real_stats[1]:.4f} -> Model {best_stats[1]:.4f}")
    print(f"    Out-Entropy: Real {real_stats[2]:.4f} -> Model {best_stats[2]:.4f}")
    print(f"    Reciprocity: Real {real_stats[3]:.4f} -> Model {best_stats[3]:.4f}")
    print("=" * 65)
    print("Plug these numbers into the top of your previous sbm_pa_test.py script to run the final P-Value validation!")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()