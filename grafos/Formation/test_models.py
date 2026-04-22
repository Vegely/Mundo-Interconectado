"""
test_models.py
======================================================
Validates 8 Models. Models are imported from compiled Cython extensions.
Computes p-values using intuitive physical metrics (Max Hub, Assortativity)
and plots distance errors using Wasserstein distances.

Fixed: Windows multiprocessing serialization for global variables.
"""
import json
import warnings
import multiprocessing
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.stats import wasserstein_distance

from bb_model import bianconi_barabasi_model
from copying_model import copying_model
from sbm_pa_model import sbm_pa_model
from ergm_model import ergm_model
from bter_model import bter_model
from kronecker_model import kronecker_model

warnings.filterwarnings("ignore")

GRAPH_FILE  = "pypi_multiseed_10k.graphml"
INPUT_FILE  = "best_parameters.json"
N_MC        = 300   # Validation samples
ALPHA       = 0.05

# Names for the 4 plotted columns in the 32-panel grid
STAT_NAMES  = ["Clustering", "W1 In-Deg", "W1 Out-Deg", "Reciprocity"]
MODEL_ORDER = ["ER (Null)", "BA (Null)", "Bianconi_BB", "Copying", "SBM_PA", "ERGM", "BTER", "Kronecker"]


def get_ccdf(degrees: np.ndarray):
    """Calculates the Complementary Cumulative Distribution Function (Log-Log suitable)."""
    if len(degrees) == 0:
        return np.array([]), np.array([])
    k, counts = np.unique(degrees, return_counts=True)
    idx = np.argsort(k)
    k = k[idx]
    counts = counts[idx]
    ccdf = np.cumsum(counts[::-1])[::-1] / len(degrees)
    return k, ccdf


def compute_stats(g: ig.Graph, real_in: np.ndarray, real_out: np.ndarray) -> np.ndarray:
    """
    Computes 6 stats total.
    0-3 are used for visual plotting (Wasserstein shape matching).
    4-5 (plus 0 and 3) are used for strict p-value checking.
    """
    # 0. Clustering
    clust = g.as_undirected(combine_edges="first").transitivity_undirected(mode="zero")
    if np.isnan(clust): clust = 0.0
    
    in_deg = np.array(g.indegree())
    out_deg = np.array(g.outdegree())
    
    # 1 & 2. Wasserstein Distances (Lower is better, 0.0 = perfect match)
    w1_in = wasserstein_distance(in_deg, real_in) if len(in_deg) > 0 else float('inf')
    w1_out = wasserstein_distance(out_deg, real_out) if len(out_deg) > 0 else float('inf')
    
    # 3. Reciprocity
    recip = g.reciprocity()
    if np.isnan(recip): recip = 0.0
    
    # 4. Max Hub (Extreme Value Bounds)
    max_in = float(np.max(in_deg)) if len(in_deg) > 0 else 0.0
    
    # 5. Assortativity (Internal Wiring Logic)
    assort = g.assortativity_degree(directed=True)
    if np.isnan(assort): assort = 0.0
        
    return np.array([clust, w1_in, w1_out, recip, max_in, assort])


def two_sided_pval(sample: np.ndarray, observed: float) -> float:
    """Calculates the empirical two-sided p-value."""
    n = len(sample)
    if n == 0: return 0.0
    leq = int(np.sum(sample <= observed))
    return 2 * min(leq + 1, n + 1 - leq + 1) / (n + 1)


def _build_graph(m_name, params, n_real, m_real):
    if m_name == "ER (Null)":
        return ig.Graph.Erdos_Renyi(n=n_real, m=m_real, directed=True)
    elif m_name == "BA (Null)" or m_name == "BA":
        return ig.Graph.Barabasi(n=n_real, m=int(params.get("m", 3)), directed=True)
    elif m_name == "Bianconi_BB":
        return bianconi_barabasi_model(n_real, m_real, int(params["m"]))
    elif m_name == "Copying":
        return copying_model(n_real, m_real, params["beta"], int(params["m_init"]))
    elif m_name == "SBM_PA":
        return sbm_pa_model(n_real, int(params["k"]), params["alpha"], int(params["m1"]), int(params["m2"]))
    elif m_name == "ERGM":
        return ergm_model(n_real, m_real, params["theta_mut"], params["theta_out"], params["theta_in"], params.get("theta_tri", 0.0))
    elif m_name == "BTER":
        return bter_model(n_real, m_real, params["alpha"], params["density"])
    elif m_name == "Kronecker":
        return kronecker_model(n_real, m_real, params["a"], params["b"], params["c"], params.get("d", -1.0))
    return ig.Graph(n_real, directed=True)


def worker_run(args):
    # Unpack the real_in and real_out arrays explicitely sent to the worker
    m_name, params, n_real, m_real, real_in, real_out = args
    try:
        g = _build_graph(m_name, params, n_real, m_real)
        stats = compute_stats(g, real_in, real_out)
        return stats
    except Exception as e:
        return None


def plot_individual_summary(m_name, c, results_dict, best_configs, n_real, m_real, k_real, p_real, k_out_real, p_out_real, real_stats):
    """Generates the highly detailed 6-panel figure for a single model."""
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"{m_name} - Detail View", fontsize=16, y=0.96)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.4)

    def _hist(ax, stat_idx, title):
        ax.hist(results_dict[m_name][:, stat_idx], bins=25, color=c, alpha=0.6, edgecolor='white')
        # Clamp dashed line to 0.0 if it's a Wasserstein metric
        target_val = 0.0 if stat_idx in [1, 2] else real_stats[stat_idx]
        ax.axvline(target_val, color='crimson', ls='--', lw=2, label=f"Target: {target_val:.3f}")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    # [0,0] Clustering
    _hist(fig.add_subplot(gs[0, 0]), 0, "Clustering Coefficient")

    # [0,1] Scatter: Clust vs W1-In
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(results_dict[m_name][:, 0], results_dict[m_name][:, 1], alpha=0.5, color=c, s=15)
    ax2.plot(real_stats[0], 0.0, 'r*', ms=12, label="Target (Reality)")
    ax2.set_xlabel("Clustering")
    ax2.set_ylabel("W1 In-Degree Error")
    ax2.set_title("Clustering vs W1 In-Degree Error", fontsize=10)
    ax2.legend(fontsize=8)

    # [1,0] W1 Out-Degree Error
    _hist(fig.add_subplot(gs[1, 0]), 2, "W1 Out-Degree Error")

    # [1,1] Reciprocity
    _hist(fig.add_subplot(gs[1, 1]), 3, "Reciprocity")

    # [0,2] In-Degree CCDF (Log-Log)
    ax5 = fig.add_subplot(gs[0, 2])
    if len(k_real) > 0:
        ax5.loglog(k_real, p_real, "o", ms=4, color="steelblue", label="Real In-deg")
    try:
        params = best_configs[m_name]["params"]
        g_rep  = _build_graph(m_name, params, n_real, m_real)
        k_rep, p_rep = get_ccdf(np.array(g_rep.indegree()))
        if len(k_rep) > 0:
            ax5.loglog(k_rep, p_rep, "-", color=c, lw=2, label=f"{m_name}")
    except Exception as e:
        pass
    ax5.set_title("In-degree CCDF (log-log)", fontsize=10)
    ax5.legend(fontsize=8)

    # [1,2] Out-Degree CCDF (Log-Log)
    ax6 = fig.add_subplot(gs[1, 2])
    if len(k_out_real) > 0:
        ax6.loglog(k_out_real, p_out_real, "o", ms=4, color="steelblue", label="Real Out-deg")
    try:
        k_out_rep, p_out_rep = get_ccdf(np.array(g_rep.outdegree()))
        if len(k_out_rep) > 0:
            ax6.loglog(k_out_rep, p_out_rep, "-", color=c, lw=2, label=f"{m_name}")
    except Exception as e:
        pass
    ax6.set_title("Out-degree CCDF (log-log)", fontsize=10)
    ax6.legend(fontsize=8)

    safe_name = m_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"summary_{safe_name}.png", bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == "__main__":
    print("[+] Loading real graph...")
    real_g = ig.read(GRAPH_FILE)
    n_real = real_g.vcount()
    m_real = real_g.ecount()

    # Empirical Distributions
    REAL_IN_DEGREES = np.array(real_g.indegree())
    REAL_OUT_DEGREES = np.array(real_g.outdegree())
    
    # Calculate real target stats
    real_clust = real_g.as_undirected(combine_edges="first").transitivity_undirected(mode="zero")
    if np.isnan(real_clust): real_clust = 0.0
    real_recip = real_g.reciprocity()
    if np.isnan(real_recip): real_recip = 0.0
    
    real_max_in = float(np.max(REAL_IN_DEGREES)) if len(REAL_IN_DEGREES) > 0 else 0.0
    real_assort = real_g.assortativity_degree(directed=True)
    if np.isnan(real_assort): real_assort = 0.0

    # Indices: 0=Clust, 1=W1_In, 2=W1_Out, 3=Recip, 4=Max_In, 5=Assort
    real_stats = np.array([real_clust, 0.0, 0.0, real_recip, real_max_in, real_assort])

    k_in_real, p_in_real = get_ccdf(REAL_IN_DEGREES)
    k_out_real, p_out_real = get_ccdf(REAL_OUT_DEGREES)

    print("[+] Loading optimized parameters...")
    with open(INPUT_FILE, "r") as f:
        best_configs = json.load(f)
        
    best_configs["ER (Null)"] = {"params": {}}
    if "BA" in best_configs:
        best_configs["BA (Null)"] = best_configs.pop("BA")
    elif "BA (Null)" not in best_configs:
        best_configs["BA (Null)"] = {"params": {"m": 3}}

    print(f"\n[+] Running parallel validation ({N_MC} samples per model)...")
    results = {m: [] for m in MODEL_ORDER}
    
    for m_name in MODEL_ORDER:
        if m_name not in best_configs:
            continue
            
        params = best_configs[m_name]["params"]
        
        # Explicitly pack REAL_IN_DEGREES and REAL_OUT_DEGREES for Windows compatibility
        tasks = [(m_name, params, n_real, m_real, REAL_IN_DEGREES, REAL_OUT_DEGREES) for _ in range(N_MC)]
        
        valid_stats = []
        with multiprocessing.Pool() as pool:
            for res in tqdm(pool.imap_unordered(worker_run, tasks), total=N_MC, desc=f"{m_name[:12]:<14}"):
                if res is not None:
                    valid_stats.append(res)
                    
        results[m_name] = np.array(valid_stats)
        
    print("\n[+] Valid samples per model:")
    for m_name in MODEL_ORDER:
        if len(results[m_name]) > 0:
            print(f"    {m_name:<14} {len(results[m_name])}/{N_MC}")

    print("\n[+] Final P-Value Results (Using Intuitive Mechanics Metrics):")
    print("-" * 85)
    for m_name in MODEL_ORDER:
        if len(results[m_name]) == 0: continue
        
        # P-values pull: Clust (0), Max In-Deg (4), Assortativity (5), Recip (3)
        pval_indices = [0, 4, 5, 3]
        ps = [two_sided_pval(results[m_name][:, i], real_stats[i]) for i in pval_indices]
        
        p_strs = [f"{p:.3f}*" if p < ALPHA else f"{p:.3f} " for p in ps]
        print(f" {m_name:<14} | Clust: {p_strs[0]} | Max Hub: {p_strs[1]} | Assort: {p_strs[2]} | Recip: {p_strs[3]}")

    print("\n[+] Plotting 32-Panel Master Grid...")
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("Network Formation: 8-Model Validation", fontsize=22, y=0.92)
    gs = gridspec.GridSpec(len(MODEL_ORDER), 4, figure=fig, hspace=0.4, wspace=0.3)
    
    colors = {"ER (Null)": "#bdc3c7", "BA (Null)": "#95a5a6", "Bianconi_BB": "#9b59b6", 
              "Copying": "#3498db", "SBM_PA": "#2ecc71", "ERGM": "#e74c3c", 
              "BTER": "#f39c12", "Kronecker": "#34495e"}
              
    for i, m_name in enumerate(MODEL_ORDER):
        if len(results[m_name]) == 0: continue
        c = colors.get(m_name, "#333333")
        
        # Plot only indices 0, 1, 2, 3 (Clust, W1-In, W1-Out, Recip)
        for j in range(4):
            ax = fig.add_subplot(gs[i, j])
            ax.hist(results[m_name][:, j], bins=30, color=c, alpha=0.7, edgecolor='white')
            
            target_val = 0.0 if j in [1, 2] else real_stats[j]
            ax.axvline(target_val, color='crimson', linestyle='dashed', linewidth=2)
            
            if i == 0:
                ax.set_title(STAT_NAMES[j], fontsize=14, pad=10)
            if j == 0:
                ax.set_ylabel(m_name, fontsize=12, fontweight='bold', labelpad=10)
                
    plt.savefig("final_8model_master_grid.png", bbox_inches='tight', dpi=150)
    print("[+] Saved final_8model_master_grid.png!")

    print("\n[+] Plotting Individual 6-Panel Summaries...")
    for m_name in MODEL_ORDER:
        if len(results[m_name]) == 0: continue
        c = colors.get(m_name, "#333333")
        plot_individual_summary(m_name, c, results, best_configs, n_real, m_real, k_in_real, p_in_real, k_out_real, p_out_real, real_stats)

    print("[+] All done!")