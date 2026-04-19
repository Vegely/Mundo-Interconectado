"""
test_models.py
======================================================
Validates 8 Models. Models are imported from the compiled Cython extensions.
Computes p-values, generates an individual 6-panel PNG for each model,
and a master 32-panel grid.
"""
import json
import warnings
import multiprocessing
import concurrent.futures
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from bb_model import bianconi_barabasi_model
from copying_model import copying_model
from sbm_pa_model import sbm_pa_model
from ergm_model import ergm_model
from bter_model import bter_model
from kronecker_model import kronecker_model

warnings.filterwarnings("ignore")

GRAPH_FILE  = "pypi_multiseed_10k.graphml"
INPUT_FILE  = "best_parameters.json"
N_MC        = 200   # Monte Carlo samples per model
ALPHA       = 0.05

STAT_NAMES  = ["Clustering", "In-Entropy", "Out-Entropy", "Reciprocity"]
MODEL_ORDER = ["ER (Null)", "BA (Null)", "Bianconi_BB", "Copying",
               "SBM_PA", "ERGM", "BTER", "Kronecker"]


def compute_stats(g: ig.Graph) -> np.ndarray:
    clust = g.as_undirected(combine_edges="first").transitivity_undirected(mode="zero")

    in_deg  = np.array(g.indegree())
    out_deg = np.array(g.outdegree())

    in_c  = np.bincount(in_deg)
    out_c = np.bincount(out_deg)

    in_p  = in_c[in_c > 0]   / len(in_deg)
    out_p = out_c[out_c > 0] / len(out_deg)

    in_ent  = float(-np.sum(in_p  * np.log2(in_p)))
    out_ent = float(-np.sum(out_p * np.log2(out_p)))

    return np.array([clust, in_ent, out_ent, g.reciprocity()])


def two_sided_pval(sample: np.ndarray, observed: float) -> float:
    """
    Percentile bootstrap two-sided p-value (corrected).
    Uses the standard (B+1) denominator to avoid p=0.
    """
    n   = len(sample)
    leq = int(np.sum(sample <= observed))
    return 2 * min(leq + 1, n - leq + 1) / (n + 1)


def _build_graph(m_name, params, n_real, m_real):
    if m_name == "ER (Null)":
        return ig.Graph.Erdos_Renyi(n=n_real, m=m_real, directed=True, loops=False)

    elif m_name == "BA (Null)":
        return ig.Graph.Barabasi(n=n_real, m=params["m"], directed=True)

    elif m_name == "Bianconi_BB":
        # bianconi_barabasi_model(n, m_real, m)
        return bianconi_barabasi_model(n_real, m_real, params["m"])

    elif m_name == "Copying":
        # copying_model(n, m_real, beta, m_init)
        return copying_model(n_real, m_real, params["beta"], params["m_init"])

    elif m_name == "SBM_PA":
        # sbm_pa_model(n, k, alpha, m1, m2)
        # Returns a directed graph directly — do NOT call as_directed().
        g = sbm_pa_model(n_real,
                         params["k"],
                         params["alpha"],
                         params["m1"],
                         params["m2"])
        return g

    elif m_name == "ERGM":
        # ergm_model(n, m_real, theta_mut, theta_star)
        return ergm_model(n_real, m_real, params["theta_mut"], params["theta_star"])

    elif m_name == "BTER":
        # bter_model(n, m_real, alpha, block_density)
        return bter_model(n_real, m_real, params["alpha"], params["density"])

    else:  # Kronecker
        # kronecker_model(n, m_real, a, b, c)
        return kronecker_model(n_real, m_real, params["a"], params["b"], params["c"])


def worker_run(args):
    m_name, params, seed, n_real, m_real = args
    np.random.seed(seed)
    g = _build_graph(m_name, params, n_real, m_real)
    return m_name, compute_stats(g)


def get_ccdf(degrees):
    k_vals = np.sort(np.unique(degrees[degrees > 0]))
    p_vals = np.array([np.mean(degrees >= k) for k in k_vals])
    return k_vals, p_vals


def main():
    print("=" * 65)
    print("  PyPI Network: Final 8-Model Validation Suite")
    print("=" * 65)

    try:
        with open(INPUT_FILE, "r") as f:
            best_configs = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{INPUT_FILE}' not found. Run train.py first.")
        return

    # Inject null models without mutating the original keys destructively
    best_configs["ER (Null)"] = {"params": {}}
    # Rename BA -> BA (Null) only if the key exists and BA (Null) is not already present
    if "BA" in best_configs and "BA (Null)" not in best_configs:
        best_configs["BA (Null)"] = best_configs.pop("BA")

    real_g = ig.Graph.Read_GraphML(GRAPH_FILE)
    if not real_g.is_directed():
        real_g = real_g.as_directed(mode="mutual")
    n_real, m_real = real_g.vcount(), real_g.ecount()
    real_stats = compute_stats(real_g)

    tasks = [
        (m_name, best_configs[m_name]["params"], 1000 + i, n_real, m_real)
        for m_name in MODEL_ORDER
        for i in range(N_MC)
    ]
    results = {m: [] for m in MODEL_ORDER}

    print(f"[+] Running parallel validation ({N_MC} samples per model)...")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=multiprocessing.cpu_count()) as executor:
        for m_name, stat_arr in executor.map(worker_run, tasks):
            results[m_name].append(stat_arr)

    for k in results:
        results[k] = np.array(results[k])

    # ── P-value table ─────────────────────────────────────────────────────────
    print("\n[+] Final P-Value Results:")
    print("-" * 75)
    for m_name in MODEL_ORDER:
        ps = [two_sided_pval(results[m_name][:, i], real_stats[i]) for i in range(4)]
        print(f" {m_name:<15} | Clust: {ps[0]:.3f} | In-Ent: {ps[1]:.3f} "
              f"| Out-Ent: {ps[2]:.3f} | Recip: {ps[3]:.3f}")

    # ── Master 32-panel grid ──────────────────────────────────────────────────
    print("\n[+] Plotting 32-Panel Master Grid...")
    colors = {
        "ER (Null)":   "#bdc3c7",
        "BA (Null)":   "#7f8c8d",
        "Bianconi_BB": "#8e44ad",
        "Copying":     "#3498db",
        "SBM_PA":      "#2ecc71",
        "ERGM":        "#e74c3c",
        "BTER":        "#f39c12",
        "Kronecker":   "#34495e",
    }

    fig = plt.figure(figsize=(20, 32))
    fig.suptitle("Network Formation: 8-Model Validation", fontsize=20, y=0.91)
    gs  = gridspec.GridSpec(len(MODEL_ORDER), 4, figure=fig, hspace=0.4, wspace=0.3)

    for stat_idx, stat_name in enumerate(STAT_NAMES):
        for row, m_name in enumerate(MODEL_ORDER):
            ax = fig.add_subplot(gs[row, stat_idx])
            ax.hist(results[m_name][:, stat_idx], bins=20,
                    color=colors[m_name], alpha=0.8, edgecolor="white")
            ax.axvline(real_stats[stat_idx], color="crimson", lw=2, ls="--")
            if row == 0:
                ax.set_title(stat_name, fontsize=14)
            if stat_idx == 0:
                ax.set_ylabel(m_name, fontsize=12, fontweight="bold")

    plt.savefig("final_8model_master_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[+] Saved final_8model_master_grid.png!")

    # ── Individual 6-panel plots ──────────────────────────────────────────────
    print("\n[+] Plotting Individual 6-Panel Summaries...")
    real_in_deg = np.array(real_g.indegree())
    k_real, p_real = get_ccdf(real_in_deg)

    for m_name in MODEL_ORDER:
        print(f"    -> Generating plot for {m_name}...")
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"PyPI vs. {m_name}", fontsize=16, y=0.95)
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

        c      = colors[m_name]
        m_data = results[m_name]

        p_clust   = two_sided_pval(m_data[:, 0], real_stats[0])
        p_in_ent  = two_sided_pval(m_data[:, 1], real_stats[1])
        p_out_ent = two_sided_pval(m_data[:, 2], real_stats[2])
        p_recip   = two_sided_pval(m_data[:, 3], real_stats[3])

        # [0,0] Clustering
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.hist(m_data[:, 0], bins=20, color=c, alpha=0.7,
                 edgecolor="white", label=m_name)
        ax0.axvline(real_stats[0], color="crimson", lw=2, ls="--", label="Real PyPI")
        ax0.set_title(f"Clustering\n(p = {p_clust:.3f})", fontsize=11)
        ax0.legend()

        # [0,1] In-Degree Entropy
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.hist(m_data[:, 1], bins=20, color=c, alpha=0.7,
                 edgecolor="white", label=m_name)
        ax1.axvline(real_stats[1], color="crimson", lw=2, ls="--", label="Real PyPI")
        ax1.set_title(f"In-deg entropy\n(p = {p_in_ent:.3f})", fontsize=11)
        ax1.legend()

        # [0,2] Scatter: Clustering vs In-Entropy
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.scatter(m_data[:, 0], m_data[:, 1], color=c, alpha=0.5,
                    label=f"{m_name} Samples")
        ax2.scatter([real_stats[0]], [real_stats[1]], color="crimson",
                    marker="*", s=150, zorder=5, label="Real PyPI")
        ax2.set_xlabel("Clustering")
        ax2.set_ylabel("In-degree Entropy")
        ax2.set_title("Clustering vs In-Entropy", fontsize=11)
        ax2.legend()

        # [1,0] Out-Degree Entropy
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(m_data[:, 2], bins=20, color=c, alpha=0.7,
                 edgecolor="white", label=m_name)
        ax3.axvline(real_stats[2], color="crimson", lw=2, ls="--", label="Real PyPI")
        ax3.set_title(f"Out-deg entropy\n(p = {p_out_ent:.3f})", fontsize=11)
        ax3.legend()

        # [1,1] Reciprocity
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(m_data[:, 3], bins=20, color=c, alpha=0.7,
                 edgecolor="white", label=m_name)
        ax4.axvline(real_stats[3], color="crimson", lw=2, ls="--", label="Real PyPI")
        ax4.set_title(f"Reciprocity\n(p = {p_recip:.3f})", fontsize=11)
        ax4.legend()

        # [1,2] In-Degree CCDF (Log-Log)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.loglog(k_real, p_real, "o", ms=4, color="steelblue", label="Real In-deg")

        params = best_configs[m_name]["params"]
        g_rep  = _build_graph(m_name, params, n_real, m_real)
        k_rep, p_rep = get_ccdf(np.array(g_rep.indegree()))
        ax5.loglog(k_rep, p_rep, "-", color=c, lw=2, label=f"{m_name} In-deg")
        ax5.set_title("In-degree CCDF (log-log)", fontsize=11)
        ax5.legend()

        clean_name = (m_name.replace(" ", "_")
                             .replace("(", "")
                             .replace(")", ""))
        plt.savefig(f"validation_{clean_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\n[+] All individual model plots saved successfully!")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()