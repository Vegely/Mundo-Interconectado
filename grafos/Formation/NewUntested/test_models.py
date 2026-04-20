"""
test_models.py
======================================================
Validates 8 Models. Models are imported from the compiled Cython extensions.
Computes p-values, generates an individual 6-panel PNG for each model,
and a master 32-panel grid.

Changes vs original:
  - Added tqdm progress bars pinned by position (same as train.py).
  - Switched to multiprocessing.Pool and a Manager Queue for IPC progress tracking.
  - worker_run returns None for degenerate/NaN graphs instead of crashing.
  - Main collection loop filters None results; prints per-model valid-sample count.
  - BTER call fixed: bter_model(n, m_real, alpha, density) — 4 args only.
  - compute_stats NaN guard added (consistent with train.py).
"""
import json
import time
import warnings
import multiprocessing
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from bb_model import bianconi_barabasi_model
from copying_model import copying_model
from sbm_pa_model import sbm_pa_model
from ergm_model import ergm_model
from bter_model import bter_model
from kronecker_model import kronecker_model

warnings.filterwarnings("ignore")

GRAPH_FILE  = "pypi_multiseed_10k.graphml"
INPUT_FILE  = "best_parameters.json"
N_MC        = 300   # Monte Carlo samples per model
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
        return bianconi_barabasi_model(n_real, m_real, params["m"])

    elif m_name == "Copying":
        return copying_model(n_real, m_real, params["beta"], params["m_init"])

    elif m_name == "SBM_PA":
        return sbm_pa_model(n_real, params["k"], params["alpha"],
                            params["m1"], params["m2"])

    elif m_name == "ERGM":
        return ergm_model(n_real, m_real, params["theta_mut"], params["theta_out"], params["theta_in"], params["theta_tri"])

    elif m_name == "BTER":
        return bter_model(n_real, m_real, params["alpha"], params["density"])

    else:  # Kronecker
        return kronecker_model(n_real, m_real, params["a"], params["b"], params["c"])


def worker_run(args):
    """
    Build one synthetic graph and return its stats.
    Returns (m_name, stats) on success, or (m_name, None) on failure.
    Puts the model_idx into the queue to update the progress bar.
    """
    m_name, model_idx, params, seed, n_real, m_real, q = args
    np.random.seed(seed)
    
    try:
        g     = _build_graph(m_name, params, n_real, m_real)
        stats = compute_stats(g)
        if np.any(np.isnan(stats)):
            result = (m_name, None)
        else:
            result = (m_name, stats)
    except Exception:
        result = (m_name, None)

    # Notify main process that one MC sample for this model is done
    if q is not None:
        q.put(model_idx)

    return result


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

    # Inject null models
    best_configs["ER (Null)"] = {"params": {}}
    if "BA" in best_configs and "BA (Null)" not in best_configs:
        best_configs["BA (Null)"] = best_configs.pop("BA")

    real_g = ig.Graph.Read_GraphML(GRAPH_FILE)
    if not real_g.is_directed():
        real_g = real_g.as_directed(mode="mutual")
    n_real, m_real = real_g.vcount(), real_g.ecount()
    real_stats = compute_stats(real_g)

    # Shared queue for progress bar updates
    manager = multiprocessing.Manager()
    q = manager.Queue()

    tasks = [
        (m_name, idx, best_configs[m_name]["params"], 1000 + i, n_real, m_real, q)
        for idx, m_name in enumerate(MODEL_ORDER)
        for i in range(N_MC)
    ]
    results = {m: [] for m in MODEL_ORDER}

    print(f"\n[+] Running parallel validation ({N_MC} samples per model)...\n")

    # Create 8 fixed-position progress bars in the main thread
    bars = [
        tqdm(
            total=N_MC,
            desc=f"{name:<15}",
            position=i,
            leave=True,
            dynamic_ncols=True
        )
        for i, name in enumerate(MODEL_ORDER)
    ]

    n_workers = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=n_workers) as pool:
        async_result = pool.map_async(worker_run, tasks)

        # Event loop: drain queue and refresh bars until all tasks complete
        completed = 0
        total_tasks = len(tasks)
        
        while completed < total_tasks:
            while not q.empty():
                try:
                    m_idx = q.get_nowait()
                    bars[m_idx].update(1)
                    bars[m_idx].refresh()
                    completed += 1
                except Exception:
                    break
            time.sleep(0.05)  # 50ms poll

    for bar in bars:
        bar.close()

    # Collect the results
    raw_results = async_result.get()
    for m_name, stat_arr in raw_results:
        if stat_arr is not None:
            results[m_name].append(stat_arr)

    # Report valid sample counts
    print("\n\n[+] Valid samples per model:")
    for m_name in MODEL_ORDER:
        n_valid = len(results[m_name])
        flag = "" if n_valid == N_MC else f"  ⚠ only {n_valid}/{N_MC} valid"
        print(f"    {m_name:<15} {n_valid:>3}/{N_MC}{flag}")
        results[m_name] = np.array(results[m_name])

    # Guard: skip any model with too few samples to compute p-values
    MIN_SAMPLES = 20
    runnable = [m for m in MODEL_ORDER if len(results[m]) >= MIN_SAMPLES]

    # ── P-value table ─────────────────────────────────────────────────────────
    print("\n[+] Final P-Value Results:")
    print("-" * 75)
    for m_name in MODEL_ORDER:
        if m_name not in runnable:
            print(f" {m_name:<15} | *** insufficient samples — skipped ***")
            continue
        ps = [two_sided_pval(results[m_name][:, i], real_stats[i]) for i in range(4)]
        sig = lambda p: "*" if p < ALPHA else " "
        print(f" {m_name:<15} | Clust: {ps[0]:.3f}{sig(ps[0])} | "
              f"In-Ent: {ps[1]:.3f}{sig(ps[1])} | "
              f"Out-Ent: {ps[2]:.3f}{sig(ps[2])} | "
              f"Recip: {ps[3]:.3f}{sig(ps[3])}")
    print("  (* = rejected at α=0.05)")

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
            if m_name in runnable:
                ax.hist(results[m_name][:, stat_idx], bins=20,
                        color=colors[m_name], alpha=0.8, edgecolor="white")
                ax.axvline(real_stats[stat_idx], color="crimson", lw=2, ls="--")
            else:
                ax.text(0.5, 0.5, "insufficient\nsamples",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color="gray")
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
        print(f"    -> {m_name}...")
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"PyPI vs. {m_name}", fontsize=16, y=0.95)
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

        c      = colors[m_name]
        m_data = results[m_name] if m_name in runnable else None

        def _pval(idx):
            return two_sided_pval(m_data[:, idx], real_stats[idx]) if m_data is not None else float("nan")

        p_clust, p_in_ent, p_out_ent, p_recip = (_pval(i) for i in range(4))

        def _hist(ax, idx, title):
            if m_data is not None:
                ax.hist(m_data[:, idx], bins=20, color=c, alpha=0.7,
                        edgecolor="white", label=m_name)
                ax.axvline(real_stats[idx], color="crimson", lw=2, ls="--",
                           label="Real PyPI")
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "insufficient samples", ha="center",
                        va="center", transform=ax.transAxes, color="gray")
            ax.set_title(title, fontsize=11)

        # [0,0] Clustering
        _hist(fig.add_subplot(gs[0, 0]), 0,
              f"Clustering\n(p = {p_clust:.3f})")

        # [0,1] In-Degree Entropy
        _hist(fig.add_subplot(gs[0, 1]), 1,
              f"In-deg entropy\n(p = {p_in_ent:.3f})")

        # [0,2] Scatter: Clustering vs In-Entropy
        ax2 = fig.add_subplot(gs[0, 2])
        if m_data is not None:
            ax2.scatter(m_data[:, 0], m_data[:, 1], color=c, alpha=0.5,
                        label=f"{m_name} Samples")
            ax2.scatter([real_stats[0]], [real_stats[1]], color="crimson",
                        marker="*", s=150, zorder=5, label="Real PyPI")
            ax2.legend(fontsize=8)
        ax2.set_xlabel("Clustering")
        ax2.set_ylabel("In-degree Entropy")
        ax2.set_title("Clustering vs In-Entropy", fontsize=11)

        # [1,0] Out-Degree Entropy
        _hist(fig.add_subplot(gs[1, 0]), 2,
              f"Out-deg entropy\n(p = {p_out_ent:.3f})")

        # [1,1] Reciprocity
        _hist(fig.add_subplot(gs[1, 1]), 3,
              f"Reciprocity\n(p = {p_recip:.3f})")

        # [1,2] In-Degree CCDF (Log-Log)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.loglog(k_real, p_real, "o", ms=4, color="steelblue",
                   label="Real In-deg")
        try:
            params = best_configs[m_name]["params"]
            g_rep  = _build_graph(m_name, params, n_real, m_real)
            k_rep, p_rep = get_ccdf(np.array(g_rep.indegree()))
            if len(k_rep) > 0:
                ax5.loglog(k_rep, p_rep, "-", color=c, lw=2,
                           label=f"{m_name} In-deg")
        except Exception:
            pass
        ax5.set_title("In-degree CCDF (log-log)", fontsize=11)
        ax5.legend(fontsize=8)

        clean_name = (m_name.replace(" ", "_")
                             .replace("(", "")
                             .replace(")", ""))
        plt.savefig(f"validation_{clean_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\n[+] All individual model plots saved successfully!")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()