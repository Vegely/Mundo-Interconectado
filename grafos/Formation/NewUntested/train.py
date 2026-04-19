"""
train.py
======================================================
Optimizes parameters for BA, Bianconi-Barabasi, and 5 theoretical models.
Models are imported from the compiled Cython extensions.
"""
import warnings
import json
import multiprocessing
import concurrent.futures
import numpy as np
import igraph as ig
from tqdm import tqdm

from bb_model import bianconi_barabasi_model
from copying_model import copying_model
from sbm_pa_model import sbm_pa_model
from ergm_model import ergm_model
from bter_model import bter_model
from kronecker_model import kronecker_model

warnings.filterwarnings("ignore")

GRAPH_FILE = "pypi_multiseed_10k.graphml"
SEARCH_ITERATIONS = 200
GRAPHS_PER_TEST   = 5
OUTPUT_FILE       = "best_parameters.json"

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

    recip = g.reciprocity()
    return np.array([clust, in_ent, out_ent, recip])


def evaluate_params(args):
    m_name, params, n_real, m_real, real_stats = args
    synth_results = []

    for _ in range(GRAPHS_PER_TEST):
        try:
            if m_name == "BA":
                g = ig.Graph.Barabasi(n=n_real, m=params["m"], directed=True)

            elif m_name == "Bianconi_BB":
                # bianconi_barabasi_model(n, m_real, m)
                g = bianconi_barabasi_model(n_real, m_real, params["m"])

            elif m_name == "Copying":
                # copying_model(n, m_real, beta, m_init)
                g = copying_model(n_real, m_real, params["beta"], params["m_init"])

            elif m_name == "SBM_PA":
                # sbm_pa_model(n, k, alpha, m1, m2)
                # k     : number of communities          (int,   5–40)
                # alpha : power-law exponent for sizes   (float, 1.1–3.0)
                # m1    : max inter-community edges/node (int,   1–7)
                # m2    : intra-community seed size      (int,   2–9)
                # Returns a directed graph directly — do NOT call as_directed().
                g = sbm_pa_model(n_real,
                                 params["k"],
                                 params["alpha"],
                                 params["m1"],
                                 params["m2"])

            elif m_name == "ERGM":
                # ergm_model(n, m_real, theta_mut, theta_star)
                g = ergm_model(n_real, m_real, params["theta_mut"], params["theta_star"])

            elif m_name == "BTER":
                # bter_model(n, m_real, alpha, block_density)
                g = bter_model(n_real, m_real, params["alpha"], params["density"])

            else:  # Kronecker
                # kronecker_model(n, m_real, a, b, c)
                g = kronecker_model(n_real, m_real, params["a"], params["b"], params["c"])

            synth_results.append(compute_stats(g))
        except Exception:
            # Skip any degenerate graph (e.g. zero edges produced by Kronecker)
            pass

    if not synth_results:
        return {"model": m_name, "params": params, "loss": float("inf"), "stats": []}

    avg_stats = np.mean(synth_results, axis=0)
    # Avoid division by zero if a real_stats component is 0
    denom = np.where(real_stats != 0, real_stats, 1.0)
    loss  = float(np.sum(((avg_stats - real_stats) / denom) ** 2))
    return {"model": m_name, "params": params, "loss": loss, "stats": avg_stats.tolist()}


def main():
    print("=" * 65)
    print("  PyPI Network: 7-Model Master Optimizer")
    print("=" * 65)

    try:
        real_g = ig.Graph.Read_GraphML(GRAPH_FILE)
        if not real_g.is_directed():
            real_g = real_g.as_directed(mode="mutual")
    except Exception as e:
        print(f"Failed to load {GRAPH_FILE}: {e}")
        return

    n_real, m_real = real_g.vcount(), real_g.ecount()
    real_stats = compute_stats(real_g)
    print(f"[+] Loaded PyPI: {n_real} nodes, {m_real} edges. Building search space...")

    rng   = np.random.default_rng(42)
    tasks = []

    for _ in range(SEARCH_ITERATIONS):

        # ── BA ──────────────────────────────────────────────────────────────
        tasks.append(("BA",
                       {"m": int(rng.integers(1, 10))},
                       n_real, m_real, real_stats))

        # ── Bianconi-Barabasi ────────────────────────────────────────────────
        # bianconi_barabasi_model(n, m_real, m)
        tasks.append(("Bianconi_BB",
                       {"m": int(rng.integers(1, 10))},
                       n_real, m_real, real_stats))

        # ── Copying ──────────────────────────────────────────────────────────
        # copying_model(n, m_real, beta, m_init)
        tasks.append(("Copying",
                       {"beta":   float(rng.uniform(0.1, 0.9)),
                        "m_init": int(rng.integers(2, 10))},
                       n_real, m_real, real_stats))

        # ── SBM_PA ───────────────────────────────────────────────────────────
        # sbm_pa_model(n, k, alpha, m1, m2)
        #   k     : number of communities          — integer in [5, 40]
        #   alpha : power-law exponent for sizes   — float  in [1.1, 3.0]
        #   m1    : max inter-community edges/node — integer in [1, 7]
        #   m2    : intra-community seed size      — integer in [2, 9]
        # NOTE: m_real is NOT a parameter of sbm_pa_model; edge count is
        #       controlled internally through m1/m2.
        # NOTE: returns undirected graph; .as_directed() called in evaluate_params.
        tasks.append(("SBM_PA",
                       {"k":    int(rng.integers(5, 40)),
                        "alpha": float(rng.uniform(1.1, 3.0)),
                        "m1":   int(rng.integers(1, 8)),
                        "m2":   int(rng.integers(2, 10))},
                       n_real, m_real, real_stats))

        # ── ERGM ─────────────────────────────────────────────────────────────
        # ergm_model(n, m_real, theta_mut, theta_star)
        tasks.append(("ERGM",
                       {"theta_mut":  float(rng.uniform(0.1, 5.0)),
                        "theta_star": float(rng.uniform(0.001, 0.1))},
                       n_real, m_real, real_stats))

        # ── BTER ─────────────────────────────────────────────────────────────
        # bter_model(n, m_real, alpha, block_density)
        tasks.append(("BTER",
                       {"alpha":   float(rng.uniform(1.1, 3.0)),
                        "density": float(rng.uniform(0.1, 0.9))},
                       n_real, m_real, real_stats))

        # ── Kronecker ────────────────────────────────────────────────────────
        # kronecker_model(n, m_real, a, b, c)
        a = float(rng.uniform(0.4, 0.8))
        b = float(rng.uniform(0.05, 0.3))
        c = float(rng.uniform(0.05, 0.3))
        tasks.append(("Kronecker",
                       {"a": a, "b": b, "c": c},
                       n_real, m_real, real_stats))

    model_names  = ["BA", "Bianconi_BB", "Copying", "SBM_PA", "ERGM", "BTER", "Kronecker"]
    best_results = {m: {"loss": float("inf")} for m in model_names}

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=multiprocessing.cpu_count()) as executor:
        
        # Wrap executor.map with tqdm to generate a progress bar
        for result in tqdm(executor.map(evaluate_params, tasks), total=len(tasks), desc="Optimizing"):
            m_name = result["model"]
            if result["loss"] < best_results[m_name]["loss"]:
                best_results[m_name] = result
                # Use tqdm.write instead of print so it doesn't break the progress bar visually
                tqdm.write(f"  🌟 New Best [{m_name:<11}] Loss: {result['loss']:.4f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(best_results, f, indent=4)

    print(f"\n[+] Done! Best parameters saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()