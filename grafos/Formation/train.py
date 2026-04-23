"""
train.py  (v2 — distribution-aware loss)
======================================================
Optimizes parameters for BA, Bianconi-Barabasi, and 5 theoretical models.
Uses Optuna (Bayesian Optimization) with a correct parallelization strategy:

  ✓ Models run IN PARALLEL across processes (parallelism is across studies)
  ✓ Each study runs SEQUENTIALLY inside its process (TPE/CMA-ES stay coherent)
  ✓ Adaptive GRAPHS_PER_TEST: cheap early exploration, precise late exploitation
  ✓ Per-model sampler: CMA-ES for continuous spaces, TPE(multivariate) for mixed
  ✓ QMC warm-start: first N trials are space-filling Sobol before Bayesian kicks in
  ✓ Empirical priors: search bounds and seed trials informed by fitted power-law stats

Progress display strategy:
  Workers NEVER write to stdout — they only put (model_idx, trial, best_loss) onto
  a shared Manager Queue. The main process owns all 7 tqdm bars (one per position)
  and is the only process that refreshes them.

PyPI degree distribution (Tabla 2.1):
  In-degree  : power law NOT rejected (KS p=0.532)  α=1.8999, x_min=3,  n_tail=1646 (24.3%)
  Out-degree : power law REJECTED     (KS p=0.000)  α=2.5158, x_min=11, n_tail=1001 (22.0%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 LOSS FUNCTION v2 — WHY THE OLD ONE FAILED AND HOW WE FIX IT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 OLD LOSS  (NMSE on 4 scalars):
   denom = np.where(real_stats != 0, real_stats, 1.0)
   loss  = sum(((avg_stats - real_stats) / denom) ** 2)

 Bug 1 — TINY DENOMINATOR (reciprocity poisoning):
   PyPI reciprocity ≈ 0.01.  A prediction of 0.05 gives error
   ((0.05 - 0.01) / 0.01)^2 = 16.0, while a completely wrong
   out-degree entropy gives only ≈ 0.22.  The optimizer sacrifices
   the entire network structure just to nail the tiny reciprocity.

 Bug 2 — ENTROPY COMPRESSION:
   Reducing the full power-law degree distribution to a single
   Shannon entropy scalar hides the heavy tail completely.
   Two distributions with the same entropy can look completely
   different (one uniform, one highly skewed).

 NEW LOSS (distribution-aware, weighted absolute):

   [A] Wasserstein-1 distance on raw in/out degree sequences.
       Forces the optimizer to match the full distribution shape
       including the power-law tail, not a compressed scalar.
       ─ Rubner, Tomasi & Guibas (2000). "The Earth Mover's Distance
         as a Metric for Image Retrieval." IJCV 40(2):99–121.
       ─ You, Ying, Ren, Hamilton & Leskovec (2018). "GraphRNN:
         Generating Realistic Graphs with Deep Auto-regressive Models."
         ICML 2018 (arXiv:1802.08773). Uses degree/clustering/orbit
         distribution distances as the standard graph-evaluation protocol.

   [B] KS-statistic on in/out degree distributions (logged only).
       The KS statistic measures the maximum CDF deviation, making
       it especially sensitive to tail mismatches in power-law nets.
       ─ Clauset, Shalizi & Newman (2009). "Power-Law Distributions
         in Empirical Data." SIAM Review 51(4):661–703.
         (Standard test used to confirm power-law degree fits.)

   [C] Absolute difference (not relative) for scalar stats.
       Eliminates the 1/0.01 amplification that breaks optimization
       for low-reciprocity networks like PyPI.
       ─ Leskovec, Chakrabarti, Kleinberg, Faloutsos & Ghahramani
         (2010). "Kronecker Graphs: An Approach to Modeling Networks."
         JMLR 11:985–1042.  Uses absolute differences of graph
         statistics as the primary goodness-of-fit measure.

   Weighted loss:
     L = w_in  · W₁(in_deg_synth,  in_deg_real)
       + w_out · W₁(out_deg_synth, out_deg_real)
       + w_c   · |clust_synth  - clust_real|
       + w_r   · |recip_synth  - recip_real|

   Wasserstein distances for PyPI O(10k) sequences typically sit in
   [0.5, 15].  |clust| and |recip| live in [0, 1].  We scale the
   scalar metrics by ×10 so they contribute equally to the landscape.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import time
import warnings
import json
import multiprocessing
import numpy as np
import igraph as ig
import optuna
from optuna.samplers import CmaEsSampler, TPESampler, QMCSampler, BaseSampler
from scipy.stats import wasserstein_distance, ks_2samp
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from bb_model import bianconi_barabasi_model
from copying_model import copying_model
from sbm_pa_model import sbm_pa_model
from ergm_model import ergm_model
from bter_model import bter_model
from kronecker_model import kronecker_model

# ── Global config ────────────────────────────────────────────────────────────
GRAPH_FILE       = "pypi_multiseed_10k.graphml"
OUTPUT_FILE      = "best_parameters.json"
TRIALS_PER_MODEL = 300
WARMUP_TRIALS    = 20
MODEL_NAMES      = ["BA", "Bianconi_BB", "Copying", "SBM_PA", "ERGM", "BTER", "Kronecker"]

# ── Empirical priors ──────────────────────────────────────────────────────────
ALPHA_IN  = 1.8999
ALPHA_OUT = 2.5158
XMIN_IN   = 3
XMIN_OUT  = 11
ALPHA_LO  = 1.2   # widened: allow shallower power-law tails
ALPHA_HI  = 3.5   # widened: α_out=2.5158 was outside old ceiling of 2.4

SEED_TRIALS = {
    "BA":          {"m": 5},
    "Bianconi_BB": {"m": 5},
    "Copying":     {"beta": 0.65, "m_init": 5},
    "SBM_PA":      {"k": 20, "alpha": ALPHA_IN, "m1": 3, "m2": 5},
    "ERGM":        {"theta_mut": -3.0, "theta_out": 1.0, "theta_in": 1.0, "theta_tri": 0.0},  # negative: suppresses mutual edges (PyPI recip≈0.01)
    "BTER":        {"alpha": ALPHA_IN, "density": 0.25},
    "Kronecker":   {"a": 0.72, "b": 0.18, "c": 0.18},
}

EVAL_SCHEDULE = {
    "early": {"threshold": 0.33, "n_graphs": 3},
    "mid":   {"threshold": 0.66, "n_graphs": 5},
    "late":  {"threshold": 1.00, "n_graphs": 8},
}

# ── Loss weights ──────────────────────────────────────────────────────────────
# Wasserstein distances on PyPI degree seqs are typically in [0.5, 15].
# Clustering and reciprocity live in [0, 1], so we multiply them by 10
# to ensure they contribute equally rather than being swamped by the
# degree-distribution terms.
# Ref: You et al. (2018) arXiv:1802.08773 use equal-weight multi-stat evaluation.
LOSS_WEIGHTS = {
    "in_wasserstein":  1.0,   # W₁ on in-degree sequences
    "out_wasserstein": 1.0,   # W₁ on out-degree sequences
    "clustering":     10.0,   # |clust_synth - clust_real|  ×10 to balance scale
    "reciprocity":    0.1,   # |recip_synth - recip_real|  ×10 to balance scale
}

def _graphs_for_trial(trial_number: int, n_trials: int) -> int:
    frac = trial_number / n_trials
    for stage in EVAL_SCHEDULE.values():
        if frac <= stage["threshold"]:
            return stage["n_graphs"]
    return EVAL_SCHEDULE["late"]["n_graphs"]

# ── Statistics ───────────────────────────────────────────────────────────────
def compute_graph_fingerprint(g: ig.Graph) -> dict:
    """
    Return a rich fingerprint of g:
      - raw in/out degree arrays (for Wasserstein + KS comparison)
      - global clustering coefficient
      - reciprocity

    The degree arrays are kept raw (not histogrammed, not entropy-compressed)
    so that wasserstein_distance() can compare the full distributional shape
    including the power-law tail.

    Ref: You et al. (2018) ICML — degree/clustering/orbit *distributions*
         (not scalars) are the gold-standard evaluation protocol for
         generative graph models.
    """
    in_deg  = np.array(g.indegree(),  dtype=np.float64)
    out_deg = np.array(g.outdegree(), dtype=np.float64)
    clust   = float(
        g.as_undirected(combine_edges="first").transitivity_undirected(mode="zero")
    )
    recip   = float(g.reciprocity())
    return {"in_deg": in_deg, "out_deg": out_deg, "clust": clust, "recip": recip}


def compute_loss_from_fingerprints(fp_list: list, real_fp: dict) -> tuple[float, list]:
    """
    Compute a weighted composite loss between a list of synthetic graph
    fingerprints and the real graph fingerprint.

    Components
    ----------
    [A] Wasserstein-1 distance on in/out degree sequences.
        Measures how much "work" it takes to transform the synthetic degree
        distribution into the real one — sensitive to the heavy power-law tail.
        ─ Rubner, Tomasi & Guibas (2000). IJCV 40(2):99–121.
        ─ You et al. (2018). GraphRNN. ICML. arXiv:1802.08773.

    [B] KS statistic on degree sequences (stored as user_attr, NOT in loss).
        Maximum CDF deviation; particularly sensitive to tail mismatches.
        ─ Clauset, Shalizi & Newman (2009). SIAM Review 51(4):661–703.

    [C] |clustering_synth - clustering_real|  (absolute, not relative).
        Avoids dividing by a small denominator.
        ─ Leskovec et al. (2010). Kronecker Graphs. JMLR 11:985–1042.

    [D] |reciprocity_synth - reciprocity_real|  (absolute, not relative).
        PyPI reciprocity ≈ 0.01; dividing by it (old code) blew the gradient
        up by 100× and made the optimizer fixate on reciprocity alone.
        ─ Leskovec et al. (2010). JMLR 11:985–1042.

    Returns
    -------
    loss  : scalar weighted sum (lower = better)
    stats : [in_wass, out_wass, clust_diff, recip_diff, ks_in, ks_out]
            for logging / JSON output
    """
    w = LOSS_WEIGHTS

    in_wass_list,  out_wass_list  = [], []
    clust_diff_list, recip_diff_list = [], []
    ks_in_list,    ks_out_list    = [], []

    for fp in fp_list:
        # ── [A] Wasserstein-1 on raw degree sequences ──────────────────────
        in_wass  = wasserstein_distance(fp["in_deg"],  real_fp["in_deg"])
        out_wass = wasserstein_distance(fp["out_deg"], real_fp["out_deg"])

        # ── [B] KS statistic (diagnostic only — NOT added to loss) ─────────
        ks_in,  _ = ks_2samp(fp["in_deg"],  real_fp["in_deg"])
        ks_out, _ = ks_2samp(fp["out_deg"], real_fp["out_deg"])

        # ── [C/D] Absolute scalar differences ──────────────────────────────
        clust_diff = abs(fp["clust"] - real_fp["clust"])
        recip_diff = abs(fp["recip"] - real_fp["recip"])

        in_wass_list.append(in_wass);   out_wass_list.append(out_wass)
        clust_diff_list.append(clust_diff); recip_diff_list.append(recip_diff)
        ks_in_list.append(ks_in);       ks_out_list.append(ks_out)

    avg_in_wass   = float(np.mean(in_wass_list))
    avg_out_wass  = float(np.mean(out_wass_list))
    avg_clust     = float(np.mean(clust_diff_list))
    avg_recip     = float(np.mean(recip_diff_list))
    avg_ks_in     = float(np.mean(ks_in_list))
    avg_ks_out    = float(np.mean(ks_out_list))

    loss = (
        w["in_wasserstein"]  * avg_in_wass  +
        w["out_wasserstein"] * avg_out_wass +
        w["clustering"]      * avg_clust    +
        w["reciprocity"]     * avg_recip
    )

    stats = [avg_in_wass, avg_out_wass, avg_clust, avg_recip, avg_ks_in, avg_ks_out]
    return float(loss), stats


# ── Core evaluation ──────────────────────────────────────────────────────────
def evaluate_params(m_name, params, n_real, m_real,
                    real_in_deg, real_out_deg, real_clust, real_recip,
                    n_graphs=5):
    """
    Generate n_graphs synthetic graphs and compute the composite loss.

    real_in_deg / real_out_deg are passed as numpy arrays so that
    wasserstein_distance() can compare sequences directly — no entropy
    compression, no histogram binning choice.
    """
    real_fp = {
        "in_deg":  real_in_deg,
        "out_deg": real_out_deg,
        "clust":   real_clust,
        "recip":   real_recip,
    }

    fp_list = []
    for _ in range(n_graphs):
        try:
            if m_name == "BA":
                g = ig.Graph.Barabasi(n=n_real, m=params["m"], directed=True)
            elif m_name == "Bianconi_BB":
                g = bianconi_barabasi_model(n_real, m_real, params["m"])
            elif m_name == "Copying":
                g = copying_model(n_real, m_real, params["beta"], params["m_init"])
            elif m_name == "SBM_PA":
                g = sbm_pa_model(n_real, params["k"], params["alpha"],
                                 params["m1"], params["m2"])
            elif m_name == "ERGM":
                g = ergm_model(n_real, m_real, params["theta_mut"], params["theta_out"], params["theta_in"], params["theta_tri"])
            elif m_name == "BTER":
                g = bter_model(n_real, m_real, params["alpha"], params["density"])
            elif m_name == "Kronecker":
                g = kronecker_model(n_real, m_real, params["a"], params["b"], params["c"])

            fp = compute_graph_fingerprint(g)
            # Guard against degenerate graphs (all-zero degrees, NaN clust, etc.)
            if (fp["in_deg"].sum() > 0 and fp["out_deg"].sum() > 0
                    and not np.isnan(fp["clust"]) and not np.isnan(fp["recip"])):
                fp_list.append(fp)
        except Exception:
            pass

    if not fp_list:
        return float("inf"), []

    return compute_loss_from_fingerprints(fp_list, real_fp)


# ── Per-model sampler selection ───────────────────────────────────────────────
def _make_sampler(m_name: str, warmup: int) -> BaseSampler:
    continuous_models = {"BTER", "ERGM", "Kronecker", "Copying"}
    if m_name in continuous_models:
        inner = CmaEsSampler(warn_independent_sampling=False, restart_strategy="ipop")
    else:
        inner = TPESampler(multivariate=True, n_startup_trials=warmup, seed=42)
    return QMCSampler(qmc_type="sobol", scramble=True, seed=42,
                      independent_sampler=inner)


# ── Worker ────────────────────────────────────────────────────────────────────
# Queue message format:
#   progress update : (model_idx, m_name, trial_number, best_loss_so_far)
#   done sentinel   : (model_idx, m_name, None, final_loss)
def optimize_model(args):
    (m_name, model_idx, n_real, m_real,
     real_in_list, real_out_list, real_clust, real_recip,
     n_trials, warmup, q) = args

    # Reconstruct numpy arrays from the serialised lists
    real_in_deg  = np.array(real_in_list,  dtype=np.float64)
    real_out_deg = np.array(real_out_list, dtype=np.float64)

    sampler = _make_sampler(m_name, warmup)
    study   = optuna.create_study(direction="minimize", sampler=sampler)

    if m_name in SEED_TRIALS:
        study.enqueue_trial(SEED_TRIALS[m_name])

    best_loss_seen = float("inf")
    loss_history   = []

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_loss_seen

        if m_name == "BA":
            params = {"m": trial.suggest_int("m", 1, 25)}   # widened: avg_deg may exceed 10
        elif m_name == "Bianconi_BB":
            params = {"m": trial.suggest_int("m", 1, 25)}   # widened: same reasoning
        elif m_name == "Copying":
            params = {
                "beta":   trial.suggest_float("beta",   0.01, 0.99),   # widened: allow near-pure copying or near-pure preferential attachment
                "m_init": trial.suggest_int(  "m_init", 2,   20),      # widened: m_init ceiling raised
            }
        elif m_name == "SBM_PA":
            params = {
                "k":     trial.suggest_int(  "k",     2,        80),           # widened: more communities allowed
                "alpha": trial.suggest_float("alpha", ALPHA_LO, ALPHA_HI),
                "m1":    trial.suggest_int(  "m1",    1,        15),           # widened
                "m2":    trial.suggest_int(  "m2",    1,        15),           # widened, also lowered floor to 1
            }
        elif m_name == "ERGM":
            params = {
                "theta_mut":  trial.suggest_float("theta_mut",  -10.0, 5.0),  # widened neg: PyPI recip≈0.01 needs strong suppression
                "theta_out":  trial.suggest_float("theta_out",  -10.0, 5.0),  # widened both sides
                "theta_in":   trial.suggest_float("theta_in",   -10.0, 5.0),  # widened both sides
                "theta_tri":  trial.suggest_float("theta_tri",   -5.0, 5.0),  # widened both sides
            }
        elif m_name == "BTER":
            params = {
                "alpha":   trial.suggest_float("alpha",   ALPHA_LO, ALPHA_HI),
                "density": trial.suggest_float("density", 0.01,     0.99),
            }
        elif m_name == "Kronecker":
            params = {
                "a": trial.suggest_float("a", 0.3,  0.95),   # widened: allow stronger diagonal dominance
                "b": trial.suggest_float("b", 0.01, 0.5),    # widened: off-diagonal may need more range
                "c": trial.suggest_float("c", 0.01, 0.5),    # widened: same
            }

        n_graphs = _graphs_for_trial(trial.number, n_trials)
        loss, stats = evaluate_params(
            m_name, params, n_real, m_real,
            real_in_deg, real_out_deg, real_clust, real_recip,
            n_graphs,
        )

        if loss == float("inf"):
            q.put((model_idx, m_name, trial.number, best_loss_seen))
            raise optuna.TrialPruned()

        # Store full stat vector for the best-trial JSON entry:
        # [in_wass, out_wass, clust_diff, recip_diff, ks_in, ks_out]
        trial.set_user_attr("stats",    stats)
        trial.set_user_attr("n_graphs", n_graphs)

        if loss < best_loss_seen:
            best_loss_seen = loss

        loss_history.append((trial.number, best_loss_seen))
        q.put((model_idx, m_name, trial.number, best_loss_seen))
        return loss

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    try:
        best_loss   = study.best_value
        best_params = study.best_params
        best_stats  = study.best_trial.user_attrs.get("stats", [])
    except ValueError:
        best_loss, best_params, best_stats = float("inf"), {}, []

    q.put((model_idx, m_name, None, best_loss))
    return m_name, {"model": m_name, "params": best_params,
                    "loss": best_loss, "stats": best_stats}, loss_history


# ── Plot convergence curves ───────────────────────────────────────────────────
def plot_convergence(all_histories: dict, output_path: str = "training_progress.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[!] matplotlib not installed — skipping convergence plot.")
        return

    colours = ["#E63946", "#2196F3", "#4CAF50", "#FF9800",
               "#9C27B0", "#00BCD4", "#FF5722"]

    fig, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0D1117")
    fig.suptitle("Bayesian Optimisation — Convergence per Model",
                 color="white", fontsize=14, fontweight="bold", y=1.01)

    for ax in (ax_abs, ax_norm):
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#8B949E")
        ax.spines[:].set_color("#30363D")
        ax.xaxis.label.set_color("#8B949E")
        ax.yaxis.label.set_color("#8B949E")
        ax.title.set_color("white")

    for (m_name, history), colour in zip(all_histories.items(), colours):
        if not history:
            continue
        trials = [h[0] for h in history]
        losses = [h[1] for h in history]
        ax_abs.plot(trials, losses, label=m_name, color=colour, linewidth=1.8, alpha=0.9)
        first = losses[0] if losses[0] != 0 else 1.0
        ax_norm.plot(trials, [l / first for l in losses],
                     label=m_name, color=colour, linewidth=1.8, alpha=0.9)

    ax_abs.set_yscale("log")
    ax_abs.set_title("Best Loss (log scale)")
    ax_abs.set_xlabel("Trial")
    ax_abs.set_ylabel("Loss  [W₁(in)+W₁(out)+10·|clust|+10·|recip|]")
    ax_abs.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax_abs.legend(fontsize=8, framealpha=0.2, labelcolor="white",
                  facecolor="#0D1117", edgecolor="#30363D")
    ax_abs.grid(axis="y", color="#30363D", linestyle="--", linewidth=0.6)

    ax_norm.set_title("Relative Improvement  (loss / loss₀)")
    ax_norm.set_xlabel("Trial")
    ax_norm.set_ylabel("Normalised loss")
    ax_norm.legend(fontsize=8, framealpha=0.2, labelcolor="white",
                   facecolor="#0D1117", edgecolor="#30363D")
    ax_norm.grid(axis="y", color="#30363D", linestyle="--", linewidth=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[+] Convergence plot saved → {output_path}")


# ── Pretty final table ────────────────────────────────────────────────────────
def print_results_table(best_results: dict):
    ranked = sorted(best_results.items(), key=lambda kv: kv[1]["loss"])
    col_w  = {"rank": 4, "model": 14, "loss": 10, "params": 52}
    sep    = (f"  +{'-'*(col_w['rank']+2)}+{'-'*(col_w['model']+2)}"
              f"+{'-'*(col_w['loss']+2)}+{'-'*(col_w['params']+2)}+")

    print("\n" + "=" * 65)
    print("  Final Results  (ranked by composite Wasserstein+|Δclust|+|Δrecip| loss)")
    print("=" * 65)
    print(sep)
    print(f"  | {'#':<{col_w['rank']}} | {'Model':<{col_w['model']}} "
          f"| {'Loss':<{col_w['loss']}} | {'Best Params':<{col_w['params']}} |")
    print(sep)
    for rank, (m_name, r) in enumerate(ranked, 1):
        params_str = json.dumps(r["params"])
        if len(params_str) > col_w["params"]:
            params_str = params_str[:col_w["params"] - 1] + "…"
        loss_str = f"{r['loss']:.5f}" if r["loss"] != float("inf") else "∞"
        print(f"  | {rank:<{col_w['rank']}} | {m_name:<{col_w['model']}} "
              f"| {loss_str:<{col_w['loss']}} | {params_str:<{col_w['params']}} |")
    print(sep)
    print("\n  Stats columns: [W₁_in, W₁_out, |Δclust|, |Δrecip|, KS_in, KS_out]")
    for m_name, r in ranked:
        if r["stats"]:
            labels = ["W1_in", "W1_out", "|Δclust|", "|Δrecip|", "KS_in", "KS_out"]
            vals   = "  ".join(f"{l}={v:.4f}" for l, v in zip(labels, r["stats"]))
            print(f"  {m_name:<14}: {vals}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  PyPI Network: 7-Model Parallel Bayesian Optimizer  [v2]")
    print("  Loss: W₁(in/out-deg) + 10·|Δclust| + 10·|Δrecip|")
    print("=" * 65)
    print(f"  Priors: α_in={ALPHA_IN} (p=0.532, not rejected) | "
          f"α_out={ALPHA_OUT} (p=0.000, rejected)")
    print(f"  Alpha search range: [{ALPHA_LO}, {ALPHA_HI}]")

    try:
        real_g = ig.Graph.Read_GraphML(GRAPH_FILE)
        if not real_g.is_directed():
            real_g = real_g.as_directed(mode="mutual")
    except Exception as e:
        print(f"Failed to load {GRAPH_FILE}: {e}")
        return

    n_real = real_g.vcount()
    m_real = real_g.ecount()

    # Compute real graph fingerprint once; pass degree arrays to workers.
    # Keeping raw arrays avoids entropy compression and lets each worker
    # call wasserstein_distance() directly.
    # Ref: You et al. (2018) arXiv:1802.08773 — distribution-level evaluation.
    real_fp       = compute_graph_fingerprint(real_g)
    real_in_list  = real_fp["in_deg"].tolist()
    real_out_list = real_fp["out_deg"].tolist()
    real_clust    = real_fp["clust"]
    real_recip    = real_fp["recip"]

    avg_deg = m_real / n_real
    print(f"[+] Loaded PyPI: {n_real} nodes, {m_real} edges | avg degree {avg_deg:.1f}")
    print(f"[+] Real stats: clust={real_clust:.4f}  recip={real_recip:.4f}")
    print(f"[+] {TRIALS_PER_MODEL} trials/model | {WARMUP_TRIALS} QMC warm-up | "
          f"adaptive eval budget | empirical seed trials")
    print(f"[+] Running all 7 models in parallel — progress bars below:\n")

    manager = multiprocessing.Manager()
    q       = manager.Queue()

    tasks = [
        (m_name, idx, n_real, m_real,
         real_in_list, real_out_list, real_clust, real_recip,
         TRIALS_PER_MODEL, WARMUP_TRIALS, q)
        for idx, m_name in enumerate(MODEL_NAMES)
    ]

    bars = [
        tqdm(
            total=TRIALS_PER_MODEL,
            desc=f"{name:<13}",
            position=i,
            leave=True,
            dynamic_ncols=True,
            postfix={"best": "∞"},
        )
        for i, name in enumerate(MODEL_NAMES)
    ]

    n_workers  = min(len(MODEL_NAMES), multiprocessing.cpu_count())
    done_count = 0

    with multiprocessing.Pool(processes=n_workers) as pool:
        async_result = pool.map_async(optimize_model, tasks)

        while done_count < len(MODEL_NAMES):
            while not q.empty():
                try:
                    model_idx, m_name, trial_num, best_loss = q.get_nowait()
                except Exception:
                    break

                bar = bars[model_idx]

                if trial_num is None:
                    bar.n = TRIALS_PER_MODEL
                    loss_str = f"{best_loss:.4f}" if best_loss != float("inf") else "∞"
                    bar.set_postfix({"best": loss_str, "done": "✓"})
                    bar.refresh()
                    done_count += 1
                else:
                    bar.n = trial_num + 1
                    if best_loss != float("inf"):
                        bar.set_postfix({"best": f"{best_loss:.4f}"})
                    bar.refresh()

            time.sleep(0.05)

    for bar in bars:
        bar.close()

    raw_results = async_result.get()

    best_results  = {}
    all_histories = {}
    for m_name, result, history in raw_results:
        best_results[m_name]  = result
        all_histories[m_name] = history

    with open(OUTPUT_FILE, "w") as f:
        json.dump(best_results, f, indent=4)

    print_results_table(best_results)
    print(f"\n[+] Best parameters saved → {OUTPUT_FILE}")
    plot_convergence(all_histories, output_path="training_progress.png")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()