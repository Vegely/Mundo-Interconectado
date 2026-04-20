"""
train.py
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
  and is the only process that refreshes them. This eliminates the jumping/collision
  problem on Windows where spawned processes have independent stdout buffers.

PyPI degree distribution (Tabla 2.1):
  In-degree  : power law NOT rejected (KS p=0.532)  α=1.8999, x_min=3,  n_tail=1646 (24.3%)
  Out-degree : power law REJECTED     (KS p=0.000)  α=2.5158, x_min=11, n_tail=1001 (22.0%)
"""
import time
import warnings
import json
import multiprocessing
import numpy as np
import igraph as ig
import optuna
from optuna.samplers import CmaEsSampler, TPESampler, QMCSampler, BaseSampler
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
ALPHA_LO  = 1.5
ALPHA_HI  = 2.4

SEED_TRIALS = {
    "BA":          {"m": 5},
    "Bianconi_BB": {"m": 5},
    "Copying":     {"beta": 0.65, "m_init": 5},
    "SBM_PA":      {"k": 20, "alpha": ALPHA_IN, "m1": 3, "m2": 5},
    "ERGM":        {"theta_mut": 2.0, "theta_star": 1.0},
    "BTER":        {"alpha": ALPHA_IN, "density": 0.25},
    "Kronecker":   {"a": 0.72, "b": 0.18, "c": 0.18},
}

EVAL_SCHEDULE = {
    "early": {"threshold": 0.33, "n_graphs": 3},
    "mid":   {"threshold": 0.66, "n_graphs": 5},
    "late":  {"threshold": 1.00, "n_graphs": 8},
}

def _graphs_for_trial(trial_number: int, n_trials: int) -> int:
    frac = trial_number / n_trials
    for stage in EVAL_SCHEDULE.values():
        if frac <= stage["threshold"]:
            return stage["n_graphs"]
    return EVAL_SCHEDULE["late"]["n_graphs"]

# ── Statistics ───────────────────────────────────────────────────────────────
def compute_stats(g: ig.Graph) -> np.ndarray:
    clust   = g.as_undirected(combine_edges="first").transitivity_undirected(mode="zero")
    in_deg  = np.array(g.indegree())
    out_deg = np.array(g.outdegree())
    in_c    = np.bincount(in_deg)
    out_c   = np.bincount(out_deg)
    in_p    = in_c[in_c > 0] / len(in_deg)
    out_p   = out_c[out_c > 0] / len(out_deg)
    in_ent  = float(-np.sum(in_p  * np.log2(in_p)))
    out_ent = float(-np.sum(out_p * np.log2(out_p)))
    recip   = g.reciprocity()
    return np.array([clust, in_ent, out_ent, recip])

# ── Core evaluation ──────────────────────────────────────────────────────────
def evaluate_params(m_name, params, n_real, m_real, real_stats, n_graphs=5):
    synth_results = []
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
                g = ergm_model(n_real, m_real, params["theta_mut"], params["theta_star"])
            elif m_name == "BTER":
                g = bter_model(n_real, m_real, params["alpha"], params["density"])
            elif m_name == "Kronecker":
                g = kronecker_model(n_real, m_real, params["a"], params["b"], params["c"])

            stats = compute_stats(g)
            if not np.any(np.isnan(stats)):
                synth_results.append(stats)
        except Exception:
            pass

    if not synth_results:
        return float("inf"), []

    avg_stats = np.mean(synth_results, axis=0)
    denom = np.where(real_stats != 0, real_stats, 1.0)
    loss  = float(np.sum(((avg_stats - real_stats) / denom) ** 2))
    return loss, avg_stats.tolist()

# ── Per-model sampler selection ───────────────────────────────────────────────
def _make_sampler(m_name: str, warmup: int) -> BaseSampler:
    continuous_models = {"BTER", "ERGM", "Kronecker", "Copying"}
    if m_name in continuous_models:
        inner = CmaEsSampler(warn_independent_sampling=False, restart_strategy="ipop")
    else:
        inner = TPESampler(multivariate=True, n_startup_trials=warmup, seed=42)
    return QMCSampler(qmc_type="sobol", scramble=True, seed=42,
                      independent_sampler=inner)

# ── Worker: NO tqdm, NO print — only queue puts ───────────────────────────────
# Queue message format:
#   progress update : (model_idx, m_name, trial_number, best_loss_so_far)
#   done sentinel   : (model_idx, m_name, None, final_loss)
def optimize_model(args):
    m_name, model_idx, n_real, m_real, real_stats_list, n_trials, warmup, q = args
    real_stats = np.array(real_stats_list)

    sampler = _make_sampler(m_name, warmup)
    study   = optuna.create_study(direction="minimize", sampler=sampler)

    if m_name in SEED_TRIALS:
        study.enqueue_trial(SEED_TRIALS[m_name])

    best_loss_seen = float("inf")
    loss_history   = []

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_loss_seen

        if m_name == "BA":
            params = {"m": trial.suggest_int("m", 1, 10)}
        elif m_name == "Bianconi_BB":
            params = {"m": trial.suggest_int("m", 1, 10)}
        elif m_name == "Copying":
            params = {
                "beta":   trial.suggest_float("beta",   0.1, 0.9),
                "m_init": trial.suggest_int(  "m_init", 2,   10),
            }
        elif m_name == "SBM_PA":
            params = {
                "k":     trial.suggest_int(  "k",     5,        40),
                "alpha": trial.suggest_float("alpha", ALPHA_LO, ALPHA_HI),
                "m1":    trial.suggest_int(  "m1",    1,        7),
                "m2":    trial.suggest_int(  "m2",    2,        9),
            }
        elif m_name == "ERGM":
            params = {
                # theta_mut: mutual-dyad param; paper Table 2 shows ~1.7-2.2
                # theta_star: GWD out-degree, alpha=ln(2); Table 2 range -8 to +1.
                # Use LINEAR scale (not log) -- param is signed and O(1).
                "theta_mut":  trial.suggest_float("theta_mut",  -1.0, 5.0),
                "theta_star": trial.suggest_float("theta_star", -5.0, 3.0),
            }
        elif m_name == "BTER":
            params = {
                "alpha":   trial.suggest_float("alpha",   ALPHA_LO, ALPHA_HI),
                "density": trial.suggest_float("density", 0.01,     0.99),
            }
        elif m_name == "Kronecker":
            params = {
                "a": trial.suggest_float("a", 0.4,  0.8),
                "b": trial.suggest_float("b", 0.05, 0.3),
                "c": trial.suggest_float("c", 0.05, 0.3),
            }

        n_graphs = _graphs_for_trial(trial.number, n_trials)
        loss, stats = evaluate_params(m_name, params, n_real, m_real,
                                      real_stats, n_graphs)

        if loss == float("inf"):
            # Notify main so bar still advances on pruned trials
            q.put((model_idx, m_name, trial.number, best_loss_seen))
            raise optuna.TrialPruned()

        trial.set_user_attr("stats",    stats)
        trial.set_user_attr("n_graphs", n_graphs)

        if loss < best_loss_seen:
            best_loss_seen = loss

        loss_history.append((trial.number, best_loss_seen))

        # Only output: send progress to main process via queue
        q.put((model_idx, m_name, trial.number, best_loss_seen))
        return loss

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    try:
        best_loss   = study.best_value
        best_params = study.best_params
        best_stats  = study.best_trial.user_attrs.get("stats", [])
    except ValueError:
        best_loss, best_params, best_stats = float("inf"), {}, []

    # Done sentinel: trial_number=None tells the main loop this worker finished
    q.put((model_idx, m_name, None, best_loss))

    result = {"model": m_name, "params": best_params,
              "loss": best_loss, "stats": best_stats}
    return m_name, result, loss_history


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
    ax_abs.set_ylabel("Loss")
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
    print("  Final Results  (ranked by loss)")
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


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  PyPI Network: 7-Model Parallel Bayesian Optimizer")
    print("=" * 65)
    print(f"  Priors: α_in={ALPHA_IN} (p=0.532, not rejected) | "
          f"α_out={ALPHA_OUT} (p=0.000, rejected)")
    print(f"  Alpha search range tightened to [{ALPHA_LO}, {ALPHA_HI}]")

    try:
        real_g = ig.Graph.Read_GraphML(GRAPH_FILE)
        if not real_g.is_directed():
            real_g = real_g.as_directed(mode="mutual")
    except Exception as e:
        print(f"Failed to load {GRAPH_FILE}: {e}")
        return

    n_real, m_real  = real_g.vcount(), real_g.ecount()
    real_stats      = compute_stats(real_g)
    real_stats_list = real_stats.tolist()

    avg_deg = m_real / n_real
    print(f"[+] Loaded PyPI: {n_real} nodes, {m_real} edges | avg degree {avg_deg:.1f}")
    print(f"[+] {TRIALS_PER_MODEL} trials/model | {WARMUP_TRIALS} QMC warm-up | "
          f"adaptive eval budget | empirical seed trials")
    print(f"[+] Running all 7 models in parallel — progress bars below:\n")

    # ── Shared queue: workers → main ──────────────────────────────────────────
    manager = multiprocessing.Manager()
    q       = manager.Queue()

    tasks = [
        (m_name, idx, n_real, m_real, real_stats_list, TRIALS_PER_MODEL, WARMUP_TRIALS, q)
        for idx, m_name in enumerate(MODEL_NAMES)
    ]

    # ── Create all 7 bars HERE in the main process ────────────────────────────
    # position=i pins each bar to a fixed terminal row.
    # Workers never touch these — only this process calls .refresh().
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

        # ── Event loop: drain queue and refresh bars until all workers done ───
        while done_count < len(MODEL_NAMES):
            while not q.empty():
                try:
                    model_idx, m_name, trial_num, best_loss = q.get_nowait()
                except Exception:
                    break

                bar = bars[model_idx]

                if trial_num is None:
                    # Sentinel — this model is done
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

            time.sleep(0.05)  # 50 ms poll — negligible CPU, responsive display

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