import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT = Path(__file__).parent

def normalize_log(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=np.float64)
    arr = np.log1p(np.clip(arr, 0, None))
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)


def normalize_linear(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=np.float64)
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)

def load_graph() -> ig.Graph:
    g = ig.Graph.Read("pypi_multiseed_10k.graphml")
    attrs = g.vertex_attributes()
    if "name" not in attrs:
        g.vs["name"] = g.vs["id"]
    return g

def compute_centralities(g: ig.Graph) -> tuple[dict, list]:
    names = g.vs["name"]

    print("  · pagerank …")
    pr_raw = (np.array(g.pagerank(damping=0.85)))

    print("  · betweenness …")
    btwn_raw = (np.array(g.betweenness(directed=True)))

    print("  · closeness …")
    clos_raw = (np.nan_to_num(np.array(g.closeness()), nan=0.0))

    centralities = {
        "pagerank": {
            "label":    "PageRank",
            "raw":      pr_raw,
            "norm":     normalize_log(pr_raw),
            "norm_tag": "log-normalised",
            "color":    "#9333ea",
        },
        "betweenness": {
            "label":    "Betweenness Centrality",
            "raw":      btwn_raw,
            "norm":     normalize_log(btwn_raw),
            "norm_tag": "log-normalised",
            "color":    "#ea6c10",
        },
        "closeness": {
            "label":    "Closeness Centrality",
            "raw":      clos_raw,
            "norm":     normalize_log(clos_raw),
            "norm_tag": "log-normalised",
            "color":    "#0284c7",
        },
    }
    return centralities, names

def write_ranking_txt(key: str, info: dict, names: list, out_dir: Path) -> None:
    raw   = info["raw"]
    norm  = info["norm"]
    order = np.argsort(raw)[::-1]

    path = out_dir / f"{key}_ranking.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{info['label']} — full node ranking\n")
        f.write(f"{'=' * 82}\n")
        f.write(f"Sorted by raw score (descending).\n")
        f.write(f"Normalisation: {info['norm_tag']}.\n")
        f.write(f"Total nodes: {len(names):,}\n")
        f.write(f"{'=' * 82}\n\n")
        f.write(f"{'Rank':<8}{'Node':<52}{'Raw Score':>16}{'Norm Score':>12}\n")
        f.write(f"{'-' * 88}\n")
        for rank, idx in enumerate(order, 1):
            f.write(
                f"{rank:<8}{str(names[idx]):<52}{raw[idx]:>16.8f}{norm[idx]:>12.6f}\n"
            )

    print(f"  {path.name}  ({len(names):,} rows)")


def print_top_bottom(key: str, info: dict, names: list, n: int = 10) -> None:
    raw   = info["raw"]
    norm  = info["norm"]
    order = np.argsort(raw)[::-1]
    bar   = "─" * 66

    print(f"\n{'═' * 66}")
    print(f"  {info['label']}")
    print(f"{'═' * 66}")

    for section_label, indices in [
        (f"TOP {n}",    order[:n]),
        (f"BOTTOM {n}", order[-n:]),
    ]:
        base_rank = 1 if "TOP" in section_label else len(order) - n + 1
        print(f"\n  {section_label}")
        print(f"  {bar}")
        print(f"  {'Rank':<6}{'Node':<40}{'Raw Score':>14}{'Norm Score':>12}")
        print(f"  {bar}")
        for offset, idx in enumerate(indices):
            node = str(names[idx])
            node = node[:38] + ".." if len(node) > 38 else node
            print(f"  {base_rank + offset:<6}{node:<40}{raw[idx]:>14.8f}{norm[idx]:>12.6f}")


def plot_distributions(centralities: dict, out_dir: Path) -> None:
    keys   = list(centralities.keys())
    n_cols = len(keys)

    GRID_COLOR  = "#cccccc"
    TICK_COLOR  = "#444444"
    LABEL_COLOR = "#222222"
    SPINE_COLOR = "#aaaaaa"

    configs = [
        {
            "suffix": "linear",
            "title": "Distribuciones de Centralidad",
            "xlabel": "Valor",
            "scale": "linear",
            "ylabel": "Densidad"
        },
        {
            "suffix": "loglog",
            "title": "Distribuciones de Centralidad (log-log)",
            "xlabel": "Valor (log-log)",
            "scale": "log",
            "ylabel": "Frecuencia"
        }
    ]

    for cfg in configs:
        fig = plt.figure(figsize=(6 * n_cols, 5), facecolor="white")
        fig.suptitle(cfg["title"], color=LABEL_COLOR, fontsize=17, fontweight="bold", y=1.05)

        gs = gridspec.GridSpec(1, n_cols, wspace=0.32,
                               left=0.07, right=0.97, top=0.88, bottom=0.15)

        for col, key in enumerate(keys):
            info  = centralities[key]
            color = info["color"]

            data = info["raw"]
            
            ax = fig.add_subplot(gs[0, col])
            ax.set_facecolor("white")

            if cfg["scale"] == "log":
                pos_data = data[data > 0]
                if len(pos_data) > 0:
                    bins = np.logspace(np.log10(pos_data.min()), np.log10(pos_data.max()), 40)
                    ax.hist(pos_data, bins=bins, color=color, alpha=0.75,
                    edgecolor="white", linewidth=0.4, density=True)
                ax.set_xscale("log")
                ax.set_yscale("log")
            else:
                ax.hist(data, bins=40, color=color, alpha=0.75,
                        edgecolor="white", linewidth=0.4, density=True)

            # --- Estética ---
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--")
            ax.xaxis.grid(cfg["scale"] == "log", color=GRID_COLOR, linewidth=0.6, linestyle="--")

            for spine in ax.spines.values():
                spine.set_edgecolor(SPINE_COLOR)
                spine.set_linewidth(0.8)

            ax.set_title(info["label"], color=LABEL_COLOR, fontsize=11, fontweight="semibold", pad=7)
            ax.set_xlabel(cfg["xlabel"], color=LABEL_COLOR, fontsize=8.5)
            ax.set_ylabel(cfg["ylabel"], color=LABEL_COLOR, fontsize=8.5)
            ax.tick_params(colors=TICK_COLOR, labelsize=7.5)

        path = out_dir / f"centrality_distributions_{cfg['suffix']}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  ok {path.name}")

def main() -> None:
    g = load_graph()
    print(f"   {g.vcount():,} nodes   {g.ecount():,} edges\n")
    centralities, names = compute_centralities(g)
    for key, info in centralities.items():
        print_top_bottom(key, info, names, n=10)
    for key, info in centralities.items():
        write_ranking_txt(key, info, names, OUT)
    plot_distributions(centralities, OUT)

if __name__ == "__main__":
    main()