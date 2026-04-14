"""
Community detection for PyPI dependency graphs.
- Louvain with a properly seeded igraph RNG → deterministic results every run
- Communities labeled by index (no LLM)
- Donut chart + txt report
"""

import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random


def detect_and_export_communities(
    g,
    txt_output: str = "communities_list.txt",
    plot_output: str = "communities_donut.png",
):
    # ── 1. Louvain with deterministic igraph RNG ──────────────────────────
    print("· Running Louvain (deterministic seed)…")
    ig.set_random_number_generator(random.Random(42))
    ug = g.as_undirected(combine_edges="first")
    comm = ug.community_multilevel()
    membership = comm.membership
    n_communities = max(membership) + 1
    print(f"  Found {n_communities} communities.")

    # ── 2. Group packages by community ────────────────────────────────────
    attr = (
        "id"   if "id"   in g.vs.attributes() else
        "name" if "name" in g.vs.attributes() else None
    )
    communities: dict[int, list[str]] = {}
    for i, c_id in enumerate(membership):
        communities.setdefault(c_id, [])
        label = g.vs[i][attr] if attr else f"pkg_{i}"
        communities[c_id].append(label)

    sorted_communities = sorted(
        communities.items(), key=lambda x: len(x[1]), reverse=True
    )
    total_pkgs = g.vcount()

    # ── 3. Build names (just numbered) ────────────────────────────────────
    community_names: dict[int, str] = {}
    for rank, (c_id, pkgs) in enumerate(sorted_communities, start=1):
        community_names[c_id] = f"Community {rank}"
        print(f"  [{len(pkgs):>5} pkgs]  {community_names[c_id]}")

    # ── 4. Export .txt ────────────────────────────────────────────────────
    print(f"\n· Writing {txt_output}…")
    with open(txt_output, "w", encoding="utf-8") as f:
        f.write("COMMUNITY REPORT (LOUVAIN)\n")
        f.write("==========================\n\n")
        for rank, (c_id, pkgs) in enumerate(sorted_communities, start=1):
            f.write(f"Community {rank} | {len(pkgs)} pkgs\n")
            f.write("-" * 60 + "\n")
            f.write(", ".join(pkgs) + "\n\n")

    # ── 5. Donut chart ────────────────────────────────────────────────────
    print(f"· Rendering {plot_output}…")
    max_slices = 9
    sizes  = [len(pkgs) for _, pkgs in sorted_communities]
    labels = [community_names[c_id] for c_id, _ in sorted_communities]

    if len(sizes) > max_slices:
        top_sizes  = sizes[:max_slices - 1]
        top_labels = labels[:max_slices - 1]
        others_size = sum(sizes[max_slices - 1:])
        sizes  = [others_size] + top_sizes
        labels = ["Other communities"] + top_labels

    cmap   = matplotlib.colormaps["tab10"]
    colors = ["#9e9e9e"] + [cmap(i) for i in range(len(sizes) - 1)]

    fig, ax = plt.subplots(figsize=(15, 7))
    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )

    ax.add_artist(plt.Circle((0, 0), 0.40, fc="white"))
    ax.text(0,  0.06, f"{total_pkgs:,}", ha="center", va="center",
            fontsize=15, fontweight="bold", color="#111111")
    ax.text(0, -0.10, "packages",        ha="center", va="center",
            fontsize=10, color="#555555")

    legend_labels = [
        f"{lbl}   —   {sz:,}  ({sz/total_pkgs*100:.1f}%)"
        for lbl, sz in zip(labels, sizes)
    ]
    ax.legend(
        wedges, legend_labels,
        loc="center left", bbox_to_anchor=(0.95, 0.5),
        frameon=False, handlelength=1.1, handleheight=1.1, fontsize=9,
    )

    plt.title(
        "Giant Strongly Connected Component — Communities\n"
        "Distribution by Community (Louvain, multiseed 10k graph)",
        fontsize=12, fontweight="bold", pad=20,
    )
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(plot_output, dpi=300, bbox_inches="tight")
    plt.close()
    print("· Done ✓")


if __name__ == "__main__":
    print("Loading dependency graph…")
    g = ig.Graph.Read_GraphML("pypi_multiseed_10k.graphml")
    print(f"  {g.vcount():,} nodes, {g.ecount():,} edges")

    detect_and_export_communities(
        g,
        txt_output="communities.txt",
        plot_output="communities_donut.png",
    )