import colorsys
import igraph as ig
import random

ig.set_random_number_generator(random.Random(42))

def generate_cycle_palette(n, saturation=0.75, value=0.92):
    palette = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
        palette.append((
            int(round(r * 255)),
            int(round(g * 255)),
            int(round(b * 255)),
        ))
    return palette


def detect_communities(g):
    ug = g.as_undirected(combine_edges="first")
    comm = ug.community_multilevel()
    membership = comm.membership

    attr = (
        "id"   if "id"   in g.vs.attributes() else
        "name" if "name" in g.vs.attributes() else None
    )

    communities: dict[int, list[str]] = {}
    for i, c_id in enumerate(membership):
        communities.setdefault(c_id, [])
        label = g.vs[i][attr] if attr else f"pkg_{i}"
        communities[c_id].append(label)

    # Keep igraph's original 0-based ordering so IDs match the visualizer
    return sorted(communities.items())


def export_txt(communities, txt_output="communities_list.txt"):
    with open(txt_output, "w", encoding="utf-8") as f:
        for c_id, pkgs in communities:
            f.write(f"Community {c_id} | {len(pkgs)} pkgs\n")
            f.write("-" * 60 + "\n")
            f.write(", ".join(pkgs) + "\n\n")
    print(f"  Package list written → {txt_output}")


def generate_latex_table(communities, output_tex="communities_table.tex"):
    n = len(communities)
    palette = generate_cycle_palette(n)

    rows = []
    for idx, (c_id, pkgs) in enumerate(communities):
        r, g, b = palette[idx]
        row = (
            rf"        \cellcolor[RGB]{{{r},{g},{b}}}{c_id} & {len(pkgs)} &  \\ \hline"
        )
        rows.append(row)

    rows_str = "\n".join(rows)

    latex = rf"""\begin{{longtable}}{{|c|c|p{{10cm}}|}}
    \caption{{Descripción de las comunidades principales detectadas en la red de dependencias de Python.}} \label{{tab:communities_desc}} \\

    \hline
    \textbf{{ID}} & \textbf{{Nodos}} & \textbf{{Descripción}} \\ \hline
    \endhead

    \hline
    \multicolumn{{3}}{{|r|}}{{{{Continúa en la siguiente página...}}}} \\ \hline
    \endfoot

    \hline
    \endlastfoot

{rows_str}
\end{{longtable}}
"""

    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"  LaTeX table written → {output_tex}")


if __name__ == "__main__":
    print("Loading dependency graph…")
    g = ig.Graph.Read_GraphML("pypi_multiseed_10k.graphml")
    print(f"  {g.vcount():,} nodes, {g.ecount():,} edges")

    print("· Running Louvain…")
    communities = detect_communities(g)
    print(f"  Found {len(communities)} communities.")

    export_txt(communities, "communities_list.txt")
    generate_latex_table(communities, "communities_table.tex")

    print("Done.")