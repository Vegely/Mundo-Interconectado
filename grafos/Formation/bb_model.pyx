# cython: language_level=3
"""
Bianconi-Barabasi fitness model (Bianconi & Barabasi, 2001).

Each new node i is assigned a fitness η_i ~ Uniform[0, 1].
Attachment probability follows eq. (1) from the paper:

    Π_i = η_i * k_i / Σ_j (η_j * k_j)

where k_i is the in-degree of node i (directed adaptation).
A +1 regularisation is added so nodes with k=0 can still receive links,
consistent with the common implementation practice when using a finite seed.

The seed is a fully-connected directed clique of size max(3, m) to ensure
all seed nodes start with positive in-degree, matching the paper's
assumption that the initial network is small but already connected.

Parameters
----------
n            : total number of nodes to grow to
m_real       : target edge count (unused here; edge count is determined by m
               and n, consistent with the paper's growth process)
m            : number of links each new node attaches (optimised by train.py)
"""
import numpy as np
import igraph as ig


def bianconi_barabasi_model(n, m_real, m):
    rng = np.random.default_rng()

    # Fitness drawn i.i.d. from Uniform[0,1] — eq. before (1) in paper
    eta    = rng.uniform(0.0, 1.0, size=n)
    in_deg = np.zeros(n, dtype=float)
    edges  = []

    # ── Seed: fully-connected directed clique ─────────────────────────────
    cs = max(3, m)
    for i in range(cs):
        for j in range(cs):
            if i != j:
                edges.append((i, j))
                in_deg[j] += 1.0

    # ── Growth + preferential attachment with fitness ─────────────────────
    # Π_i = η_i * k_i  (eq. 1); +1 regularisation avoids zero weights
    for t in range(cs, n):
        w        = eta[:t] * (in_deg[:t] + 1.0)
        actual_m = min(m, t)
        targets  = rng.choice(t, size=actual_m, replace=False, p=w / w.sum())
        for tgt in targets:
            edges.append((t, int(tgt)))
            in_deg[tgt] += 1.0

    return ig.Graph(n=n, edges=edges, directed=True)
