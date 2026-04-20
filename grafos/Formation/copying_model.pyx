# cython: language_level=3
"""
Copying model (Kumar et al. / Kleinberg et al., Web graph model).

At every step a new node t is created and adds exactly m_per_node links.
Each link is decided independently (Section 4.1 of the paper):

    With probability (1 - beta)  →  RANDOM:  link to a node chosen
                                              uniformly at random from
                                              all existing nodes.

    With probability beta        →  COPY:    pick a random prototype
                                              node G, then pick one of
                                              G's out-neighbours uniformly
                                              at random and link to it.

This per-link independence is the key mechanism that produces Zipfian
(power-law) in-degree distributions via the implicit preferential
attachment hidden inside the copying step.

The seed is a fully-connected directed clique of size m_init, giving every
seed node a positive out-degree so the copy step can immediately work.

m_per_node is derived from m_real so the generated graph approximately
matches the target edge count of the real network.

Parameters
----------
n      : total number of nodes
m_real : target edge count (used to calibrate edges per new node)
beta   : copying probability  (optimised by train.py)
m_init : seed clique size     (optimised by train.py)
"""
import numpy as np
import igraph as ig


def copying_model(n, m_real, beta, m_init):
    rng = np.random.default_rng()

    # ── Seed: fully-connected directed clique ─────────────────────────────
    edges   = []
    adj_out = [[] for _ in range(n)]

    for i in range(m_init):
        for j in range(m_init):
            if i != j:
                edges.append((i, j))
                adj_out[i].append(j)

    # ── Calibrate edges per new node to hit m_real ────────────────────────
    seed_edges = len(edges)
    remaining  = max(1, n - m_init)
    m_per_node = max(1, round((m_real - seed_edges) / remaining))

    # ── Growth via copying ────────────────────────────────────────────────
    for t in range(m_init, n):
        targets     = set()
        max_tries   = m_per_node * 30          # avoid infinite loop
        attempts    = 0

        while len(targets) < min(m_per_node, t) and attempts < max_tries:
            attempts += 1

            if rng.random() < beta:
                # COPY: pick prototype G, pick one of G's out-neighbours
                proto = int(rng.integers(0, t))
                if adj_out[proto]:
                    idx = int(rng.integers(0, len(adj_out[proto])))
                    w   = adj_out[proto][idx]
                    if w != t:
                        targets.add(w)
            else:
                # RANDOM: uniform over all existing nodes
                w = int(rng.integers(0, t))
                targets.add(w)

        for w in targets:
            edges.append((t, w))
            adj_out[t].append(w)

    return ig.Graph(n=n, edges=edges, directed=True)
