# cython: language_level=3
"""
Growing Stochastic Block Model with Preferential Attachment (GSBM) — Directed.
Implements the algorithm from:
  "A Growing Stochastic Block Model with Preferential Attachment"
  Gombojav, Badraa, Purevsuren — JMIS Vol. 12, No. 3, 2025

KEY CHANGE vs original: edges are now generated as *directed* (u → v) so the
model no longer needs the caller to call .as_directed(mode="mutual"), which was
forcing reciprocity = 1.0 on every generated graph and causing catastrophic loss.

During intra-community PA the current node u emits an arc u→v.  A reciprocal
arc v→u is also added independently with probability `recip_prob` (derived from
m1/m2 ratio) to produce a realistic, non-trivial reciprocity.  Inter-community
arcs follow the same rule.

Parameters
----------
n      : int   — total number of nodes
k      : int   — number of communities
alpha  : float — power-law exponent for community-size distribution
m1     : int   — controls max inter-community edges per node  ([0, 2*m1])
m2     : int   — controls intra-community initialisation and max PA edges ([1, 2*m2])

Returns
-------
igraph.Graph — *directed* graph with n nodes  (caller must NOT call as_directed)
"""

import numpy as np
import igraph as ig


# ---------------------------------------------------------------------------
# Algorithm 2 — Assign nodes to k communities using power-law distribution
# ---------------------------------------------------------------------------
def _generate_community(n, k, alpha):
    S = np.array([1.0 / (i ** alpha) for i in range(1, k + 1)], dtype=float)
    t = S.sum()
    S = S / t
    sizes = np.floor(S * n).astype(int)

    remainder = n - sizes.sum()
    i = 0
    while remainder > 0:
        sizes[i % k] += 1
        remainder -= 1
        i += 1

    communities = []
    start = 0
    for size in sizes:
        communities.append(np.arange(start, start + size, dtype=int))
        start += size

    return communities


# ---------------------------------------------------------------------------
# Algorithm 3 — Initialise intra-community edges for the first m2 nodes
# Now adds directed arcs (u → v) and a reciprocal (v → u) with prob recip_prob.
# ---------------------------------------------------------------------------
def _initialize_community(deg, edges_set, community, m2, recip_prob, rng):
    m2_hat = min(len(community), m2)
    step = 1
    S = list(community[:m2_hat])
    R = list(S)

    for u in S:
        R = [v for v in R if v != u]
        if not R:
            step *= 2
            continue

        num_connect = max(1, len(R) // step)
        num_connect = min(num_connect, len(R))

        chosen = rng.choice(R, size=num_connect, replace=False)
        for v in chosen:
            # Directed arc u → v
            arc = (int(u), int(v))
            if arc not in edges_set:
                edges_set.add(arc)
                deg[u] += 1
                deg[v] += 1
            # Reciprocal arc v → u with probability recip_prob
            if rng.random() < recip_prob:
                rev = (int(v), int(u))
                if rev not in edges_set:
                    edges_set.add(rev)
                    deg[v] += 1
                    deg[u] += 1
        step *= 2

    return set(S)


# ---------------------------------------------------------------------------
# Algorithm 1 — GSBM main function  (directed edition)
# ---------------------------------------------------------------------------
def sbm_pa_model(n, k, alpha, m1, m2):
    """
    GSBM(n, k, alpha, m1, m2) — returns a DIRECTED igraph.Graph.
    Do NOT call .as_directed() on the result.
    """
    rng = np.random.default_rng()

    # Reciprocity probability: heuristic based on m2 density.
    # High m2 → denser intra-community → more chances of mutual ties.
    recip_prob = float(np.clip(1.0 / (m2 + 1), 0.05, 0.5))

    deg = np.zeros(n, dtype=float)
    edges_set = set()

    # -----------------------------------------------------------------------
    # Phase 1 — assign nodes to communities
    # -----------------------------------------------------------------------
    communities = _generate_community(n, k, alpha)

    comm_of = np.empty(n, dtype=int)
    for ci, members in enumerate(communities):
        comm_of[members] = ci

    # -----------------------------------------------------------------------
    # Phase 2 — intra-community directed PA edges
    # -----------------------------------------------------------------------
    for comm in communities:
        S_set = _initialize_community(deg, edges_set, comm, m2, recip_prob, rng)

        m2_hat = min(len(comm), m2)
        remaining = comm[m2_hat:]

        for u in remaining:
            S_list = list(S_set)

            w = np.array([deg[v] for v in S_list], dtype=float)
            total = w.sum()
            if total == 0:
                w = np.ones(len(S_list)) / len(S_list)
            else:
                w = w / total

            l = int(rng.integers(1, 2 * m2 + 1))
            l = min(l, len(S_list))

            chosen = rng.choice(S_list, size=l, replace=False, p=w)

            for v in chosen:
                # Directed arc u → v
                arc = (int(u), int(v))
                if arc not in edges_set:
                    edges_set.add(arc)
                    deg[u] += 1
                    deg[v] += 1
                # Reciprocal arc v → u with probability recip_prob
                if rng.random() < recip_prob:
                    rev = (int(v), int(u))
                    if rev not in edges_set:
                        edges_set.add(rev)
                        deg[v] += 1
                        deg[u] += 1

            S_set.add(int(u))

    # -----------------------------------------------------------------------
    # Phase 3 — inter-community directed PA edges
    # -----------------------------------------------------------------------
    for ci, comm in enumerate(communities):
        outside = np.where(comm_of != ci)[0]
        if len(outside) == 0:
            continue

        for u in comm:
            w = deg[outside].copy()
            total = w.sum()
            if total == 0:
                w = np.ones(len(outside)) / len(outside)
            else:
                w = w / total

            l = int(rng.integers(0, 2 * m1 + 1))
            if l == 0:
                continue
            l = min(l, len(outside))

            chosen = rng.choice(outside, size=l, replace=False, p=w)

            for v in chosen:
                # Directed arc u → v
                arc = (int(u), int(v))
                if arc not in edges_set:
                    edges_set.add(arc)
                    deg[u] += 1
                    deg[v] += 1
                # Reciprocal arc v → u with probability recip_prob
                if rng.random() < recip_prob:
                    rev = (int(v), int(u))
                    if rev not in edges_set:
                        edges_set.add(rev)
                        deg[v] += 1
                        deg[u] += 1

    # Return a DIRECTED graph — caller must NOT call as_directed()
    return ig.Graph(n=n, edges=list(edges_set), directed=True)
