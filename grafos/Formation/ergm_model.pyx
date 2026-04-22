# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Exponential Random Graph Model (ERGM) for directed graphs.

UPDATED: 
Included Geometrically Weighted In-Degree (to shape the in-degree power-law tail) 
and Transitive Triangles (to explicitly reward clustering and fix transitivity penalties).

Sufficient statistics:
  u1(y) = Density / Arcs
  u2(y) = Mutual dyads / Reciprocity
  u3(y) = GWD Out-degree
  u4(y) = GWD In-degree
  u5(y) = Transitive Triangles (src -> k -> tgt)
"""

import numpy as np
import igraph as ig
cimport numpy as np
from libc.math cimport exp, log

# ── Module-level C constants ──────────────────────────────────────────────────
# α = ln 2, recommended in literature to give a geometrically decreasing weight sequence
cdef double ALPHA   = 0.6931471805599453    # ln(2)
cdef double FACTOR  = 0.5                    # 1 - e^{-α}


def ergm_model(int n, int m_real, double theta_mut, double theta_out, double theta_in, double theta_tri):
    """
    Draw a directed graph from the ERGM distribution via Metropolis-Hastings.

    Parameters
    ----------
    n          : number of nodes
    m_real     : target arc count (calibrates baseline density)
    theta_mut  : reciprocity parameter
    theta_out  : geometrically weighted out-degree parameter
    theta_in   : geometrically weighted in-degree parameter
    theta_tri  : transitive triangles parameter (clustering)
    """
    cdef:
        int    step, src, tgt, cur, sign, out_red, in_red, n_steps, k, shared
        double theta_density, p_target
        double d1, d2, d3, d4, d5, log_odds
        np.int32_t[:, :]  adj
        np.int32_t[:]     out_deg
        np.int32_t[:]     in_deg
        double[:]         rvals
        np.int64_t[:]     src_arr, tgt_arr

    # ─────────────────────────────────────────────────────────────────────────
    # 1. CALIBRATE DENSITY PARAMETER
    # ─────────────────────────────────────────────────────────────────────────
    p_target      = float(m_real) / max(n * (n - 1), 1)
    p_target      = max(1e-9, min(1.0 - 1e-9, p_target))
    theta_density = log(p_target / (1.0 - p_target))

    # ─────────────────────────────────────────────────────────────────────────
    # 2. INITIALISE MATRICES AND DEGREES
    # ─────────────────────────────────────────────────────────────────────────
    rng    = np.random.default_rng()
    adj_np = (rng.random((n, n)) < p_target).astype(np.int32)
    np.fill_diagonal(adj_np, 0)
    adj    = adj_np

    out_deg_np = adj_np.sum(axis=1, dtype=np.int32)
    out_deg    = out_deg_np
    
    in_deg_np  = adj_np.sum(axis=0, dtype=np.int32)
    in_deg     = in_deg_np

    # ─────────────────────────────────────────────────────────────────────────
    # 3. PRE-GENERATE RANDOM VARIATES
    # ─────────────────────────────────────────────────────────────────────────
    n_steps = 200_000
    rvals   = rng.random(n_steps)
    src_arr = rng.integers(0, n, n_steps, dtype=np.int64)
    tgt_arr = rng.integers(0, n, n_steps, dtype=np.int64)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. METROPOLIS-HASTINGS SWEEP
    # ─────────────────────────────────────────────────────────────────────────
    for step in range(n_steps):
        src = <int>src_arr[step]
        tgt = <int>tgt_arr[step]
        if src == tgt:
            continue

        cur  = adj[src, tgt]
        sign = 1 - 2 * cur          # +1 if adding arc, -1 if removing

        # Δu1: Density
        d1 = <double>sign

        # Δu2: Mutual dyads / Reciprocity
        d2 = sign * adj[tgt, src]

        # Δu3: GWD Out-degree
        out_red = out_deg[src] - cur
        d3 = -sign * FACTOR * exp(-ALPHA * out_red)

        # Δu4: GWD In-degree
        in_red = in_deg[tgt] - cur
        d4 = -sign * FACTOR * exp(-ALPHA * in_red)

        # Δu5: Transitive Triangles (Clustering coefficient proxy)
        # We count how many times the path src -> k -> tgt exists.
        # This explicitly calculates shared partners.
        shared = 0
        if theta_tri != 0.0:
            for k in range(n):
                if adj[src, k] and adj[k, tgt]:
                    shared += 1
        d5 = sign * shared

        # Log acceptance ratio
        log_odds = (theta_density * d1 + 
                    theta_mut * d2 + 
                    theta_out * d3 + 
                    theta_in  * d4 + 
                    theta_tri * d5)

        # Accept / reject
        if log_odds >= 0.0 or rvals[step] < exp(log_odds):
            adj[src, tgt]  = 1 - cur
            out_deg[src]  += sign
            in_deg[tgt]   += sign

    # ─────────────────────────────────────────────────────────────────────────
    # 5. BUILD AND RETURN igraph
    # ─────────────────────────────────────────────────────────────────────────
    rows, cols = np.where(adj_np)
    return ig.Graph(
        n        = n,
        edges    = list(zip(rows.tolist(), cols.tolist())),
        directed = True,
    )