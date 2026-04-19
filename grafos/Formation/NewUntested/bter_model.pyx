# cython: language_level=3
"""
Block Two-Level Erdős–Rényi (BTER) model — directed adaptation.
Based on: Kolda, Pinar, Plantenga & Seshadhri (2013).

KEY CHANGES vs original:
  1. Phase 2 edge budget fixed: targets m_real directly instead of sum_out/2,
     which was systematically underestimating the edge count.
  2. Reciprocity phase added: after Phase 1+2, a fraction of existing edges
     get a reciprocal arc added.  The fraction is controlled by `block_density`,
     which the optimiser can tune (high density → more reciprocal pairs).
     This gives the model a degree of freedom to match the real graph's
     reciprocity instead of always landing near zero.

Parameters
----------
n             : number of nodes
m_real        : target edge count
alpha         : power-law exponent for degree distribution
block_density : target clustering coefficient AND reciprocity proxy
                (optimised by train.py)
"""
import numpy as np
import igraph as ig


def bter_model(n, m_real, alpha, block_density):
    rng = np.random.default_rng()

    # ── PREPROCESSING: degree sequence ────────────────────────────────────
    d_max  = max(2, int(np.sqrt(n)))
    d_vals = np.arange(1, d_max + 1, dtype=float)
    probs  = d_vals ** (-alpha)
    probs /= probs.sum()

    degrees = rng.choice(d_max, size=n, p=probs) + 1
    degrees = np.minimum(degrees, n - 1).astype(float)

    total = degrees.sum()
    if total > 0:
        degrees = np.maximum(1.0, np.round(degrees * (m_real / total)))
    degrees = np.minimum(degrees, n - 1).astype(int)

    order      = np.argsort(degrees)
    sorted_deg = degrees[order]

    # ── PHASE 1: intra-block directed ER ─────────────────────────────────
    # ρ_b = ∛(c_d) where c_d = block_density  (§3.5)
    rho = float(np.clip(block_density, 1e-9, 1.0) ** (1.0 / 3.0))

    edges          = set()
    out_deg_phase1 = np.zeros(n, dtype=float)
    in_deg_phase1  = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        d          = int(sorted_deg[i])
        block_size = min(d + 1, n - i)
        block      = order[i: i + block_size]

        if block_size > 1:
            for ui in range(block_size):
                for vi in range(block_size):
                    if ui != vi:
                        u = int(block[ui])
                        v = int(block[vi])
                        if rng.random() < rho:
                            if (u, v) not in edges:
                                edges.add((u, v))
                                out_deg_phase1[u] += 1.0
                                in_deg_phase1[v]  += 1.0
        i += block_size

    # ── PHASE 2: directed Chung-Lu on excess degrees (§3.6) ──────────────
    excess_out = np.maximum(0.0, degrees.astype(float) - out_deg_phase1)
    excess_in  = np.maximum(0.0, degrees.astype(float) - in_deg_phase1)

    sum_out = excess_out.sum()
    sum_in  = excess_in.sum()

    if sum_out > 1.0 and sum_in > 1.0:
        p_out = excess_out / sum_out
        p_in  = excess_in  / sum_in

        # FIX: target m_real minus what Phase 1 already produced,
        # not sum_out/2 which chronically underestimates the budget.
        phase1_count = len(edges)
        n_phase2 = max(1, m_real - phase1_count)

        srcs = rng.choice(n, size=n_phase2, p=p_out)
        tgts = rng.choice(n, size=n_phase2, p=p_in)

        for u, v in zip(srcs.tolist(), tgts.tolist()):
            if u != v:
                edges.add((int(u), int(v)))

    # ── PHASE 3: reciprocity boost ────────────────────────────────────────
    # For each directed edge (u→v) that exists, add (v→u) with probability
    # recip_prob.  This gives the optimiser a way to match real-graph
    # reciprocity; block_density is reused as the proxy (values [0.1, 0.9]
    # cover the realistic range of reciprocity fractions).
    recip_prob = float(block_density)   # same parameter, new role
    if recip_prob > 0:
        existing = list(edges)          # snapshot before we start adding
        for (u, v) in existing:
            if rng.random() < recip_prob:
                rev = (v, u)
                if rev not in edges:
                    edges.add(rev)

    return ig.Graph(n=n, edges=list(edges), directed=True)
