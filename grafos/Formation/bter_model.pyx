# cython: language_level=3
"""
Block Two-Level Erdős–Rényi (BTER) model — directed adaptation.
Based on: Kolda, Pinar, Plantenga & Seshadhri (2013).

FIXES vs previous version:
  1. block_density now has ONE role: clustering target only.
     rho = block_density^(1/3) per §3.5 — nothing else.

  2. Phase 3 (reciprocity hack) REMOVED. It was running after Phase 2
     already hit the m_real edge budget, then adding up to density*|E|
     more edges, which inflated the degree distribution by 10–90% and
     destroyed the entropy terms in the loss function (~300x higher loss).

  3. Phase 1 directed ER loop naturally produces non-zero reciprocity:
     (u→v) and (v→u) are independent Bernoulli(rho) draws, so E[recip]
     ≈ rho inside each block. No separate phase needed.

  4. n_phase2 guarded with max(0, ...) instead of max(1, ...) to avoid
     injecting a spurious extra edge when Phase 1 already meets budget.

Parameters
----------
n             : number of nodes
m_real        : target edge count
alpha         : power-law exponent for degree distribution (α_in ≈ 1.9 for PyPI)
block_density : target clustering coefficient; rho = density^(1/3)
"""
import numpy as np
import igraph as ig


def bter_model(n, m_real, alpha, block_density):
    rng = np.random.default_rng()

    # ── PREPROCESSING: power-law degree sequence ──────────────────────────
    d_max  = max(2, int(np.sqrt(n)))
    d_vals = np.arange(1, d_max + 1, dtype=float)
    probs  = d_vals ** (-alpha)
    probs /= probs.sum()

    degrees = rng.choice(d_max, size=n, p=probs) + 1
    degrees = np.minimum(degrees, n - 1).astype(float)

    # Rescale so the expected total degree matches m_real
    total = degrees.sum()
    if total > 0:
        degrees = np.maximum(1.0, np.round(degrees * (m_real / total)))
    degrees = np.minimum(degrees, n - 1).astype(int)

    order      = np.argsort(degrees)
    sorted_deg = degrees[order]

    # ── PHASE 1: intra-block directed ER (§3.5) ───────────────────────────
    # For a block of degree-d nodes, the ER connectivity rho = c_d^(1/3)
    # ensures each node expects c_d * C(d,2) triangles, matching the
    # target clustering coefficient.
    # The directed loop over (ui, vi) with ui≠vi covers both (u,v) and
    # (v,u) as independent draws, giving E[reciprocity] ≈ rho per block
    # without any additional phase.
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
    # Each node's excess degree = desired - what Phase 1 already gave it.
    # We draw exactly (m_real - |Phase 1 edges|) more directed edges using
    # Chung-Lu weighted sampling, keeping the total edge count at m_real.
    excess_out = np.maximum(0.0, degrees.astype(float) - out_deg_phase1)
    excess_in  = np.maximum(0.0, degrees.astype(float) - in_deg_phase1)

    sum_out = excess_out.sum()
    sum_in  = excess_in.sum()

    if sum_out > 1.0 and sum_in > 1.0:
        p_out = excess_out / sum_out
        p_in  = excess_in  / sum_in

        phase1_count = len(edges)
        n_phase2     = max(0, m_real - phase1_count)  # never negative

        if n_phase2 > 0:
            srcs = rng.choice(n, size=n_phase2, p=p_out)
            tgts = rng.choice(n, size=n_phase2, p=p_in)

            for u, v in zip(srcs.tolist(), tgts.tolist()):
                if u != v:
                    edges.add((int(u), int(v)))

    return ig.Graph(n=n, edges=list(edges), directed=True)
