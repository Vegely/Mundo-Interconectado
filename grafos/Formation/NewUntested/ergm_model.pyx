# cython: language_level=3
"""
ergm_model.pyx — Exponential Random Graph (p*) model simulator.

Compliant with:
    Snijders (2002) "MCMC Estimation of Exponential Random Graph Models"

Fixes applied vs original
--------------------------
1. **Adjacency matrix** (g×g int array) replaces the edge set — the paper
   represents graphs as Y = (y_ij) throughout (Section 2).

2. **Single-arc Metropolis-Hastings** (Eq. 17 / Section 5.1) replaces edge
   swaps.  Each step picks one arc (i,j) and proposes toggling it; swaps
   change two arcs simultaneously and are not the algorithm described.

3. **Correct delta** (Eq. 6): logit P{Y_ij=1|rest} = θ' [u(y^{ij1}) − u(y^{ij0})].
   Original only tracked reciprocity and in-degree change heuristically.

4. **Density parameter theta_ties** added (θ_1 in Eq. 3).  A model without
   a density term is not a proper exponential family — ties are the
   canonical first sufficient statistic (Eq. 2, Section 3).

5. **Inversion steps** at probability p_inv=0.01 (Eq. 19-21, Section 5.3).
   These "big jumps" to the complement graph are the paper's main remedy
   for the bimodality / slow-mixing problem described in Sections 3-4.

6. **Sufficient statistics** match the paper's directed-graph model:
       u1 = ties, u2 = mutual dyads (theta_mut),
       u3 = in-twostars (theta_star)              [Section 3.2 / 8]
"""

import numpy as np
cimport numpy as np
from libc.math cimport exp, log


# ---------------------------------------------------------------------------
# Sufficient-statistic delta  (Eq. 6)
# u(y^{ij1}) − u(y^{ij0}) for the three-effect model:
#   u1 = ties, u2 = mutual dyads, u3 = in-twostars
# ---------------------------------------------------------------------------
cdef void _delta_stats(int in_deg_rj_other, int reverse_arc,
                       double* d1, double* d2, double* d3) noexcept nogil:
    """
    Compute change in stats using tracked in-degree[cite: 35, 37].
    d1 (ties)        : +1 for the proposed arc[cite: 35].
    d2 (mutual dyads): +1 if the reverse arc Y[rj,ri] exists[cite: 36].
    d3 (in-twostars) : number of OTHER nodes sending to rj.
    """
    d1[0] = 1.0          
    d2[0] = <double>reverse_arc   
    d3[0] = <double>in_deg_rj_other


# ---------------------------------------------------------------------------
# Inversion-step acceptance (Eq. 20, Gibbs version)
# ---------------------------------------------------------------------------
cdef double _inversion_prob(int[:, :] Y, int g, int p,
                             double theta_ties, double theta_mut,
                             double theta_star) nogil:
    """
    pr(y) = exp(θ'u(1−y)) / [exp(θ'u(1−y)) + exp(θ'u(y))]   (Eq. 20).

    Computed from scratch for current Y and its complement (1−Y).
    """
    # ── All cdef declarations at the top of this function ─────────────────
    cdef int i, j
    cdef double ties_y, mut_y, istar_y
    cdef double ties_c, mut_c, istar_c
    cdef double indeg, indeg_c
    cdef double n_possible, asym_y, null_y
    cdef double theta_u, theta_uc, mx, pr

    # ── Initialise accumulators ────────────────────────────────────────────
    ties_y  = 0.0
    mut_y   = 0.0
    istar_y = 0.0
    ties_c  = 0.0
    mut_c   = 0.0
    istar_c = 0.0

    # ── Compute statistics for Y ──────────────────────────────────────────
    for i in range(g):
        indeg = 0.0
        for j in range(g):
            if i == j:
                continue
            ties_y += Y[i][j]
            if Y[i][j] == 1 and Y[j][i] == 1 and i < j:
                mut_y += 1.0
            indeg += Y[j][i]
        istar_y += indeg * (indeg - 1.0) / 2.0   # C(indeg, 2)

    # ── Compute statistics for complement (1 − Y) ─────────────────────────
    # complement: y_ij_c = 1 − y_ij (diagonal stays 0)
    n_possible = <double>(g * (g - 1))
    ties_c  = n_possible - ties_y
    # mutual dyads in complement: null dyads in Y become mutual in complement
    asym_y = ties_y - 2.0 * mut_y
    null_y = g * (g - 1) / 2.0 - mut_y - asym_y
    mut_c  = null_y

    # in-twostars of complement: use complement in-degrees = (g-1) − indeg_y
    for i in range(g):
        indeg = 0.0
        for j in range(g):
            if i != j:
                indeg += Y[j][i]
        indeg_c = (g - 1) - indeg
        istar_c += indeg_c * (indeg_c - 1.0) / 2.0

    # ── Acceptance probability (log-sum-exp for numerical stability) ───────
    theta_u  = theta_ties * ties_y  + theta_mut * mut_y  + theta_star * istar_y
    theta_uc = theta_ties * ties_c  + theta_mut * mut_c  + theta_star * istar_c

    mx = theta_uc if theta_uc > theta_u else theta_u
    pr = exp(theta_uc - mx) / (exp(theta_uc - mx) + exp(theta_u - mx))
    return pr


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------
def ergm_model(int n, int m_real,
               double theta_mut, double theta_star,
               int n_steps=0, double p_inv=0.00):
    """
    Simulate one draw from an ERGM via Metropolis-Hastings (Eq. 17)
    augmented with inversion steps (Eq. 19-21, Section 5.3).

    Model (three sufficient statistics, directed graph)
    ---------------------------------------------------
    u1(y) = Σ y_ij                    (ties)
    u2(y) = Σ_{i<j} y_ij y_ji         (mutual dyads)
    u3(y) = Σ_i C(y_{+i}, 2)          (in-twostars)

    Parameters   (Section 3.3 / Section 8 notation)
    ------------------------------------------------
    n          : number of nodes g
    m_real     : target edge count; used to set theta_ties so the
                 stationary mean density ≈ m_real / (n*(n-1))  (Eq. 13)
    theta_mut  : reciprocity / mutual-dyad parameter  (θ_2)
    theta_star : in-twostar parameter                  (θ_3)
    n_steps    : MCMC steps; default = 100 * n^2 (paper Appendix)
    p_inv      : inversion-step probability; paper recommends 0.01
                 (Section 5.3 / Appendix)

    Returns
    -------
    igraph.Graph  directed graph sampled from the ERGM
    """
    import igraph as ig

    # ── All cdef declarations at the top of this function ─────────────────
    cdef int i, j, t, ri, rj, new_val
    cdef double density, theta_ties
    cdef double d1, d2, d3, logit_val
    cdef double accept_prob, pr
    cdef int[:, :] Y
    cdef int[:] in_degrees  

    # ── Setup ──────────────────────────────────────────────────────────────
    if n_steps == 0:
        n_steps = 2_000_000

    rng = np.random.default_rng()

    # theta_ties (density parameter, θ_1): set so that expected density
    # at the reciprocity-and-instar model with θ_2=θ_3=0 matches
    # m_real / (n*(n-1)).  That is the logit of the target density.
    density = m_real / (<double>(n * (n - 1)))
    if density < 1e-6:
        density = 1e-6
    if density > 1.0 - 1e-6:
        density = 1.0 - 1e-6
    theta_ties = log(density / (1.0 - density))

    # Initialise Y ~ Bernoulli(density) with zero diagonal (Appendix)
    Y_np = (rng.random((n, n)) < density).astype(np.int32)
    for i in range(n):
        Y_np[i, i] = 0
    Y = Y_np
    in_degrees = np.sum(Y_np, axis=0).astype(np.int32)

    # ── MCMC loop ──────────────────────────────────────────────────────────
    for t in range(n_steps):

        # --- Inversion step (Eq. 19-21, Section 5.3) ----------------------
        if rng.random() < p_inv:
            pr = _inversion_prob(Y, n, 3, theta_ties, theta_mut, theta_star)
            if rng.random() < pr:
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            Y[i][j] = 1 - Y[i][j]
            continue   # skip single-arc update this step

        # --- Single-arc Metropolis-Hastings (Eq. 17, Section 5.1) ---------
        ri = int(rng.integers(0, n))
        rj = int(rng.integers(0, n - 1))
        if rj >= ri:
            rj += 1

        # Compute u(y^{ij1}) − u(y^{ij0})  (Eq. 6)
        _delta_stats(in_degrees[rj] - Y[ri][rj], Y[rj][ri], &d1, &d2, &d3)

        # logit P{Y_ij=1 | rest} = θ' Δu  (Eq. 6)
        logit_val = theta_ties * d1 + theta_mut * d2 + theta_star * d3

        # MH acceptance (Eq. 17):
        #   Y[ri,rj]=0 → propose 1: accept with P = logistic(logit_val)
        #   Y[ri,rj]=1 → propose 0: accept with P = logistic(−logit_val)
        if Y[ri][rj] == 0:
            accept_prob = 1.0 / (1.0 + exp(-logit_val))
            new_val = 1 if rng.random() < accept_prob else 0
        else:
            accept_prob = 1.0 / (1.0 + exp(logit_val))
            new_val = 0 if rng.random() < accept_prob else 1

        if new_val != Y[ri][rj]:
            if new_val == 1:
                in_degrees[rj] += 1
            else:
                in_degrees[rj] -= 1
            Y[ri][rj] = new_val

    # Return as igraph directed graph
    edges = [(i, j) for i in range(n) for j in range(n)
             if i != j and Y[i][j] == 1]
    return ig.Graph(n=n, edges=edges, directed=True)
