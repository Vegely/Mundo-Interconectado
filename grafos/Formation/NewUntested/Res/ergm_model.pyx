# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Exponential Random Graph Model (ERGM) for directed graphs.

Implements the MCMC sampler described in:
  [1] Snijders (2002)  "MCMC Estimation of Exponential Random Graph Models"
      Journal of Social Structure 3(2).
  [2] Snijders, Pattison, Robins & Handcock (2006)  "New Specifications
      for ERGMs"  Sociological Methodology 36, pp. 99–153.

─────────────────────────────────────────────────────────────────────────────
Sufficient statistics  (directed graph, §6 of [2]; eqs. 7–8 of [1]):

  u1(y) = Σ_{i≠j} y_ij                         (arc count / density)
  u2(y) = Σ_{i<j} y_ij · y_ji                  (mutual dyads / reciprocity)
  u3(y) = Σ_i  exp(−α · y_i+)                  (geometrically weighted
                                                  out-degrees, eq. 31 of [2])

Model probability (eq. 3–4 of [1], eq. 4 of [2]):

  P(Y = y) ∝ exp( θ_density · u1(y)
                + θ_mut     · u2(y)
                + θ_star    · u3(y) )

Arc-toggle change statistics  (eq. 6–8 of [1]; eq. 7, 17, 35 of [2]):

  Δu1 = ±1
  Δu2 = ±y_ji                                   (mutual dyad formed/broken)
  Δu3 = −(1−e^{−α}) · e^{−α·ỹ_i+}  · sign     (ỹ_i+ = out-deg without arc)

MCMC: Metropolis-Hastings with single-arc toggle (§5.1 of [1]).  The
acceptance probability is  min(1, exp(θ·Δu))  (eq. 16–17 of [1]).

─────────────────────────────────────────────────────────────────────────────
Fixed hyperparameter
  α = ln 2  (↔ λ = 2), recommended in §3.1 of [2]; gives a geometrically
  decreasing weight sequence with the largest practically useful decay.

θ_density is set automatically to calibrate the expected arc count to
m_real (Bernoulli baseline: θ_density = logit(m_real / n(n−1))).

Parameters exposed to the optimiser
  θ_mut   : reciprocity parameter   (mutual dyads)
  θ_star  : GWD out-degree parameter
"""

import numpy as np
import igraph as ig
cimport numpy as np
from libc.math cimport exp, log

# ── Module-level C constants ──────────────────────────────────────────────────
# α = ln 2, recommended in §3.1 of [2] (λ = 2).
cdef double ALPHA   = 0.6931471805599453    # ln(2)
cdef double E_ALPHA = 0.5                    # e^{−α}
cdef double FACTOR  = 0.5                    # 1 − e^{−α}  (prefactor in Δu3)


def ergm_model(int n, int m_real, double theta_mut, double theta_star):
    """
    Draw a directed graph from the ERGM distribution via Metropolis-Hastings.

    Parameters
    ----------
    n           : number of nodes
    m_real      : target arc count (calibrates θ_density)
    theta_mut   : reciprocity parameter (θ for mutual dyads, u2)
    theta_star  : geometrically weighted out-degree parameter (θ for u3)

    Returns
    -------
    ig.Graph   directed, n nodes, approximately m_real arcs
    """
    # ── Declare all C-level variables up front (Cython requirement) ───────────
    cdef:
        int    step, src, tgt, cur, sign, out_red, n_steps
        double theta_density, p_target
        double d1, d2, d3, log_odds
        np.int32_t[:, :]  adj
        np.int32_t[:]     out_deg
        double[:]         rvals
        np.int64_t[:]     src_arr, tgt_arr

    # ─────────────────────────────────────────────────────────────────────────
    # 1.  CALIBRATE DENSITY PARAMETER
    # ─────────────────────────────────────────────────────────────────────────
    # For the null model (θ_mut = θ_star = 0) this gives exactly the desired
    # expected arc count.  With nonzero θ_mut / θ_star the actual density will
    # differ slightly; the optimiser compensates by adjusting the parameters.
    p_target      = float(m_real) / max(n * (n - 1), 1)
    p_target      = max(1e-9, min(1.0 - 1e-9, p_target))
    theta_density = log(p_target / (1.0 - p_target))

    # ─────────────────────────────────────────────────────────────────────────
    # 2.  INITIALISE ADJACENCY MATRIX  (§ "Simulation" of [1])
    # ─────────────────────────────────────────────────────────────────────────
    # Seed at the target Bernoulli density so the chain starts close to the
    # target distribution and requires less burn-in.
    rng    = np.random.default_rng()
    adj_np = (rng.random((n, n)) < p_target).astype(np.int32)
    np.fill_diagonal(adj_np, 0)      # no self-loops
    adj    = adj_np                  # typed memoryview (no copy)

    out_deg_np = adj_np.sum(axis=1, dtype=np.int32)
    out_deg    = out_deg_np

    # ─────────────────────────────────────────────────────────────────────────
    # 3.  PRE-GENERATE RANDOM VARIATES FOR THE MARKOV CHAIN
    # ─────────────────────────────────────────────────────────────────────────
    # 2 × 10^6 steps.  At ~10^8 pure-C iterations / second this runs in
    # ≈ 20 ms regardless of n.  Pre-generation avoids Python RNG overhead
    # inside the tight inner loop.
    n_steps = 2_000_000
    rvals   = rng.random(n_steps)
    src_arr = rng.integers(0, n, n_steps, dtype=np.int64)
    tgt_arr = rng.integers(0, n, n_steps, dtype=np.int64)

    # ─────────────────────────────────────────────────────────────────────────
    # 4.  METROPOLIS-HASTINGS SWEEP  (§5.1 of [1]; eq. 16–17)
    # ─────────────────────────────────────────────────────────────────────────
    # Proposal: pick (src, tgt) uniformly from all n(n−1) ordered pairs and
    # toggle the arc src → tgt.  Symmetry of q implies acceptance ratio:
    #
    #   A(y → y*) = min(1,  exp(θ · Δu))
    #
    # Change statistics for the three sufficient statistics:
    #
    #   Δu1 = sign                                        (eq. 7 of [1])
    #   Δu2 = sign · y_{tgt,src}                         (eq. 8 of [1])
    #   Δu3 = −sign · (1−e^{−α}) · e^{−α·ỹ_{src}+}     (eq. 17/35 of [2])
    #
    # where  sign = +1  when adding,  −1  when removing.
    for step in range(n_steps):
        src = <int>src_arr[step]
        tgt = <int>tgt_arr[step]
        if src == tgt:
            continue

        cur  = adj[src, tgt]
        sign = 1 - 2 * cur          # +1 adding, −1 removing

        # Δu1  ───────────────────────────────────────────────────────────────
        d1 = <double>sign

        # Δu2: mutual dyads (eq. 8 of [1]) ──────────────────────────────────
        # Adding arc src→tgt creates a mutual dyad iff tgt→src already exists.
        d2 = sign * adj[tgt, src]

        # Δu3: GWD out-degree change (eq. 17 / eq. 35 of [2]) ───────────────
        # ỹ_{src}+ = out-degree of src in the reduced graph where arc is 0.
        # Reduced out-degree = out_deg[src] − y_{src,tgt}  (i.e. − cur).
        out_red = out_deg[src] - cur
        d3 = -sign * FACTOR * exp(-ALPHA * out_red)

        # Log acceptance ratio ────────────────────────────────────────────────
        log_odds = theta_density * d1 + theta_mut * d2 + theta_star * d3

        # Accept / reject (Metropolis-Hastings rule) ──────────────────────────
        if log_odds >= 0.0 or rvals[step] < exp(log_odds):
            adj[src, tgt]  = 1 - cur
            out_deg[src]  += sign

    # ─────────────────────────────────────────────────────────────────────────
    # 5.  BUILD AND RETURN igraph
    # ─────────────────────────────────────────────────────────────────────────
    rows, cols = np.where(adj_np)
    return ig.Graph(
        n     = n,
        edges = list(zip(rows.tolist(), cols.tolist())),
        directed = True,
    )
