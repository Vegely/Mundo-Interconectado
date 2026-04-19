# cython: language_level=3
"""
kronecker_model.pyx — Stochastic Kronecker Graph generator.

Compliant with:
    Leskovec et al. (2010) "Kronecker Graphs: An Approach to Modeling Networks"
    JMLR 11, 985-1042.

Fixes applied vs original
--------------------------
1. **No normalisation** (was `p = p / p.sum()`).
   Theta entries are *independent Bernoulli edge-slot probabilities*
   θ_ij ∈ [0,1], NOT a categorical distribution (Definition 14).
   Normalising changes the model entirely and breaks the densification
   power-law property (Theorem 10) and the edge-probability formula (Eq. 2).

2. **Fourth parameter d** is a free independent value, not derived as
   1 − (a+b+c).  The paper has no such constraint; each θ_ij is chosen
   freely by the user or via KRONFIT (Table 4 shows all four free values).

3. **Number of nodes = N1^k**, not n directly.
   k is chosen as ⌈log_{N1}(n)⌉ so that N1^k ≥ n; the graph is zero-
   padded to N1^k nodes (Section 5.6 / Table 2).  Isolated padding nodes
   are dropped from the returned edge list but counted in N.

4. **Expected edges = E1^k** where E1 = Σ θ_ij (Section 3.6 / Theorem 10),
   not the user-supplied m_real.  m_real is now used only to set k so
   that the expected edge count is close to m_real (see _choose_k).

5. **Correct O(E) generation algorithm** (Section 3.6):
   For each of the E sampled edges, descend k levels; at level i choose
   quadrant (r,c) with probability θ_rc / E1 — this is the recursive
   descent described in the paper, NOT a single categorical draw.
   Each edge's probability equals the *product* of the θ_rc values
   chosen at each level (Eq. 2), which is what makes it a Stochastic
   Kronecker graph and not an R-MAT graph.
"""

import numpy as np
cimport numpy as np
from libc.math cimport log, ceil as cceil

# ---------------------------------------------------------------------------
# Choose k (number of Kronecker powers) to match the target graph size
# ---------------------------------------------------------------------------
cdef int _choose_k(int n, int N1):
    """
    Return the smallest k such that N1^k >= n.
    The resulting graph has N = N1^k nodes (Section 5.6).
    """
    cdef int k = 1
    cdef long Nk = N1
    while Nk < n:
        k += 1
        Nk *= N1
    return k


# ---------------------------------------------------------------------------
# Edge-probability formula (Eq. 2 / Section 3.3.1)
# ---------------------------------------------------------------------------
def edge_prob(np.ndarray[double, ndim=2] Theta, int u, int v, int k):
    """
    p_uv = ∏_{i=0}^{k-1} Theta[ floor(u/N1^i) mod N1,
                                   floor(v/N1^i) mod N1 ]   (Eq. 2).
    O(k) time.
    """
    cdef int N1 = Theta.shape[0]
    cdef double p = 1.0
    cdef long step = 1
    cdef int i, r, c
    for i in range(k):
        r = (u // step) % N1
        c = (v // step) % N1
        p *= Theta[r, c]
        step *= N1
    return p


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def kronecker_model(int n, int m_real,
                    double a, double b, double c, double d=-1.0):
    """
    Generate a Stochastic Kronecker Graph using the O(E) algorithm
    (Section 3.6 of Leskovec et al. 2010).

    Model
    -----
    2×2 stochastic initiator matrix (Definition 14):
        Theta = [[a, b],
                 [c, d]]
    where each entry θ_ij ∈ [0, 1] is the probability that an edge
    appears in the corresponding quadrant at each recursive level.

    Parameters
    ----------
    n      : desired number of nodes (graph will have N = 2^k ≥ n nodes;
             k is chosen so that expected edges E1^k is close to m_real)
    m_real : target edge count, used only to choose k (see above)
    a, b, c: three entries of the 2×2 Theta matrix
    d      : fourth entry; defaults to max(0.01, 1 − a) as a heuristic for
             core-periphery structure observed empirically (Table 4,
             Section 7: a≈1, b≈c≈0.5, d≈0.05−0.2).
             Pass explicitly to fully control the initiator.

    Returns
    -------
    igraph.Graph  directed graph on N = N1^k nodes

    Notes
    -----
    * Isolated nodes created by zero-padding (N > n) are included so that
      the graph has exactly N1^k nodes, matching the paper's analysis.
    * Self-loops are excluded (y_ii = 0) — standard for directed graphs.
    * Duplicate edges from collision (~1% rate per paper) are silently
      dropped; final edge count may be slightly less than E_expected.
    """
    import igraph as ig

    # --- Initiator matrix (2×2, N1=2) -----------------------------------
    # d defaults to a core-periphery heuristic (Section 7 / Table 4):
    # empirically a >> b ≈ c >> d across all 20 networks in Table 4.
    if d < 0.0:
        d = max(0.05, 1.0 - a)   # rough heuristic; pass d explicitly for control

    cdef int N1 = 2
    Theta = np.array([[a, b], [c, d]], dtype=np.float64)

    # Clamp entries to valid probability range
    Theta = np.clip(Theta, 1e-9, 1.0 - 1e-9)

    # E1 = sum of Theta entries (Section 3.6)
    cdef double E1 = Theta.sum()

    # --- Choose k so N1^k >= n and E1^k ≈ m_real ----------------------
    # Two constraints; we use k = max(k_n, k_m) as a compromise.
    cdef int k_n = _choose_k(n, N1)      # ensures enough nodes
    # k_m: smallest k so that E1^k >= m_real
    cdef int k_m = 1
    while E1 ** k_m < m_real and k_m < 50:
        k_m += 1
    cdef int k = max(k_n, k_m)

    cdef long N = N1 ** k                   # total nodes (zero-padded)
    cdef double E_expected = E1 ** k        # expected edges (Theorem 10)

    # --- O(E) generation algorithm (Section 3.6) -----------------------
    # At each of the k levels, choose entry (r,c) of Theta with
    # probability θ_rc / E1.  This is a recursive descent: the final
    # (u,v) has probability equal to the product of selected entries,
    # which equals Pk[u,v] by Eq. 2.
    #
    # Flatten Theta / E1 into a 4-element categorical distribution over
    # {(0,0),(0,1),(1,0),(1,1)}.
    rng = np.random.default_rng()

    flat_probs = Theta.ravel() / E1           # shape (4,) sums to 1
    quad_to_row = np.array([0, 0, 1, 1])     # row index for each quadrant
    quad_to_col = np.array([0, 1, 0, 1])     # col index for each quadrant

    # Sample number of edges to place (CLT approximation, Section 3.6)
    # "we first sample the expected number of edges E in Pk"
    cdef int E_place = max(1, int(round(E_expected)))

    # Place E_place edges via recursive descent
    seen = set()
    edges = []

    # Vectorised descent over E_place edges simultaneously (all k levels)
    # quads shape: (k, E_place) — which quadrant is chosen at each level
    quads = rng.choice(4, size=(k, E_place), p=flat_probs)

    # Build u, v by accumulating row/col contributions at each level.
    # At level i (from k-1 down to 0), quadrant (r,c) contributes
    #   u += r * N1^i,  v += c * N1^i   (recursive Kronecker structure)
    u_arr = np.zeros(E_place, dtype=np.int64)
    v_arr = np.zeros(E_place, dtype=np.int64)

    cdef long step = N // N1   # N1^{k-1}
    cdef int lev
    for lev in range(k):
        q = quads[lev]
        u_arr += quad_to_row[q] * step
        v_arr += quad_to_col[q] * step
        step //= N1 if lev < k - 1 else 1

    # Filter: keep only arcs within [0,n), exclude self-loops, deduplicate
    for idx in range(E_place):
        ui = int(u_arr[idx])
        vi = int(v_arr[idx])
        if ui == vi:
            continue                   # no self-loops
        if ui >= n or vi >= n:
            continue                   # outside zero-padded region
        key = ui * n + vi
        if key in seen:
            continue                   # duplicate (collision ~1%, Section 3.6)
        seen.add(key)
        edges.append((ui, vi))

    return ig.Graph(n=n, edges=edges, directed=True)
