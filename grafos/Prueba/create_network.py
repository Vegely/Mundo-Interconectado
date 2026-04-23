import asyncio
import aiohttp
import pandas as pd
import networkx as nx
import re
import time

# A diverse list of starting points
SEED_PACKAGES = [
    "pandas",       # Data Science
    "tensorflow",   # Machine Learning / AI
    "django",       # Heavy Web Applications
    "fastapi",      # Lightweight Web APIs
    "kivy",         # Mobile/Desktop App UI
    "PyQt5",        # Desktop Applications
    "pygame",       # Game Development
    "ansible",      # DevOps and Server Management
    "pytest",       # Testing framework
    "jupyter"       # Interactive computing
]

MAX_NODES = 10000
MAX_CONCURRENT_REQUESTS = 40

# Set True  → follow optional extras deps too (bigger graph, ~10k nodes).
# Set False → only mandatory runtime deps (smaller but stricter graph).
INCLUDE_EXTRAS = True

# ---------------------------------------------------------------------------
# PEP 503 normalization
# Treats "Sphinx", "sphinx", "sphinx_", "sphinx-" as the same canonical key.
# PyPI itself uses this exact rule for package identity.
# ---------------------------------------------------------------------------
_NORMALIZE_RE = re.compile(r"[-_.]+")

def normalize(name: str) -> str:
    """Return the PEP 503 canonical form of a package name."""
    return _NORMALIZE_RE.sub("-", name).lower()

# ---------------------------------------------------------------------------
# PEP 508 dependency-string parser
#
# requires_dist strings can look like:
#   requests
#   requests>=2.0
#   requests[security]>=2.0          ← extras marker in the name part
#   requests>=2.0; python_version>"3.6"
#   requests>=2.0; extra == "dev"    ← extras-only conditional dep
#   ansible-core~=2.15               ← ~= "compatible release" operator
#
# The old script split on [\s=><(;] which MISSED '~', turning
# "ansible-core~=2.15" into node "ansible-core~" → 404 on PyPI.
#
# _EXTRACT_NAME_RE only matches valid name chars [A-Za-z0-9._-], so every
# version operator (>=, <=, ~=, !=, >, <) or [ bracket naturally stops it.
# ---------------------------------------------------------------------------
_VALID_PKG_RE = re.compile(
    r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?|[A-Za-z0-9])$"
)
_EXTRACT_NAME_RE = re.compile(
    r"^\s*([A-Za-z0-9]([A-Za-z0-9._\-]*[A-Za-z0-9])?)"
)
# PEP 440 version operators — used as a final sanity check
_VERSION_OPERATORS = re.compile(r"[~!<>=]")


def parse_dep_name(dep_str: str) -> str | None:
    """
    Extract the bare package name from a PEP 508 requirement string.

    Returns the normalized name, or None if the string is garbage/invalid.

    When INCLUDE_EXTRAS is False, deps gated solely on `extra == "..."` are
    dropped (optional install-time deps only).  When INCLUDE_EXTRAS is True
    (the default), ONLY the package name is extracted — the marker is ignored —
    so extras deps still contribute nodes and edges to the graph.

    WHY THIS MATTERS FOR GRAPH SIZE:
      - Mandatory-only (INCLUDE_EXTRAS=False): ~150 nodes — the graph hits
        leaf packages (numpy, certifi, six…) quickly and drains.
      - With extras (INCLUDE_EXTRAS=True): ~10k nodes — optional deps like
        numpy→scipy, requests→PySocks, etc. cascade the graph outward.
    """
    if not dep_str or not dep_str.strip():
        return None

    # Split off the environment marker (the part after ';')
    if ";" in dep_str:
        name_part, marker_part = dep_str.split(";", 1)
        # If INCLUDE_EXTRAS is False, skip deps that are extras-only
        if not INCLUDE_EXTRAS and "extra" in marker_part:
            return None
    else:
        name_part = dep_str

    # Extract the leading package name token.
    # Stops naturally at version operators, '[', whitespace, etc.
    m = _EXTRACT_NAME_RE.match(name_part)
    if not m:
        return None

    raw_name = m.group(1)

    # Must look like a real package name
    if not _VALID_PKG_RE.match(raw_name):
        return None

    # Single-char names are not real packages
    if len(raw_name) < 2:
        return None

    norm = normalize(raw_name)

    # Final guard: normalized name must not contain version-operator chars.
    # Catches any exotic leakage (e.g. "ansible-core~" from older parsers).
    if _VERSION_OPERATORS.search(norm):
        return None

    return norm


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------
async def fetch_dependencies(session, pkg_name: str, semaphore):
    """
    Fetch requires_dist for pkg_name from PyPI.

    Returns (normalized_name, canonical_pypi_name, list_of_dep_strings | None).
    The canonical name comes from info.name in the API response — it may
    differ in casing/separators from what we requested.
    """
    norm = normalize(pkg_name)
    # Always build the URL from the NORMALIZED name — this is the final safety
    # net that prevents garbage like "ansible-core~" from ever hitting the
    # network. PyPI accepts normalized names and redirects as needed.
    url = f"https://pypi.org/pypi/{norm}/json"

    async with semaphore:
        for attempt in range(3):
            try:
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        info = data.get("info", {})
                        canonical = info.get("name", pkg_name)  # PyPI's official name
                        return norm, canonical, info.get("requires_dist")
                    elif response.status == 404:
                        return norm, pkg_name, None          # Package doesn't exist
                    elif response.status == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        return norm, pkg_name, None
            except Exception:
                await asyncio.sleep(1)
        return norm, pkg_name, None


# ---------------------------------------------------------------------------
# Main crawl
# ---------------------------------------------------------------------------
async def main():
    edges: list[tuple[str, str]] = []       # (source_norm, target_norm)
    canonical: dict[str, str] = {}          # norm -> canonical PyPI name
    visited: set[str] = set()               # normalized names already fetched

    # Seed queue with normalized names
    queue: set[str] = {normalize(p) for p in SEED_PACKAGES}
    # Pre-populate canonical map with our seeds (will be overwritten on fetch)
    for p in SEED_PACKAGES:
        canonical[normalize(p)] = p

    mode = "all deps (including extras)" if INCLUDE_EXTRAS else "mandatory deps only"
    print(f"--- Starting multi-seed crawl up to {MAX_NODES} nodes [{mode}] ---")
    print(f"Seeds: {', '.join(SEED_PACKAGES)}")
    start_time = time.time()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(connector=connector) as session:
        while queue and len(visited) < MAX_NODES:
            batch_size = min(MAX_CONCURRENT_REQUESTS * 2, MAX_NODES - len(visited))
            current_batch = [queue.pop() for _ in range(min(batch_size, len(queue)))]
            visited.update(current_batch)

            # Fetch using whatever spelling we know; PyPI resolves it.
            tasks = [
                fetch_dependencies(session, canonical.get(norm, norm), semaphore)
                for norm in current_batch
            ]
            results = await asyncio.gather(*tasks)

            for norm_src, canon_src, requires_dist in results:
                # Update canonical map with PyPI's authoritative name
                canonical[norm_src] = canon_src

                if not requires_dist:
                    continue

                for dep_str in requires_dist:
                    norm_dep = parse_dep_name(dep_str)

                    if norm_dep is None:
                        continue

                    # Self-loops are meaningless
                    if norm_dep == norm_src:
                        continue

                    edges.append((norm_src, norm_dep))

                    if norm_dep not in visited:
                        queue.add(norm_dep)

            print(
                f"  Processed: {len(visited):>6} / {MAX_NODES}"
                f"   Queue: {len(queue):>6}"
                f"   Edges so far: {len(edges):>7}"
            )

    elapsed = time.time() - start_time
    print(f"\nScraping complete in {elapsed:.2f} seconds.")

    if not edges:
        print("No edges found — nothing to save.")
        return

    # ------------------------------------------------------------------
    # Build graph using normalized names (deduplicates Sphinx/sphinx/etc.)
    # Add a 'label' attribute with the canonical PyPI spelling for display.
    # ------------------------------------------------------------------
    df = pd.DataFrame(edges, columns=["Source", "Target"]).drop_duplicates()
    df.to_csv("pypi_multiseed_10k.csv", index=False)
    print(f"Saved CSV: 'pypi_multiseed_10k.csv' with {len(df)} edges.")

    G = nx.from_pandas_edgelist(
        df, source="Source", target="Target", create_using=nx.DiGraph()
    )

    # Attach canonical display name as a node attribute
    for node in G.nodes():
        G.nodes[node]["label"] = canonical.get(node, node)

    nx.write_graphml(G, "pypi_multiseed_10k.graphml")
    print(
        f"Saved GraphML: 'pypi_multiseed_10k.graphml' "
        f"with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )

    # Quick sanity report
    print("\n--- Top 10 most-depended-upon packages ---")
    in_deg = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:10]
    for norm_name, deg in in_deg:
        print(f"  {canonical.get(norm_name, norm_name):<30} in-degree: {deg}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted.")