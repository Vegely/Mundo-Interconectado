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

async def fetch_dependencies(session, pkg_name, semaphore):
    """Fetches JSON data safely using a semaphore to prevent server bans."""
    url = f"https://pypi.org/pypi/{pkg_name}/json"
    
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        return pkg_name, data.get("info", {}).get("requires_dist")
                    elif response.status == 429: 
                        await asyncio.sleep(2 ** attempt) 
                        continue
                    else:
                        return pkg_name, None
            except Exception:
                await asyncio.sleep(1)
        return pkg_name, None 

async def main():
    edges = []
    visited = set()
    
    # Initialize the queue with ALL of our seed packages
    queue = set(SEED_PACKAGES)
    
    print(f"--- Starting multi-seed crawl up to {MAX_NODES} nodes ---")
    print(f"Seeds: {', '.join(SEED_PACKAGES)}")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        while queue and len(visited) < MAX_NODES:
            batch_size = min(MAX_CONCURRENT_REQUESTS * 2, MAX_NODES - len(visited))
            current_batch = [queue.pop() for _ in range(min(batch_size, len(queue)))]
            
            visited.update(current_batch)
            
            tasks = [fetch_dependencies(session, pkg, semaphore) for pkg in current_batch]
            results = await asyncio.gather(*tasks)
            
            for current_pkg, requires_dist in results:
                if requires_dist:
                    for dep in requires_dist:
                        dep_name = re.split(r'[\s=><(;]', dep)[0].strip()
                        
                        if dep_name and "extra" not in dep_name:
                            edges.append({"Source": current_pkg, "Target": dep_name})
                            
                            if dep_name not in visited:
                                queue.add(dep_name)
                                
            print(f"Nodes fully processed: {len(visited)} / {MAX_NODES}... (Queue size: {len(queue)})")

    # Data Export
    print(f"\nScraping complete in {time.time() - start_time:.2f} seconds.")
    
    if edges:
        df = pd.DataFrame(edges).drop_duplicates()
        df.to_csv("pypi_multiseed_10k.csv", index=False)
        print(f"Saved CSV: 'pypi_multiseed_10k.csv' with {len(df)} total edges.")

        G = nx.from_pandas_edgelist(df, source='Source', target='Target', create_using=nx.DiGraph())
        nx.write_graphml(G, "pypi_multiseed_10k.graphml")
        print(f"Saved GraphML: 'pypi_multiseed_10k.graphml' with {G.number_of_nodes()} nodes.")
    else:
        print("\nFailed to find dependencies.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted. Note: Run the script fully to save the files.")