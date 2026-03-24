import networkx as nx

# 1. Define the files to merge
files_to_merge = [
    'USA_South_highvoltage.graphml', 
    'USA_Midwest_highvoltage.graphml', 
    'USA_Northeast_highvoltage.graphml'
]

graphs = []

# 2. Load each file
for file in files_to_merge:
    print(f"Loading {file}...")
    try:
        graphs.append(nx.read_graphml(file))
    except FileNotFoundError:
        print(f"  -> Error: Could not find {file}. Check the name/path.")

# 3. Combine them into a single graph
print("\nCombining graphs...")
# nx.compose_all stitches them together seamlessly
G_usa = nx.compose_all(graphs)

# 4. Save the combined graph to a new GraphML file
output_file = 'USA_combined_highvoltage.graphml'
nx.write_graphml(G_usa, output_file)

print(f"\nSuccess! Combined graph saved as: {output_file}")
print(f"Total combined nodes: {G_usa.number_of_nodes()}")
print(f"Total combined edges: {G_usa.number_of_edges()}")