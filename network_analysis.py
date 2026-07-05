# Network analysis on the subreddit similarity graph.
# Run directly to print the results, or import analyze(sim) to reuse it in the app.

import numpy as np
import networkx as nx
from networkx.algorithms import community


# k = 8 CHANGE HERE: It defines how many neighbors each subreddit keeps in the graph.
# Alters the density and structure significantly. 
# 8 is a good default; tradeoff: connectedness vs. sparsity.
def analyze(sim, k=8):
    """Run community detection, betweenness and percolation on a similarity matrix.
    Returns the results as a text report (so it can be printed or shown in the app)."""
    subreddits = list(sim.index)        # Get subreddit names list
    matrix = sim.to_numpy()             # Convert to NumPy array
    report = []

    #----------------------------------------------------
    # Build a k-nearest-neighbour graph: each subreddit keeps its k most similar ones.
    # This avoids an arbitrary similarity threshold and keeps the graph connected.
    graph = nx.Graph()
    graph.add_nodes_from(subreddits)            
    for sub in subreddits:                  # Loop through each subreddit
        neighbours = sim.loc[sub].drop(sub).nlargest(k)         # Find top k neighbors
        for other in neighbours.index:
            graph.add_edge(sub, other, weight=float(sim.loc[sub, other]))       # Connect nodes with weight

    #----------------------------------------------------
    # Community detection + modularity (how cleanly language splits Reddit into groups)
    communities = community.louvain_communities(graph, weight="weight", seed=42)    # Find subreddit clusters
    modularity = community.modularity(graph, communities, weight="weight")          # Measure likeness of clusters
    report.append(f"Detected {len(communities)} Distinct Language Cultures (Communities). modularity Q = {modularity:.3f}")
    for i, group in enumerate(sorted(communities, key=len, reverse=True)):      # Format clustering summary text
        members = sorted(group)                                                 # Sort members alphabetically
        report.append(f"  [{i}] ({len(group)}) {', '.join(members[:8])}")

    #----------------------------------------------------
    # Betweenness centrality: the "bridge" subreddits that connect different groups
    report.append("")
    report.append(f"Language Translators: Bridges (betweenness), {nx.number_connected_components(graph)} component(s):")
    betweenness = nx.betweenness_centrality(graph, weight="weight")         # Calculate node bridge/betweenness scores
    bridges = sorted(betweenness.items(), key=lambda item: item[1], reverse=True)
    for sub, score in bridges[:10]:                 # Loop top ten bridges
        report.append(f"  {score:.3f}  r/{sub}")

    #----------------------------------------------------
    # LEAVING THIS OUT for report, since it mostly confuses the user.
    # Percolation: lower the similarity threshold step by step and watch the graph grow.
    # We also remember the first (highest) threshold at which each subreddit gets an edge.
    #report.append("")
    #report.append("Percolation (threshold -> components, largest):")
    first_edge = {}
    n = len(subreddits)
    for t in np.round(np.linspace(0.9, 0.1, 9), 2):     # Loop decreasing similarity thresholds
        step_graph = nx.Graph()                         # temp graph
        step_graph.add_nodes_from(subreddits)           # Add all subreddit nodes
        for i in range(n):                              # Loop matrix rows, columns
            for j in range(i + 1, n):
                if matrix[i, j] >= t:                       # Check similarity exceeds threshold
                    step_graph.add_edge(subreddits[i], subreddits[j])       # Connect similar subreddits
        components = list(nx.connected_components(step_graph))          # Find isolated sub-graphs
        largest = max(len(c) for c in components)                       # Find biggest component size
        #report.append(f"  t={t:.2f}: {len(components)} components, largest = {largest}")
        for sub in subreddits:
            if sub not in first_edge and step_graph.degree(sub) > 0:
                first_edge[sub] = t

    # Islands: subreddits that only connect at a low threshold, or never.
    # A subreddit that never connects gets -1 so it sorts to the top as most isolated.
    report.append("")
    report.append("Unique language: Most isolated subreddits (linguistic islands):")
    islands = sorted(subreddits, key=lambda sub: first_edge.get(sub, float('inf')))       # Sort by isolation level
    for sub in islands[:10]:                                # Loop top ten isolated
        threshold = first_edge.get(sub, "never")
        report.append(f"  first edge at t={threshold}  r/{sub}")

    return "\n".join(report)


if __name__ == "__main__":
    import pickle
    from netvis import pre_filter_data, generate_similarity_matrix

    # Settings (same defaults as the app)
    NUM_SUBS = 100
    MAX_PCT = 0.80
    TOP_N = 100
    REP = "tfidf"
    USE_SVD = True

    with open("results_0.0000001_300_freq.pickle", "rb") as f:
        data = pickle.load(f)
    data = {name: {**val, "counter": dict(val["counter"])} for name, val in data.items()}
    pre_filter_data(data, num_subs=NUM_SUBS, remove_standard=False,
                    standard_words=set(), max_pct=MAX_PCT, top_n=TOP_N)
    sim, _ = generate_similarity_matrix(data, representation=REP, use_svd=USE_SVD)

    print(analyze(sim))
