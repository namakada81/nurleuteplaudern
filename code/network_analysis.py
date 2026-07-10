# Network analysis on the subreddit similarity graph.
# Run directly to print the results, or import analyze(sim) to reuse it in the app.

import numpy as np
import networkx as nx
from networkx.algorithms import community


# k = 8 It defines how many neighbors each subreddit keeps in the graph.
# Alters the density and structure significantly. 
# 8 is a good default; tradeoff: connectedness vs. sparsity.
def analyze(sim, graph_similarity_threshold, k=8):
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
            if sim.loc[sub, other] > graph_similarity_threshold:  # Only add edges with significant similarities
                graph.add_edge(sub, other, weight=float(sim.loc[sub, other]))       # Connect nodes with weight

    #----------------------------------------------------
    # Community detection + modularity (how cleanly language splits Reddit into groups)
    communities = community.louvain_communities(graph, weight="weight", seed=42)    # Find subreddit clusters
    modularity = community.modularity(graph, communities, weight="weight")          # Measure likeness of clusters
    actual_communities = [group for group in communities if len(group) > 1]  # Filter out singletons
    report.append(f"Detected {len(actual_communities)} Distinct Language Cultures (Communities with >1 member). modularity Q = {modularity:.3f}")
    for i, group in enumerate(sorted(actual_communities, key=len, reverse=True)):      # Format clustering summary text
        members = sorted(group)                                                 # Sort members alphabetically
        report.append(f"  [{i}] ({len(group)}) {', '.join(members[:12])}")

    #----------------------------------------------------
    # Betweenness centrality: the "bridge" subreddits that connect different groups
    report.append("")
    report.append(f"Language Translators: Bridges (betweenness), {nx.number_connected_components(graph)} component(s):")
    betweenness = nx.betweenness_centrality(graph, weight="weight")         # Calculate node bridge/betweenness scores
    bridges = sorted(betweenness.items(), key=lambda item: item[1], reverse=True)
    for sub, score in bridges[:10]:                 # Loop top ten bridges
        report.append(f"  {score:.3f}  r/{sub}")

    # Islands: subreddits whose closest neighbor is furthest away
    # Low max similarity = language resembles no other subreddit
    report.append("")
    report.append("Unique language: Most isolated subreddits (linguistic islands):")
    max_sim = matrix.copy()                 # Copy similarity matrix
    np.fill_diagonal(max_sim, -1)           # Ignore self-similarity (always 1.0)
    best = max_sim.max(axis=1)              # Each subreddit's closest neighbor similarity
    for i in np.argsort(best)[:10]:         # Ten subreddits with the weakest best match
        report.append(f"  max similarity {best[i]:.3f}  r/{subreddits[i]}")

    return "\n".join(report)
