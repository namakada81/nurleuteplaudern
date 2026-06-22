import nltk
from spellchecker import SpellChecker
from collections import Counter
from datasets import load_dataset
import pandas as pd
import math
import pickle

import networkx as nx
from pyvis.network import Network

from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def clean_and_count(df):
    def is_valid_token(token):
        if token.isalpha():
            return True
        
        # 1. Temporarily strip apostrophes to check the base characters
        clean_token = token.replace("'", "")
        if clean_token and clean_token.isalpha():
            return True

        return False
    
    full_freq = nltk.FreqDist()
    tokenizer = nltk.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    for text in df['body']:
        tokens = tokenizer.tokenize(text)
        clean_tokens = [t for t in tokens if is_valid_token(t)]
        full_freq.update(clean_tokens)
    return full_freq


def mine_words(filename, min_relevance=0.00001, top_n_subs=100):
    ds = load_dataset("webis/tldr-17", revision="refs/convert/parquet")
    ds = ds['train']
    df = ds.to_pandas()
    print(len(ds))

    #nltk_words = [word.lower() for word in nltk.corpus.words.words()]
    #spell = SpellChecker()
    #spell.word_frequency.load_words(nltk_words)

    top_subs = df['subreddit'].value_counts().head(top_n_subs).index.tolist()

    results = {}
    print(top_subs)
    for idx, subreddit in enumerate(top_subs):
        # find common words in the subreddit
        filtered = df[df['subreddit'] == subreddit]
        full_freq = clean_and_count(filtered)

        # apply cutoff to only consider the most common words
        subreddit_words_with_relevance = {word: count / full_freq.N() for word, count in full_freq.items() if count / full_freq.N() >= min_relevance}
        final_flagged = Counter(subreddit_words_with_relevance)
        results[subreddit] = {'size_idx': idx, 'counter': final_flagged}

        print(f'{subreddit}({len(final_flagged)}): {final_flagged.most_common()}')

    with open(filename, "wb") as f:
        pickle.dump(results, f)

def filter_distinct_words(results, remove_top_n, keep_top_n):
    """
    Removes universally common words across Reddit and isolates 
    the top highly-specific words for each individual subreddit.
    """
    # Remove the 'total' tracker if it exists
    results.pop('total', None)
    
    # --- 1. Identify Global Noise ---
    global_noise = Counter()
    for s_id, word_frequencies in results.items():
        for word, relevance in word_frequencies.items():
            # We use a fractional power (0.15) to dampen extreme outliers 
            # so one massive subreddit doesn't skew the whole list
            global_noise[word] += relevance ** 0.15
            
    # Extract just the string names of the words to remove
    top_noisy_tuples = global_noise.most_common(remove_top_n)
    words_to_remove = set(word for word, count in top_noisy_tuples)
    print(f"Removing the top {remove_top_n} globally noisy words: {words_to_remove}")

    # --- 2. Filter Subreddits ---
    filtered_subreddits = {}
    for s_id, word_frequencies in results.items():
        
        # Keep words that are NOT in our noise list
        clean_words = {
            word: relevance 
            for word, relevance in word_frequencies.items() 
            if word not in words_to_remove
        }
        
        # Grab the top 50 remaining words for this specific subreddit
        top_words = dict(Counter(clean_words).most_common(keep_top_n))
        filtered_subreddits[s_id] = top_words
        
    return filtered_subreddits

def visualize_network(similarity_df, threshold=0.15, max_edges_per_node=5, remove_isolated_nodes=False):
    """
    Builds an interactive HTML network graph from the similarity matrix.
    Limits connections to a strict maximum per node to prevent central blob hubs.
    """
    print(f"Generating network graph (Threshold: {threshold}, Max Edges: {max_edges_per_node})...")
    
    # 1. Initialize a NetworkX graph
    G = nx.Graph()
    
    # 2. Add nodes (subreddits)
    for sub in similarity_df.index:
        G.add_node(sub, title=sub)

    # 3. Extract all unique pairs of subreddits that pass the threshold
    edges = []
    subreddits = similarity_df.columns
    for i in range(len(subreddits)):
        # Start j at i + 1 to avoid duplicate edges (A->B and B->A) and self-loops (A->A)
        for j in range(i + 1, len(subreddits)): 
            score = similarity_df.iloc[i, j]
            if score >= threshold:
                edges.append((score, subreddits[i], subreddits[j]))

    # 4. Sort all edges globally from strongest similarity to weakest
    edges.sort(reverse=True, key=lambda x: x[0])

    # 5. Add edges sequentially while respecting the strict degree limit
    for score, sub_A, sub_B in edges:
        # Check current connection count (degree) of both nodes
        if G.degree(sub_A) < max_edges_per_node and G.degree(sub_B) < max_edges_per_node:
            G.add_edge(sub_A, sub_B, weight=score, value=score * 5)

    # 6. Clean up orphans (nodes that ended up with 0 connections)
    if remove_isolated_nodes:
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        print(f"Removed {len(isolated_nodes)} unconnected subreddits from the view.")
                
    # 7. Create the PyVis network from NetworkX
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    
    # 8. Save and open the interactive HTML file
    net.show("subreddit_clusters.html", notebook=False)
    print("Graph saved to subreddit_clusters.html! Open it in your browser.")

def analyze():
    with open("results_cleaned.pickle", "rb") as f:
        results = pickle.load(f)

    subreddit_data = filter_distinct_words(results, remove_top_n=200, keep_top_n=100)

    subreddit_names = list(subreddit_data.keys())
    word_score_dicts = list(subreddit_data.values())

    print("Building the Sparse Matrix...")
    vectorizer = DictVectorizer(sparse=True)
    sparse_matrix = vectorizer.fit_transform(word_score_dicts)
    print(f"Original Matrix shape: {sparse_matrix.shape[0]} subreddits x {sparse_matrix.shape[1]} words")

    print("Applying LSA (TruncatedSVD)...")
    # LSA reduces the massive word columns into core "concepts" or "topics"
    # 50 to 100 components is standard for text. It cannot exceed your number of subreddits/words.
    n_components = min(50, sparse_matrix.shape[0] - 1) 
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = svd.fit_transform(sparse_matrix)

    print("Calculating Cosine Similarity...")
    # Calculate similarity on the dense, topic-based LSA matrix instead of the raw words
    similarity_matrix = cosine_similarity(lsa_matrix)

    # Format into a readable Pandas DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix, 
        index=subreddit_names, 
        columns=subreddit_names
    )

    visualize_network(similarity_df, threshold=0.01, max_edges_per_node=5, remove_isolated_nodes=False)

    # --- View the Results ---
    if len(subreddit_names) > 0:
        target_sub = subreddit_names[0] # Just grabbing the first subreddit to test
        print(f"\nTop 3 subreddits most similar to {target_sub}:")
        
        # Sort descending, slice [1:4] to skip itself (1.0 similarity)
        similar_subs = similarity_df[target_sub].sort_values(ascending=False)[1:4]
        
        for sub, score in similar_subs.items():
            print(f"  {sub}: {score:.3f} similarity")

#analyze()

mine_words("results_0.0000001_300.pickle", min_relevance=0.000005, top_n_subs=300)