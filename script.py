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


def mine_words(min_relevance):
    ds = load_dataset("webis/tldr-17", revision="refs/convert/parquet")
    ds = ds['train']
    df = ds.to_pandas()
    print(len(ds))

    nltk_words = [word.lower() for word in nltk.corpus.words.words()]
    spell = SpellChecker()
    spell.word_frequency.load_words(nltk_words)

    id_to_name = df.drop_duplicates(subset=['subreddit_id']).set_index('subreddit_id')['subreddit'].to_dict()

    top_100_subs = df['subreddit_id'].value_counts().head(100).index.tolist()
    total_freq = Counter()

    results = {}
    print(top_100_subs)
    for s_id in top_100_subs:
        # find common words in the subreddit
        filtered = df[df['subreddit_id'] == s_id]
        full_freq = clean_and_count(filtered)

        # apply cutoff to only consider the most common words
        subreddit_words_with_relevance = {word: count / full_freq.N() for word, count in full_freq.items() if word not in spell and count / full_freq.N() >= min_relevance}
        final_flagged = Counter(subreddit_words_with_relevance)
        total_freq.update(final_flagged)
        results[s_id] = final_flagged

        print(f'{id_to_name[s_id]}({len(final_flagged)}): {final_flagged.most_common()}')
        
    results['total'] = total_freq
    print(f'total log freq: {total_freq.most_common()}')

    with open("results_nltk_corpus.pickle", "wb") as f:
        pickle.dump(results, f)

def filter_distinct_words(results, remove_top_n = 200, keep_top_n=50):
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
    print(f"Removing the top {remove_top_n} globally noisy words...")

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

def visualize_network(similarity_df, threshold=0.3):
    """
    Builds an interactive HTML network graph from the similarity matrix.
    Only draws lines between subreddits if their similarity > threshold.
    """
    print(f"Generating network graph (Threshold: {threshold})...")
    
    # 1. Initialize a NetworkX graph
    G = nx.Graph()
    
    # 2. Add nodes (subreddits)
    for sub in similarity_df.index:
        G.add_node(sub, title=sub) # 'title' is the hover tooltip in PyVis

    # 3. Add edges (Combined Top-K and Threshold)
    K = 3               # Max connections per subreddit
    threshold = 0.15    # Minimum similarity score required
    
    for target_sub in similarity_df.index:
        # Sort similarities for this sub, skip the first one (which is itself)
        top_matches = similarity_df[target_sub].sort_values(ascending=False)[1:K+1]
        
        for neighbor_sub, score in top_matches.items():
            # Only connect them if they pass the minimum threshold
            if score >= threshold:
                # NetworkX automatically handles duplicates (A->B is the same as B->A)
                G.add_edge(target_sub, neighbor_sub, weight=score, value=score * 5)

    # 4. Clean up orphans (nodes that failed the threshold test)
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes)} unconnected subreddits from the view.")
                
    # 4. Create the PyVis network from NetworkX
    # physics=True makes the nodes dynamically push and pull each other
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    
    # 5. Save and open the interactive HTML file
    net.show("subreddit_clusters.html", notebook=False)
    print("Graph saved to subreddit_clusters.html! Open it in your browser.")

def analyze():
    with open("results_cleaned.pickle", "rb") as f:
        results = pickle.load(f)

    subreddit_data = filter_distinct_words(results, remove_top_n=20, keep_top_n=100)

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

    visualize_network(similarity_df, threshold=0.2)

    # --- View the Results ---
    if len(subreddit_names) > 0:
        target_sub = subreddit_names[0] # Just grabbing the first subreddit to test
        print(f"\nTop 3 subreddits most similar to {target_sub}:")
        
        # Sort descending, slice [1:4] to skip itself (1.0 similarity)
        similar_subs = similarity_df[target_sub].sort_values(ascending=False)[1:4]
        
        for sub, score in similar_subs.items():
            print(f"  {sub}: {score:.3f} similarity")

analyze()

#mine_words(0.00001)