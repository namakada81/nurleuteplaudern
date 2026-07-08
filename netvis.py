import streamlit as st
import streamlit.components.v1 as components
import pickle
import networkx as nx
from pyvis.network import Network
from collections import Counter
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import nltk
from network_analysis import analyze  # network analysis feature (added)

#Start: streamlit run netvis.py

@st.cache_data
def load_data():
    with open("results_0.0000001_300_freq.pickle", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_nltk_words():
    nltk.download('words', quiet=True)
    return set(word.lower() for word in nltk.corpus.words.words())

def visualize_network(similarity_df, threshold, max_edges_per_node, subreddit_data, display_scores, remove_isolated_nodes=False):
    G = nx.Graph()

    for sub in similarity_df.index:
        top_words = []
        if sub in display_scores:
            sorted_words = sorted(display_scores[sub].items(), key=lambda item: item[1], reverse=True)
            top_words = [word for word, score in sorted_words[:10]]

        if top_words:
            node_tooltip = f"r/{sub}\nTop Words:\n" + "\n".join(f"- {w}" for w in top_words)
        else:
            node_tooltip = f"r/{sub}"

        G.add_node(sub, title=node_tooltip)

    edges = []
    subreddits = similarity_df.columns
    for i in range(len(subreddits)):
        for j in range(i + 1, len(subreddits)): 
            score = similarity_df.iloc[i, j]
            if score >= threshold:
                edges.append((score, subreddits[i], subreddits[j]))

    edges.sort(reverse=True, key=lambda x: x[0])

    for score, sub_A, sub_B in edges:
        if G.degree(sub_A) < max_edges_per_node and G.degree(sub_B) < max_edges_per_node:

            words_A = set(display_scores.get(sub_A, {}).keys())
            words_B = set(display_scores.get(sub_B, {}).keys())
            shared = words_A.intersection(words_B)

            shared_scored = [
                (w, display_scores[sub_A].get(w, 0) + display_scores[sub_B].get(w, 0))
                for w in shared
            ]
            shared_scored.sort(key=lambda x: x[1], reverse=True)
            
            top_shared = [w for w, s in shared_scored[:10]]
            
            if top_shared:
                edge_tooltip = f"Similarity: {score:.3f}\nTop Shared Words:\n" + "\n".join(f"- {w}" for w in top_shared)
            else:
                edge_tooltip = f"Similarity: {score:.3f}\nNo top words shared."
            
            G.add_edge(sub_A, sub_B, weight=score, value=score * 5, title=edge_tooltip)

    if remove_isolated_nodes:
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
                
    # Enable interaction physics and tooltips
    net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    
    # Configure Pyvis options
    net.set_options("""
    var options = {
      "interaction": {
        "hover": true,
        "hoverConnectedEdges": true
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": false
      }
    }
    """)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        return tmp.name

def pre_filter_data(data, num_subs, remove_standard, standard_words, max_pct, top_n):
    """
    Consolidates 4 filtering steps into a 2-pass optimization for massive performance gains.
    """
    word_doc_freq = Counter()
    total_valid_subs = 0
    
    # --- PASS 1: Drop Subs, Drop Standard Words, and Count Frequencies ---
    for subreddit in list(data.keys()):
        
        # 1. Subreddit Limit Filter
        if data[subreddit]["size_idx"] >= num_subs:
            del data[subreddit]
            continue # Skip processing words for a deleted sub
            
        total_valid_subs += 1
        counter = data[subreddit]["counter"]
        
        # 2. Standard Word Filter & Document Frequency Counting
        for word in list(counter.keys()):
            if remove_standard and word in standard_words:
                del counter[word]
            else:
                word_doc_freq[word] += 1
                
    # --- PREPARE PASS 2: Calculate Global Noise Threshold ---
    max_allowed_subs = total_valid_subs * max_pct
    words_to_remove = set()
    
    if max_allowed_subs < total_valid_subs:
        words_to_remove = {
            word for word, count in word_doc_freq.items() 
            if count > max_allowed_subs
        }

    # --- PASS 2: Drop Global Noise, Slice to Top N ---
    for subreddit, val in data.items():
        counter = val["counter"]
        
        # 3. Filter Max Subreddit Appearance Noise
        if words_to_remove:
            for word in list(counter.keys()):
                if word in words_to_remove:
                    del counter[word]

        # 4. Keep Top N Words
        # Wrapping in Counter() ensures we can use .most_common() even if it's a standard dict
        val["counter"] = dict(Counter(counter).most_common(top_n))

def generate_similarity_matrix(data, representation="tfidf", use_svd=True, min_relevance=0.0):
    """
    Dynamically builds a similarity matrix based on user-selected techniques.
    Representation options: 'raw', 'relevance', 'tfidf'
    Also returns per-subreddit word scores for display in tooltips.
    """
    subreddit_names = list(data.keys())

    # --- Step 1: Base Representation ---
    if representation == "relevance":
        word_score_dicts = []
        for val in data.values():
            counter = val["counter"]
            total_words = val["full_freq_n"]

            relevance_dict = {
                word: count / total_words
                for word, count in counter.items()
                if (count / total_words) >= min_relevance
            }
            word_score_dicts.append(relevance_dict)
    else:
        # For 'raw' or 'tfidf', we start with the base counts
        word_score_dicts = [val["counter"] for val in data.values()]

    # --- Step 2: Vectorization ---
    vectorizer = DictVectorizer(sparse=True)
    working_matrix = vectorizer.fit_transform(word_score_dicts)

    if working_matrix.shape[1] == 0:
        st.warning("Whoops! Your filters are too aggressive and removed ALL words. Try lowering the relevance threshold or adjusting the filter percentages.")
        st.stop() # This gracefully halts the app without throwing a Python error

    # --- Step 3: TF-IDF (If selected) ---
    if representation == "tfidf":
        tfidf_transformer = TfidfTransformer(sublinear_tf=True)
        working_matrix = tfidf_transformer.fit_transform(working_matrix)

    # Build per-subreddit word score dicts for tooltip display, reflecting the
    # actual representation (raw counts, relevance, or TF-IDF weights).
    feature_names = vectorizer.get_feature_names_out()
    display_scores = {}
    dense = working_matrix.toarray() if hasattr(working_matrix, "toarray") else working_matrix
    for i, name in enumerate(subreddit_names):
        row = dense[i]
        display_scores[name] = {feature_names[j]: float(row[j]) for j in row.nonzero()[0]}

    # --- Step 4: SVD Dimensionality Reduction (If selected) ---
    if use_svd:
        # Bound the components by both rows (subs) AND columns (words)
        n_components = min(100, working_matrix.shape[0] - 1, working_matrix.shape[1] - 1)

        # If the filter left us with 0 or 1 words, SVD is mathematically impossible
        if n_components <= 0:
            st.warning(f"Not enough words left to run SVD (Only {working_matrix.shape[1]} words survived). Try lowering your relevance threshold or adjusting your standard word filters.")
            st.stop()

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        working_matrix = svd.fit_transform(working_matrix)

    # --- Step 5: Cosine Similarity ---
    # working_matrix could now be raw counts, TF-IDF, or SVD components.
    # Cosine similarity works on all of them.
    similarity_matrix = cosine_similarity(working_matrix)

    similarity_df = pd.DataFrame(
        similarity_matrix, index=subreddit_names, columns=subreddit_names
    )

    return similarity_df, display_scores

def main():
    st.set_page_config(layout="wide")
    st.title("Reddit Subreddit Clusters")

    nltk_words = load_nltk_words()
    try:
        tokenized_data = load_data()
    except FileNotFoundError:
        st.error("Could not find data file. Please ensure it is in the same directory.")
        return


    st.sidebar.header("Pre Filtering Controls")
    number_of_subreddits = st.sidebar.slider(
        "Number of Subreddits", 
        min_value=1, max_value=len(tokenized_data), value=min(150,len(tokenized_data)), step=1,
        help="Look only at the top n subreddits with the most posts"
    )
    remove_standard_words = st.sidebar.checkbox("Remove Standard English Words", value=False)
    max_subreddit_percentage = st.sidebar.slider(
        "Max Subreddit Appearance (%)", 
        min_value=0.0, max_value=100.0, value=60.0, step=1.0,
        help="If a word appears in more than this % of subreddits, it is removed as noise."
    )
    keep_top_n = st.sidebar.slider(
        "Keep Top N Words per Sub", 
        min_value=10, max_value=500, value=300, step=10,
        help="How many unique words to keep for defining each subreddit's 'topic'."
    )
    st.sidebar.header("Graph Controls")  
    graph_edge_similarity_threshold = st.sidebar.slider(
        "Similarity Threshold", 
        min_value=0.0, max_value=1.0, value=0.07, step=0.001,
        help="Minimum cosine similarity score required to draw an edge."
    )
    max_edges = st.sidebar.number_input(
        "Max Edges per Node", 
        min_value=1, max_value=20, value=4, step=1,
        help="Strict limit on connections to prevent central 'blob' hubs."
    )
    remove_isolated = st.sidebar.checkbox("Remove Isolated Nodes", value=False)



    st.sidebar.header("Matrix Configuration")
    rep_choice = st.sidebar.selectbox(
        "Word Representation", 
        options=["tfidf", "raw", "relevance"],
        format_func=lambda x: {"tfidf": "TF-IDF", "raw": "Raw Word Counts", "relevance": "Relative Relevance"}[x]
    )
    min_rel = 0.0
    if rep_choice == "relevance":
        min_rel = st.sidebar.slider(
            "Minimum Relevance Threshold (x10000^-1)", 
            min_value=0.0, 
            max_value=5.0,
            value=0.1, 
            step=0.01,
        )
    use_svd_toggle = st.sidebar.checkbox("Use Latent Semantic Analysis", value=True)
    

    # --- Data Pipeline ---
    with st.spinner("Recalculating Matrix..."):
        data = {k: {**v, "counter": dict(v["counter"])} for k, v in tokenized_data.items()}

        pre_filter_data(
            data=data,
            num_subs=number_of_subreddits,
            remove_standard=remove_standard_words,
            standard_words=nltk_words,
            max_pct=max_subreddit_percentage / 100,
            top_n=keep_top_n
        )

        similarity_df, display_scores = generate_similarity_matrix(
            data=data,
            representation=rep_choice,
            use_svd=use_svd_toggle,
            min_relevance=min_rel / 10000
        )

    # --- Rendering Pyvis Graph ---
    with st.spinner("Generating Graph... Hover over edges to see shared words!"):
        html_file_path = visualize_network(
            similarity_df,
            threshold=graph_edge_similarity_threshold,
            max_edges_per_node=max_edges,
            subreddit_data=data,
            display_scores=display_scores,
            remove_isolated_nodes=remove_isolated
        )
        
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_data = f.read()

        components.html(html_data, height=750)

    # --- Network analysis---
    # Runs community detection, betweenness and percolation on the current similarity matrix.
    if st.sidebar.button("Run Network Analysis"):
        st.subheader("Network Analysis (current filter settings)")
        with st.spinner("Analyzing network..."):
            report = analyze(similarity_df, graph_similarity_threshold=graph_edge_similarity_threshold, k=8)
        st.code(report)

if __name__ == "__main__":
    main()