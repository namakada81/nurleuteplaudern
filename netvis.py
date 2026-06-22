import streamlit as st
import streamlit.components.v1 as components
import pickle
import networkx as nx
from pyvis.network import Network
from collections import Counter
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

@st.cache_data
def load_data():
    with open("results_no_spellcheck_150.pickle", "rb") as f:
        return pickle.load(f)

def filter_distinct_words(results, max_subreddit_percentage, keep_top_n):
    # Make a copy so we don't mutate the cached original dict
    results_copy = {k: dict(v) for k, v in results.items()}
    
    total_subreddits = len(results_copy)
    max_allowed_subs = total_subreddits * max_subreddit_percentage
    
    # 1. Count how many subreddits each word appears in
    word_doc_freq = Counter()
    for s_id, value in results_copy.items():
        for word in value['counter'].keys():
            word_doc_freq[word] += 1
            
    # 2. Identify words that exceed the percentage threshold
    words_to_remove = set(
        word for word, count in word_doc_freq.items() 
        if count > max_allowed_subs
    )

    # 3. Filter the subreddits
    filtered_subreddits = {}
    for s_id, value in results_copy.items():
        clean_words = {
            word: relevance 
            for word, relevance in value['counter'].items() 
            if word not in words_to_remove
        }
        top_words = dict(Counter(clean_words).most_common(keep_top_n))
        filtered_subreddits[s_id] = top_words
        
    return filtered_subreddits

def visualize_network(similarity_df, threshold, max_edges_per_node, subreddit_data, remove_isolated_nodes=False):
    G = nx.Graph()
    
    # --- NEW: Add Node Tooltips ---
    for sub in similarity_df.index:
        top_words = []
        if sub in subreddit_data:
            # Sort this subreddit's words by relevance score
            sorted_words = sorted(subreddit_data[sub].items(), key=lambda item: item[1], reverse=True)
            # Grab the top 10 words
            top_words = [word for word, score in sorted_words[:10]]
            
        # Use plain text formatting with \n
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

    # Sort edges by highest similarity score first
    edges.sort(reverse=True, key=lambda x: x[0])

    for score, sub_A, sub_B in edges:
        if G.degree(sub_A) < max_edges_per_node and G.degree(sub_B) < max_edges_per_node:
            
            # Calculate shared words for the edge tooltip
            words_A = set(subreddit_data[sub_A].keys())
            words_B = set(subreddit_data[sub_B].keys())
            shared = words_A.intersection(words_B)
            
            # Rank shared words by their combined frequency/relevance score
            shared_scored = [
                (w, subreddit_data[sub_A][w] + subreddit_data[sub_B][w]) 
                for w in shared
            ]
            shared_scored.sort(key=lambda x: x[1], reverse=True)
            
            # Get top 10 shared words
            top_shared = [w for w, s in shared_scored[:10]]
            
            # --- FIX: Use \n instead of HTML tags ---
            if top_shared:
                edge_tooltip = f"Similarity: {score:.3f}\nTop Shared Words:\n" + "\n".join(f"- {w}" for w in top_shared)
            else:
                edge_tooltip = f"Similarity: {score:.3f}\nNo top words shared."
            
            # Add edge with the plain-text tooltip
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

def main():
    st.set_page_config(layout="wide")
    st.title("Reddit Subreddit Clusters")

    # --- Sidebar Controls ---
    st.sidebar.header("Adjust Parameters")
    
    max_subreddit_percentage = st.sidebar.slider(
        "Max Subreddit Appearance (%)", 
        min_value=0.01, max_value=1.00, value=0.90, step=0.01,
        help="If a word appears in more than this % of subreddits, it is removed as noise."
    )

    keep_top_n = st.sidebar.slider(
        "Keep Top N Words per Sub", 
        min_value=10, max_value=500, value=100, step=10,
        help="How many unique words to keep for defining each subreddit's 'topic'."
    )
    
    st.sidebar.markdown("---")
    
    threshold = st.sidebar.slider(
        "Similarity Threshold", 
        min_value=0.00, max_value=1.00, value=0.01, step=0.01,
        help="Minimum cosine similarity score required to draw an edge."
    )
    
    max_edges = st.sidebar.slider(
        "Max Edges per Node", 
        min_value=1, max_value=20, value=5, step=1,
        help="Strict limit on connections to prevent central 'blob' hubs."
    )
    
    remove_isolated = st.sidebar.checkbox("Remove Isolated Nodes", value=False)

    # --- Data Pipeline ---
    try:
        results = load_data()
    except FileNotFoundError:
        st.error("Could not find 'results_no_spellcheck_150.pickle'. Please ensure it is in the same directory.")
        return

    with st.spinner("Recalculating Matrix..."):
        subreddit_data = filter_distinct_words(
            results, 
            max_subreddit_percentage=max_subreddit_percentage, 
            keep_top_n=keep_top_n
        )

        subreddit_names = list(subreddit_data.keys())
        word_score_dicts = list(subreddit_data.values())

        vectorizer = DictVectorizer(sparse=True)
        sparse_matrix = vectorizer.fit_transform(word_score_dicts)

        n_components = min(100, sparse_matrix.shape[0] - 1)
        if n_components > 0:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            lsa_matrix = svd.fit_transform(sparse_matrix)

            similarity_matrix = cosine_similarity(lsa_matrix)
            #similarity_matrix = cosine_similarity(sparse_matrix)
            similarity_df = pd.DataFrame(
                similarity_matrix, index=subreddit_names, columns=subreddit_names
            )
        else:
            st.warning("Not enough data to run SVD. Try adjusting parameters.")
            st.stop()

    # --- Rendering Pyvis Graph ---
    with st.spinner("Generating Graph... Hover over edges to see shared words!"):
        html_file_path = visualize_network(
            similarity_df, 
            threshold=threshold, 
            max_edges_per_node=max_edges, 
            subreddit_data=subreddit_data, # Passed in to calculate overlaps
            remove_isolated_nodes=remove_isolated
        )
        
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
            
        components.html(html_data, height=750)

    # --- NEW: Explicit Streamlit UI for Deep-Diving Overlaps ---
    st.markdown("---")
    st.subheader("Deep-Dive: Subreddit Word Overlap")
    st.markdown("Select two subreddits to see all the shared vocabulary driving their connection.")
    
    col1, col2 = st.columns(2)
    with col1:
        sub_a = st.selectbox("Select Subreddit A", options=subreddit_names, index=0)
    with col2:
        # Default to a different index to avoid matching sub_a immediately
        sub_b = st.selectbox("Select Subreddit B", options=subreddit_names, index=min(1, len(subreddit_names)-1))
        
    if sub_a and sub_b and sub_a != sub_b:
        words_A = set(subreddit_data[sub_a].keys())
        words_B = set(subreddit_data[sub_b].keys())
        shared_words = words_A.intersection(words_B)
        
        if shared_words:
            # Sort them by their combined scores in both subreddits
            scored_overlap = [
                (w, subreddit_data[sub_a][w], subreddit_data[sub_b][w]) 
                for w in shared_words
            ]
            scored_overlap.sort(key=lambda x: x[1] + x[2], reverse=True)
            
            st.success(f"Found **{len(shared_words)}** shared words between r/{sub_a} and r/{sub_b}.")
            
            # Display neatly in a dataframe
            df_overlap = pd.DataFrame(scored_overlap, columns=["Word", f"Score in {sub_a}", f"Score in {sub_b}"])
            st.dataframe(df_overlap, use_container_width=True)
        else:
            st.warning("No shared words found based on the current filtering parameters.")

if __name__ == "__main__":
    main()