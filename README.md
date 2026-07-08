# A Map of Reddit, Drawn from Words
_Group members: [Lucas Cimerman], [Lukas Mauz], [Roman Lohmiller]_

Reddit is composed of thousands of distinct communities (subreddits), each with its own topics and jargon. We draw a "map" of Reddit by measuring how similar subreddits are in the language their users write, and visualize the result as an interactive network graph.

## Introduction

Every online community develops its own vocabulary: gamers write about "ganking" and "ults", investors about ETFs and dividends. We exploit this by representing each subreddit purely through its word usage. From the 300 largest subreddits in the Webis TLDR-17 corpus, we extract per-subreddit word frequency vectors, filter out uninformative common words, and compute pairwise cosine similarities. Subreddit pairs above a similarity threshold are connected by an edge, producing a graph in which topical clusters, bridge communities, and linguistic outliers become visible. An interactive Streamlit app lets users adjust every step of this pipeline (filtering, featurization, dimensionality reduction, thresholds) and explore the resulting graphs, complemented by automated network analysis (community detection, betweenness, isolation).

## Dataset

We use the [Webis TLDR-17 corpus](https://huggingface.co/datasets/webis/tldr-17), a collection of millions of Reddit posts, each paired with an author-written TL;DR. It was originally built for summarization, but as the summaries are irrelevant to us, we discard them and all other fields but `body` and `subreddit`.

The corpus is heavily skewed. A handful of large subreddits contribute most of the posts, and a long tail contributes very little. Estimating a stable vocabulary for a subreddit with fifty posts is hopeless, so we restrict ourselves to the **top 300 subreddits by post count**. This keeps the communities with enough text to have a recognisable "voice".

## Methods

### Setup
We used a python 3.14.5 virtual environment to run our code on a laptop running Fedora Linux 44 (Kernel 7.0.11) with an AMD Ryzen 5500U and 16GB RAM.

Creating and entering the virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

All python dependencies are listed in the `requirements.txt` and can be installed using 
```bash
pip install -r requirements.txt
``` 

The pipeline splits into two stages. Running
```bash
python mine_tokens.py
```
does the expensive one-time work of calculating the subreddit specific word frequencies and saves the result to a file. 
For the reader's convenience, we precomputed this file and added it to the git repository, as the computation can be lengthy.


```bash
streamlit run netvis.py
```
opens a Streamlit app that loads the word frequency file and builds the graph live, in which the pre filtering and similarity matrix options can be adjusted.

### Vocabulary Building

For each of the top 300 subreddits we tokenize every post body with NLTK's `TweetTokenizer`, chosen over a plain word tokenizer because Reddit text is closer to tweets than to prose: casual, full of contractions, elongated words ("soooo"), and mentions. We lowercase everything, collapse repeated characters, strip handles, and keep only alphabetic tokens (allowing apostrophes, so "don't" survives).

This gives us the term frequencies for each subreddit, i.e. its raw vocabulary representation. We drop very rare words with a tiny relevance cutoff to get rid of outliers and typos.

### Graph Generation Pipeline 

To allow for flexible data exploration, we created a graphical interface to easily adjust important parameters. The pipeline can be split up into five parts:

**1. Filtering:** 
Some words are useless for telling communities apart. Function words like "the" and "and" appear everywhere, so we optionally remove standard English words using NLTK's word list using the option *Remove Standard English Words*.
We also drop words that appear in more than a set percentage of all subreddits (e.g. 80%). Increasing this threshold keeps more common words, while decreasing the threshold removes most shared words such that only highly specific words remain. Finally we keep only the subreddit's N (e.g. 100) most relevant words for further processing. This method gives us a word vector with words that are fairly common between some subreddits, but do not appear in most subreddits, which we can use to compare the language used in the subreddits.

**2. Representation:** We implemented three different featurization options:

- *Raw counts*; as the baseline, where the total number of occurrences of a word in each subreddit is encoded. This does not take into account the size of the subreddits text data.
- *Relevance*; this is the relative term frequency, where each count divided by the subreddit's total word count, so big subreddits don't dominate. We also added an additional filter to only keep words with a minimum relevance score that can be adjusted, such that extremely rare words that do not really represent the subreddits language get filtered out (primarily useful when keeping a lot of words per subreddit, to only keep the words with a high relevance).
- *TF-IDF*; down-weights words common across many subreddits and up-weights the distinctive ones.

**3. Dimensionality reduction:** Optionally we apply Latent Semantic Analysis to project the sparse term matrix down to at most 100 components, yielding higher similarity scores for semantically similar subreddits.

**4. Similarity calculation:** Whatever the representation, we compute the cosine similarity between every pair of subreddits, giving a similarity matrix.


**5. Graph construction and analysis:** To construct the graph, we draw an edge between two subreddits when their similarity clears a threshold. To eliminate clutter created by highly connective subreddits, we implemented an upper limit for node degrees. The result is rendered with Pyvis, so you can drag nodes around, hover a node to see its top words, and hover an edge to see the words two subreddits share. This allows for a highly interactive exploration of the resulting graph and facilitates an easy comparison between the pipelines options to find differences and similarities between the implemented possible options.
To allow analysis of graphs with many edges (in our case the top 8 outgoing edges of each node), we also implemented three graph analysis methods that calculate different matrices for the resulting graph:
- *Language Communities:* This method aims at finding clusters with high connectedness in the graph. Depending on the parameters, this usually returns clusters with shared topics. We used the Louvain method to calculate the communities.
- *Language Translators:* This method finds subreddits that act as bridges between subreddits, by calculating the betweenness of each subreddit. Betweenness is a measure for the ratio of shortest paths this node is a part of. These subreddits have similar vocabulary to many other subreddits which are not strongly connected to each other.
- *Isolated Subreddits:* These subreddits are the ones which have the lowest maximum similarity to all other subreddits.
  
## Results and Discussion

### Exploring the Graph
The interactive App can be used to explore different settings and the resulting graphs. Using the default settings we chose, the resulting graph looks like this:

<img src="figures/default_settings.png" width="100%" style="max-width: 700px;" alt="Default graph">

We can easily find many clusters that intuitively make sense, such as the tech cluster:

<img src="figures/tech_cluster.png" width="100%" style="max-width: 500px;" alt="Tech cluster">

or the politics cluster:

<img src="figures/politics_cluster.png" width="100%" style="max-width: 500px;" alt="Politics cluster">

or the religion cluster:

<img src="figures/religion_cluster.png" width="100%" style="max-width: 500px;" alt="Religion cluster">

Most connections in the graph make sense intuitively, and the edges can be inspected showing the top shared words between the subreddit. For example, ADHD and Drugs both seem to commonly talk about ADHD medication which seem to commonly be abused as drugs:

<img src="figures/drugs_adhd.png" width="100%" style="max-width: 500px;" alt="Drugs ADHD">

### Exploring parameter settings
We discovered that even slight changes of some parameters can cause vastly different looking graphs. For example changing the *similarity threshold* from the default 0.07 to slightly higher values while leaving the remaining parameters unchanged results in a way less connected graph and many isolated nodes, showing that most of the similarity scores are rather small, while some parameters such as keep top N words have a smaller impact on the resulting graph and usually only slightly change the similarity scores between subreddits.

The *max subreddit appearance* parameter also has a high impact on most of the graph, where a decrease in allowed subreddit appearance (= filtering out more of the common shared words) quickly leads to a highly disconnected graph when not adjusting the other parameters. Interestingly in this scenario, we can see that some highly specific clusters (like the League of Legends / Smite / DotA2 cluster) still stay connected, because they have a really large vocabulary of unique words (e.g. champion names, specific gaming terms like "ult", "gank", or "laning" which do not get filtered out even with the higher filtering settings), while more general connections get removed rather quickly.

| 25% Appearance | 60% Appearance |
| :--- | :--- |
| <img src="figures/appearance_25.png" width="100%" alt="Full graph at 25% appearance"> | <img src="figures/appearance_60.png" width="100%" alt="Full graph at 60% appearance"> |
| <img src="figures/lol_25.png" width="100%" alt="Cluster at 25% appearance"> | <img src="figures/lol_60.png" width="100%" alt="Cluster at 60% appearance"> |


We also investigated the difference between the different *featurization methods* (raw word count, relevance, TF-IDF) and concluded that while there are slight differences, the general appearance of the graph stays mostly the same. Also, not using latent semantic analysis to reduce the number of dimensions generally decreases similarity scores and connectedness of the graph, but this can be counteracted by decreasing the similarity threshold as well.

### Network analysis 

*Language Communities:* The communities detected via modularity match the ones we found in our visual analysis of the graph and share common topics such as Gaming (DestinyTheGame, Diablo, DnD, DotA2, Eve, Games, GlobalOffensive, Guildwars2) or Relationships (AskMen, AskReddit, AskWomen, OkCupid, TwoXChromosomes, relationship_advice, relationships, sex)

*Language Translators:* Our hypothesis was that bridges in the graph are also bridges in the vocabulary of the different subreddits. With default settings we get: r/funny, r/cars, r/books, r/SubredditDrama, r/Games, r/movies, r/CFB, r/unitedkingdom, r/IAmA, r/running as the top 10 highest betweenness (centrality) subreddits. This makes sense since these are mostly generic subreddits that cover broad topics such as funny things, cars or drama, meaning they have connections into different clusters and thereby function as "shortcuts" between them.

*Isolated Subreddits:* The top 10 most isolated subreddits are: r/asoiaf, r/SquaredCircle, r/pokemon, r/mylittlepony, r/starcraft, r/soccer, r/Eve, r/malefashionadvice, r/smashbros, r/masseffect. Most of these subreddits contain a lot of named entities specific to themselves such as character names. Since named entities likely end up in the featurization as a dimension, the subreddits featurization points into a vastly different direction than any other subreddit. Even tough this metric in included in the network analysis report, is not based on the analysis of the graph directly, but derived from the similarity matrix.

Interestingly, some subreddits, in this case r/Eve, can appear in both a Language community while at the same time being one of the most isolated subreddits. Because Louvain was run on a denser version of the graph (top 8 outgoing edges), r/Eve was clustered with other gaming subreddits, even though its overall absolute similarity scores were low enough to make it isolated under stricter threshold settings Depending on the parameters, some subreddits may also appear in both the language translators and the isolated subreddits for the same reason.


## Conclusion
Our method of filtering out a small number of relevant words per subreddit and computing similarities works pretty well and yields stable results. The Streamlit application is a powerful tool to understand and compare the different implemented methods and filters, and can produce a variety of different graphs. The tool is also a fun way to interactively explore connections between subreddits and find interesting and sometimes unexpected shared vocabulary. Overall, we conclude that any of the featurization methods can yield interesting and meaningful results if the parameters are tuned accordingly, but the graph is often very cluttered and highly connected or very sparse, making manual exploration difficult with many parameter settings. Also directly comparing large graphs visually is often difficult, especially for highly connected graphs, showing the importance of automated network analysis methods and metrics to gain additional interesting information.

## Contributions

| Team Member     | Contributions                                             |
|-----------------|-----------------------------------------------------------|
| Lukas Mauz      | network analysis methods, graph generation, report |
| Lucas Cimerman  | tokenization, presentation, matrix calculations, report |
| Roman Lohmiller | pre-filterig, similarity matrix calculations, graph generation, report |
