# A Map of Reddit, Drawn from Words
_Group members: [Lucas Cimerman], [Lukas Mauz], [Roman Lohmiller]_

Reddit is a platform composed of diverse, distinct communities, allowing users to join and participate in multiple groups simultaneously. To analyze the composition of these so-called subreddits, we wanted to draw a graphical map based on the similarity in language used on them.

Our idea is simple. r/leagueoflegends and r/DotA2 should sit close together, because they talk about the same things: lanes, cooldowns, patches. r/personalfinance and r/wallstreetbets share a vocabulary too, even if the tone is different. If that intuition holds, then the words alone should be enough to reconstruct the structure of Reddit based solely language.

## Introduction

The base idea is simple; we encode the language used in each subreddit as a vector of numbers and compare them using cosine similarity. 

## Dataset

We use the [Webis TLDR-17 corpus](https://huggingface.co/datasets/webis/tldr-17), a collection of roughly 3.8 million Reddit posts, each paired with an author-written TL;DR. It was originally built for summarization, but as the summaries are irrelevant to us, we discard them and all other fields but `body` and `subreddit`.

The corpus is heavily skewed. A handful of large subreddits contribute most of the posts, and a long tail contributes very little. Estimating a stable vocabulary for a subreddit with fifty posts is hopeless, so we restrict ourselves to the **top 300 subreddits by post count**. This keeps the communities with enough text to have a recognisable "voice".

## Methods

### Setup

MAYBE TALK ABOUT DEPENDENCIES

The pipeline splits into two stages. `tokenize.py` does the expensive one-time work of counting words per subreddit and saves the result to a pickle. `netvis.py` is a Streamlit app that loads that pickle and builds the graph live, so every filter and parameter can be changed without recomputing anything.

### Vocabulary Building

For each of the top 300 subreddits we tokenize every post body with NLTK's `TweetTokenizer` — chosen over a plain word tokenizer because Reddit text is closer to tweets than to prose: casual, full of contractions, elongated words ("soooo"), and mentions. We lowercase everything, collapse repeated characters, strip handles, and keep only alphabetic tokens (allowing apostrophes, so "don't" survives).

This gives us a word-frequency counter for each subreddit, its raw vocabulary representation. We drop the very rarest words with a tiny relevance cutoff, then store each subreddit as `{counter, total_word_count}`.

### Graph Generation Pipeline 

To allow for flexible data exploration, we created a graphical interface to easily adjust important parameters.

**Filtering.** 
Some words are useless for telling communities apart. Function words like "the" and "and" appear everywhere, so we optionally remove standard English words using NLTK's word list using the option *Remove Standard English Words*.
 We also drop words that appear in more than a set percentage of all subreddits (default 80%). Finally we keep only the top N words per subreddit to define its "topic" and remove noise.

**Representation.** We implemented three different featurization options:

- *Raw counts*; as the baseline.
- *Relevance*; each count divided by the subreddit's total word count, so big subreddits don't dominate.
- *TF-IDF*; down-weights words common across many subreddits and up-weights the distinctive ones.

**Dimensionality reduction.** Optionally we apply Latent Semantic Analysis to project the sparse term matrix down to at most 100 components, smoothing out noise and grouping words that co-occur.

**Similarity.** Whatever the representation, we compute the cosine similarity between every pair of subreddits, giving a 300×300 similarity matrix.

<!-- FIGURE 4: A heatmap of the similarity matrix, with rows/columns ordered by cluster (e.g. via hierarchical clustering). If the method works, you should see bright blocks on the diagonal — this is strong visual evidence and worth including. Consider showing it for TF-IDF vs raw counts side by side. -->

**Graph construction.** To construct the graph, we draw an edge between two subreddits when their similarity clears a threshold. To eliminate clutter created by highly connective subreddits, we implemented an upper limit for node degrees.The result is rendered with Pyvis, so you can drag nodes around, hover a node to see its top words, and hover an edge to see the words two subreddits share.

## Results and Discussion

In this section we discuss 3 example setups, one for each of the word representations. We do this to highlight the strenghts of each of the representations and encourage the reader to try out other setups.

### Baseline

Allow all words
reuslts in fully connected graph


### Use MaxSubredditAp

Big improvement
MAx 90
Works better, see clusters

Max 20
Less clusters, barely connected 
Get highly specific lingo



### Keep top N words

Increasing Feature Size allows removal of MaxSub if increase of Similarity Threshhold

### Similarity Threshhold 

Hyperparameter

### Relevance Changes somethin

### DF IDF

Works great without filters


