import nltk
from spellchecker import SpellChecker
from collections import Counter
from datasets import load_dataset
import pandas as pd
import math
import pickle

min_relevance = 0.00001 # minimum relative word frequency for the word to be included in the subreddits specific words

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


    spell = SpellChecker()

    id_to_name = df.drop_duplicates(subset=['subreddit_id']).set_index('subreddit_id')['subreddit'].to_dict()

    top_100_subs = df['subreddit_id'].value_counts().head(100).index.tolist()
    total_freq = Counter()
    sqrt_freq = Counter()

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

    with open("results.pickle", "wb") as f:
        pickle.dump(results, f)

def analyze():
    with open("results.pickle", "rb") as f:
        results = pickle.lead(f)
    
    sqrt_freq = Counter()
    for s_id, freq in results:
        if s_id == 'total':
            continue

        for word, relevance in freq.items():
            sqrt_freq[word] += math.sqrt(relevance)
    
    print(sqrt_freq.most_common())

analyze()