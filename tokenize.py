import nltk
from collections import Counter
from datasets import load_dataset
import pickle

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
    print(f'Loaded Dataset, total {len(ds)} entries')
    print(f'Tokeinzing data with min_relevance={min_relevance} and top_n_subs={top_n_subs}')

    top_subs = df['subreddit'].value_counts().head(top_n_subs).index.tolist()

    results = {}
    print(top_subs)
    for idx, subreddit in enumerate(top_subs):
        # find common words in the subreddit
        filtered = df[df['subreddit'] == subreddit]
        full_freq = clean_and_count(filtered)

        # apply cutoff to only consider the most common words
        subreddit_words_with_relevance = {word: count for word, count in full_freq.items() if count / full_freq.N() >= min_relevance}
        final_flagged = Counter(subreddit_words_with_relevance)
        results[subreddit] = {'size_idx': idx, 'counter': final_flagged, 'full_freq_n': full_freq.N()}

        print(f'Parsed {subreddit} ({len(final_flagged)} unique words)')

    with open(filename, "wb") as f:
        pickle.dump(results, f)

mine_words("results_0.0000001_300_freq.pickle", min_relevance=0.00000001, top_n_subs=300)