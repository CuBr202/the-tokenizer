"""
Code written by GPT-5 and optimized by Gemini 2.5
Optimized script to detect meaningless tokens in a Qwen3 tokenizer.
This version uses the Aho-Corasick algorithm for a single-pass analysis,
dramatically reducing runtime from hours to minutes.

Requirements:
    pip install transformers jieba tqdm pandas pickle pyahocorasick

Usage:
    python detect_meaningless_tokens.py corpus.txt output.csv
"""
import os
import sys
import re
import copy
import pickle
from collections import Counter, defaultdict
from math import log2

import pandas as pd
import jieba
import ahocorasick
from tqdm import tqdm
from transformers import AutoTokenizer

# Set HF endpoint if needed, e.g., for users in China
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---------- Helper Functions ----------

def entropy(counts):
    """Calculates the entropy of a list of counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts]
    return -sum(p * log2(p) for p in probs if p > 0)

def process_corpus_single_pass(corpus: str, tokens: list):
    """
    Processes the corpus in a single pass to gather all required statistics.
    
    Args:
        corpus (str): The text corpus to analyze.
        tokens (list): A list of token strings from the tokenizer's vocabulary.

    Returns:
        tuple: A tuple containing token_freq, standalone_freq, and neighbor_counters.
    """
    print("Building Aho-Corasick automaton...")
    A = ahocorasick.Automaton()
    for idx, token in enumerate(tokens):
        # Store the token string itself as the value
        A.add_word(token, token)
    A.make_automaton()
    print("Automaton built.")

    token_freq = Counter()
    standalone_freq = Counter()
    # Use defaultdict for convenience
    neighbor_counters = defaultdict(lambda: {'left': Counter(), 'right': Counter()})

    # Pre-compile regex for efficiency
    zh_char_pattern = re.compile(r'[\u4e00-\u9fa5]')

    print("Processing corpus in a single pass...")
    # Use tqdm to show progress over the corpus processing
    for end_index, token in tqdm(A.iter(corpus), total=len(corpus), unit="bytes", unit_scale=True):
        token_freq[token] += 1
        
        start_index = end_index - len(token) + 1
        
        # --- Check for standalone occurrences ---
        # Check character before the token
        is_left_zh = False
        if start_index > 0:
            if zh_char_pattern.match(corpus[start_index - 1]):
                is_left_zh = True
        
        # Check character after the token
        is_right_zh = False
        if end_index + 1 < len(corpus):
            if zh_char_pattern.match(corpus[end_index + 1]):
                is_right_zh = True

        if not is_left_zh and not is_right_zh:
            standalone_freq[token] += 1
            
        # --- Collect neighbors for entropy calculation ---
        if start_index > 0:
            left_char = corpus[start_index - 1]
            if zh_char_pattern.match(left_char):
                neighbor_counters[token]['left'][left_char] += 1
                
        if end_index + 1 < len(corpus):
            right_char = corpus[end_index + 1]
            if zh_char_pattern.match(right_char):
                neighbor_counters[token]['right'][right_char] += 1

    return token_freq, standalone_freq, neighbor_counters

# ---------- Main ----------

def main(corpus_path, output_path):
    global token_freq_pickle, standalone_freq_pickle, neighbor_counters_pickle

    cache_dir = "cache"
    stats_cache_path = os.path.join(cache_dir, "corpus_stats.pkl")
    
    os.makedirs(cache_dir, exist_ok=True)

    print("Loading Qwen3 tokenizer...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    vocab = tok.get_vocab()
    tokens = list(vocab.keys())
    print(f"Loaded {len(tokens)} tokens.")

    if os.path.exists(stats_cache_path):
        print(f"Loading cached stats from {stats_cache_path}...")
        with open(stats_cache_path, 'rb') as f:
            stats = pickle.load(f)
        token_freq = stats['token_freq']
        standalone_freq = stats['standalone_freq']
        neighbor_counters = stats['neighbor_counters']
    else:
        print("Reading corpus...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = f.read()
        
        token_freq, standalone_freq, neighbor_counters = process_corpus_single_pass(corpus, tokens)

        """token_freq_pickle, standalone_freq_pickle, neighbor_counters_pickle = copy.deepcopy(token_freq), copy.deepcopy(standalone_freq), copy.deepcopy(neighbor_counters)
        
        print(f"Caching computed stats to {stats_cache_path}...")
        with open(stats_cache_path, 'wb') as f:
            pickle.dump({
                'token_freq': token_freq_pickle,
                'standalone_freq': standalone_freq_pickle,
                'neighbor_counters': neighbor_counters_pickle
            }, f)
        del token_freq_pickle, standalone_freq_pickle, neighbor_counters_pickle"""

    print("Computing neighbor entropy...")
    neighbor_ent = {}
    for t in tqdm(tokens):
        # It's much faster to compute entropy now from pre-aggregated counters
        if t in neighbor_counters:
            left_counts = neighbor_counters[t]['left'].values()
            right_counts = neighbor_counters[t]['right'].values()
            neighbor_ent[t] = entropy(list(left_counts) + list(right_counts))
        else:
            neighbor_ent[t] = 0.0

    #print("Checking against jieba dictionary...")
    #dict_words = set(jieba.dt.FREQ.keys())

    print("Aggregating data...")
    data = []
    for t in tqdm(tokens):
        f = token_freq.get(t, 0)
        # Filter out low-frequency or short tokens early
        if f < 10 or len(t) <= 1:
            continue
        
        ratio = standalone_freq.get(t, 0) / (f + 1e-6)
        #in_dict = t in dict_words
        in_dict = True
        ent = neighbor_ent.get(t, 0.0)
        
        data.append({
            "token": t,
            "freq": f,
            "standalone_ratio": ratio,
            "entropy": ent,
            "in_dict": in_dict,
        })

    print("Building and saving DataFrame...")
    df = pd.DataFrame(data)
    df.sort_values("freq", ascending=False, inplace=True)

    # Flag meaningless tokens based on the defined criteria
    df["meaningless"] = (
        (df["standalone_ratio"] < 0.05) & 
        (~df["in_dict"]) & 
        (df["entropy"] < 1.0)
    )

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Done. Results saved to {output_path}")
    print(f"Meaningless tokens found: {df['meaningless'].sum()} / {len(df)}")


# ---------- Run ----------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python detect_meaningless_tokens_optimized.py corpus.txt output.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])