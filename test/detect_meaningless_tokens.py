"""
Optimized & byte-safe version for Qwen3 tokenizer
Author: GPT-5 (improved with byte-level support)
"""

import os, sys, re, pickle
from collections import Counter, defaultdict
from math import log2
import pandas as pd
from tqdm import tqdm
import ahocorasick
from transformers import AutoTokenizer

# ---------- Helper ----------

def entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts]
    return -sum(p * log2(p) for p in probs if p > 0)

# ---------- Byte Utilities ----------

def bytes_to_latin1_str(b: bytes) -> str:
    """Map bytes [0-255] → Unicode U+0000–U+00FF using Latin-1."""
    return b.decode('latin-1')

def latin1_str_to_bytes(s: str) -> bytes:
    """Inverse mapping."""
    return s.encode('latin-1')

# ---------- Processing ----------

def process_corpus_bytelevel(corpus_bytes: bytes, token_bytes_list):
    print("Building Aho-Corasick automaton (byte-level via Latin-1)...")
    A = ahocorasick.Automaton()
    for token_bytes in token_bytes_list:
        token_str = bytes_to_latin1_str(token_bytes)
        A.add_word(token_str, token_bytes)
    A.make_automaton()
    print("Automaton built.")

    token_freq = Counter()
    standalone_freq = Counter()
    neighbor_counters = defaultdict(lambda: {'left': Counter(), 'right': Counter()})

    corpus_str = bytes_to_latin1_str(corpus_bytes)
    corpus_len = len(corpus_str)

    print("Scanning corpus (byte-level, single pass)...")
    for end_index, token_bytes in tqdm(A.iter(corpus_str), total=corpus_len, unit="char", unit_scale=True):
        token_freq[token_bytes] += 1
        start_index = end_index - len(bytes_to_latin1_str(token_bytes)) + 1

        # Left / right neighbors in bytes
        if start_index > 0:
            left_b = latin1_str_to_bytes(corpus_str[start_index - 1])[0]
            neighbor_counters[token_bytes]['left'][left_b] += 1
        if end_index + 1 < corpus_len:
            right_b = latin1_str_to_bytes(corpus_str[end_index + 1])[0]
            neighbor_counters[token_bytes]['right'][right_b] += 1

        # Standalone = not surrounded by visible printable bytes
        if start_index > 0 and end_index + 1 < corpus_len:
            left_visible = 32 <= latin1_str_to_bytes(corpus_str[start_index - 1])[0] <= 126
            right_visible = 32 <= latin1_str_to_bytes(corpus_str[end_index + 1])[0] <= 126
            if not left_visible and not right_visible:
                standalone_freq[token_bytes] += 1

    return token_freq, standalone_freq, neighbor_counters

# ---------- Main ----------

def main(corpus_path, output_path):
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "corpus_stats.pkl")

    print("Loading Qwen3 tokenizer...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    vocab = tok.get_vocab()
    tokens = list(vocab.keys())
    print(f"Loaded {len(tokens)} tokens.")

    # Convert tokens → bytes (as tokenizer would encode)
    token_bytes_list = []
    for t in tokens:
        # Convert token (which may include byte fallback) to bytes
        try:
            token_bytes = tok.convert_tokens_to_string([t]).encode('utf-8', errors='replace')
        except Exception:
            token_bytes = t.encode('utf-8', errors='replace')
        token_bytes_list.append(token_bytes)

    if os.path.exists(cache_file):
        print(f"Loading cached stats from {cache_file}...")
        with open(cache_file, 'rb') as f:
            stats = pickle.load(f)
        token_freq = stats['token_freq']
        standalone_freq = stats['standalone_freq']
        neighbor_counters = stats['neighbor_counters']
    else:
        print("Reading corpus as bytes...")
        with open(corpus_path, "rb") as f:
            corpus_bytes = f.read()

        token_freq, standalone_freq, neighbor_counters = process_corpus_bytelevel(
            corpus_bytes, token_bytes_list
        )

        """print(f"Caching computed stats to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'token_freq': token_freq,
                'standalone_freq': standalone_freq,
                'neighbor_counters': neighbor_counters
            }, f)"""

    print("Computing neighbor entropy...")
    neighbor_ent = {}
    for tb in tqdm(token_bytes_list):
        if tb in neighbor_counters:
            left_counts = neighbor_counters[tb]['left'].values()
            right_counts = neighbor_counters[tb]['right'].values()
            neighbor_ent[tb] = entropy(list(left_counts) + list(right_counts))
        else:
            neighbor_ent[tb] = 0.0

    print("Aggregating data...")
    data = []
    for tb in tqdm(token_bytes_list):
        f = token_freq.get(tb, 0)
        if f < 10 or len(tb) <= 1:
            continue
        ratio = standalone_freq.get(tb, 0) / (f + 1e-6)
        ent = neighbor_ent.get(tb, 0.0)
        tok_str = tb.decode('utf-8', errors='replace')
        data.append({
            "token": tok_str,
            "freq": f,
            "standalone_ratio": ratio,
            "entropy": ent
        })

    df = pd.DataFrame(data)
    df.sort_values("freq", ascending=False, inplace=True)

    df["meaningless"] = (
        (df["standalone_ratio"] < 0.05) &
        (df["entropy"] < 1.0)
    )

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Done. Results saved to {output_path}")
    print(f"Meaningless tokens found: {df['meaningless'].sum()} / {len(df)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python detect_meaningless_tokens_bytelevel.py corpus.txt output.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
