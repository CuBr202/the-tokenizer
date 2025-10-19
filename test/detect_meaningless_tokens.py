"""
Qwen3 tokenizer meaningless token detector
Target: Detect meaningless tokens from the tokenizer's vocabulary,
not from the corpus vocabulary.

Byte-safe + corpus-aware + full-vocab analysis
Author: GPT-5
"""

import os, sys, re
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

def bytes_to_latin1_str(b: bytes) -> str:
    return b.decode('latin-1')

def latin1_str_to_bytes(s: str) -> bytes:
    return s.encode('latin-1')

# ---------- Processing ----------

def process_corpus_bytelevel(corpus_bytes: bytes, token_bytes_list):
    """
    Count occurrences and neighbor entropy for all tokenizer tokens.
    Corpus is treated as byte stream; tokens are byte sequences.
    """
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

        # Left / right neighbors
        if start_index > 0:
            left_b = latin1_str_to_bytes(corpus_str[start_index - 1])[0]
            neighbor_counters[token_bytes]['left'][left_b] += 1
        if end_index + 1 < corpus_len:
            right_b = latin1_str_to_bytes(corpus_str[end_index + 1])[0]
            neighbor_counters[token_bytes]['right'][right_b] += 1

        # Standalone check
        if start_index > 0 and end_index + 1 < corpus_len:
            left_visible = 32 <= latin1_str_to_bytes(corpus_str[start_index - 1])[0] <= 126
            right_visible = 32 <= latin1_str_to_bytes(corpus_str[end_index + 1])[0] <= 126
            if not left_visible and not right_visible:
                standalone_freq[token_bytes] += 1

    return token_freq, standalone_freq, neighbor_counters


# ---------- Main ----------

def main(corpus_path, output_path):

    print("Loading Qwen3 tokenizer...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    vocab = tok.get_vocab()
    tokens = list(vocab.keys())
    print(f"Loaded {len(tokens)} tokens from tokenizer vocabulary.")

    # Convert tokens → bytes
    token_bytes_list = []
    token_bytes_map = {}
    for t in tokens:
        try:
            token_bytes = tok.convert_tokens_to_string([t]).encode('utf-8', errors='replace')
        except Exception:
            token_bytes = t.encode('utf-8', errors='replace')
        token_bytes_list.append(token_bytes)
        token_bytes_map[token_bytes] = t

    # Read corpus bytes
    print("Reading corpus as bytes...")
    with open(corpus_path, "rb") as f:
        corpus_bytes = f.read()

    token_freq, standalone_freq, neighbor_counters = process_corpus_bytelevel(
        corpus_bytes, token_bytes_list
    )

    # ---------- Aggregate results ----------
    print("Computing neighbor entropy and aggregating results...")
    data = []
    byte_pattern = re.compile(rb"[\x00-\x08\x0B\x0C\x0E-\x1F]")  # control bytes

    for tb in tqdm(token_bytes_list):
        freq = token_freq.get(tb, 0)
        ratio = standalone_freq.get(tb, 0) / (freq + 1e-6)
        ent = 0.0
        if tb in neighbor_counters:
            left_counts = neighbor_counters[tb]['left'].values()
            right_counts = neighbor_counters[tb]['right'].values()
            ent = entropy(list(left_counts) + list(right_counts))

        tok_str = tb.decode('utf-8', errors='replace')

        # Heuristics for meaningless tokens:
        #  - never appears (freq == 0)
        #  - mostly control bytes
        #  - low entropy and no standalone usage
        meaningless = (
            (freq == 0)
            or byte_pattern.search(tb)
            or (ent < 0.5 and ratio < 0.05 and freq < 10)
        )

        data.append({
            "token": tok_str,
            "freq_in_corpus": freq,
            "standalone_ratio": ratio,
            "entropy": ent,
            "length_bytes": len(tb),
            "meaningless": meaningless
        })

    df = pd.DataFrame(data)
    df.sort_values("freq_in_corpus", ascending=False, inplace=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Done. Results saved to {output_path}")
    print(f"Meaningless tokens found: {df['meaningless'].sum()} / {len(df)}")


# ---------- Run ----------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python detect_meaningless_tokens.py corpus.txt output.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
