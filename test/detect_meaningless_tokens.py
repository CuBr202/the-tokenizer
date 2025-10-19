"""
Fast Qwen tokenizer meaningless token detector
Optimized for 100GB corpora with multiprocessing + streaming
Author: Me, GPT-5 & Gemini-2.5 (fixed multiprocessing pickle issue)
"""

import os, sys, re, multiprocessing as mp
from collections import Counter, defaultdict
from math import log2
import pandas as pd
from tqdm import tqdm
import ahocorasick
from transformers import AutoTokenizer

CHUNK_SIZE = 32 * 1024 * 1024  # 32MB per chunk

# ---------- Helpers ----------

def entropy(counts):
    total = sum(counts)
    if total == 0: 
        return 0.0
    probs = [c / total for c in counts]
    return -sum(p * log2(p) for p in probs if p > 0)

def bytes_to_latin1_str(b): 
    return b.decode('latin-1')

def latin1_str_to_bytes(s): 
    return s.encode('latin-1')

def neighbor_counter_factory():
    """Top-level factory for defaultdict (multiprocessing-safe)."""
    return {'left': Counter(), 'right': Counter()}

# ---------- Worker ----------

def process_chunk(args):
    """Process a chunk of the corpus and count token stats."""
    chunk_bytes, token_bytes_list = args
    corpus_str = bytes_to_latin1_str(chunk_bytes)

    # Build automaton per process (fast for ~few MB)
    A = ahocorasick.Automaton()
    for tb in token_bytes_list:
        A.add_word(bytes_to_latin1_str(tb), tb)
    A.make_automaton()

    token_freq = Counter()
    standalone_freq = Counter()
    neighbor_counters = defaultdict(neighbor_counter_factory)

    corpus_len = len(corpus_str)
    for end_index, tb in A.iter(corpus_str):
        token_freq[tb] += 1
        start_index = end_index - len(bytes_to_latin1_str(tb)) + 1

        # Neighbors
        if start_index > 0:
            left_b = latin1_str_to_bytes(corpus_str[start_index - 1])[0]
            neighbor_counters[tb]['left'][left_b] += 1
        if end_index + 1 < corpus_len:
            right_b = latin1_str_to_bytes(corpus_str[end_index + 1])[0]
            neighbor_counters[tb]['right'][right_b] += 1

        # Standalone check
        if start_index > 0 and end_index + 1 < corpus_len:
            lvis = 32 <= latin1_str_to_bytes(corpus_str[start_index - 1])[0] <= 126
            rvis = 32 <= latin1_str_to_bytes(corpus_str[end_index + 1])[0] <= 126
            if not lvis and not rvis:
                standalone_freq[tb] += 1

    return token_freq, standalone_freq, neighbor_counters


# ---------- Main ----------

def main(corpus_path, output_path, n_workers=mp.cpu_count()):

    print(f"Loading Qwen tokenizer...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    vocab = tok.get_vocab()
    tokens = list(vocab.keys())
    print(f"Loaded {len(tokens)} tokens.")

    # Convert tokens → bytes once
    token_bytes_list = []
    token_bytes_map = {}
    for t in tokens:
        try:
            tb = tok.convert_tokens_to_string([t]).encode('utf-8', errors='replace')
        except Exception:
            tb = t.encode('utf-8', errors='replace')
        token_bytes_list.append(tb)
        token_bytes_map[tb] = t

    print(f"Starting parallel corpus scan with {n_workers} workers...")
    pool = mp.Pool(processes=n_workers)
    tasks = []

    # Chunked reading
    with open(corpus_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            tasks.append((chunk, token_bytes_list))

    results = list(tqdm(pool.imap_unordered(process_chunk, tasks), total=len(tasks)))
    pool.close()
    pool.join()

    # Merge results
    print("Aggregating results...")
    token_freq = Counter()
    standalone_freq = Counter()
    neighbor_counters = defaultdict(neighbor_counter_factory)

    for tf, sf, nc in results:
        token_freq.update(tf)
        standalone_freq.update(sf)
        for k, v in nc.items():
            neighbor_counters[k]['left'].update(v['left'])
            neighbor_counters[k]['right'].update(v['right'])

    # Compute metrics
    print("Computing final metrics...")
    byte_pattern = re.compile(rb"[\x00-\x08\x0B\x0C\x0E-\x1F]")
    data = []
    for tb in tqdm(token_bytes_list):
        freq = token_freq.get(tb, 0)
        ratio = standalone_freq.get(tb, 0) / (freq + 1e-6)
        if tb in neighbor_counters:
            ent = entropy(list(neighbor_counters[tb]['left'].values()) +
                          list(neighbor_counters[tb]['right'].values()))
        else:
            ent = 0.0
        tok_str = tb.decode('utf-8', errors='replace')
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

    print(f"✅ Done. Saved to {output_path}")
    print(f"Meaningless tokens: {df['meaningless'].sum()} / {len(df)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python detect_meaningless_tokens_fast.py corpus.txt output.csv [n_workers]")
        sys.exit(1)
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else mp.cpu_count()
    main(sys.argv[1], sys.argv[2], n_workers)
