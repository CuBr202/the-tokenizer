"""
Fast Qwen tokenizer meaningless token detector
Optimized for 100GB corpora with multiprocessing + streaming
Author: Me, GPT-5 & Gemini-2.5 (fixed multiprocessing pickle issue)
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys, re, multiprocessing as mp
import argparse
from collections import Counter, defaultdict
from math import log2
import pandas as pd
from tqdm import tqdm
import ahocorasick
from transformers import AutoTokenizer
from datasets import load_dataset, ReadInstruction
from itertools import islice



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

def batch_iter(iterable, batch_size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, int(batch_size)))
        if not batch:
            break
        yield batch

def assemble_batch(batch):
    """Encode a batch of samples into bytes (fast in parallel)."""
    joined_text = "\n".join(s["text"] for s in batch)
    return joined_text.encode("utf-8", errors="ignore")

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

        # Standalone check (only for english-style languages)
        if start_index > 0 and end_index + 1 < corpus_len:
            lvis = 32 <= latin1_str_to_bytes(corpus_str[start_index - 1])[0] <= 126
            rvis = 32 <= latin1_str_to_bytes(corpus_str[end_index + 1])[0] <= 126
            if not lvis and not rvis:
                standalone_freq[tb] += 1

    return token_freq, standalone_freq, neighbor_counters


# ---------- Main ----------

def main(dataset_mode, corpus_path, output_path, n_workers=mp.cpu_count(), CHUNK_SIZE = 32 * 1024 * 1024):
    BATCH_SIZE = CHUNK_SIZE/ (8*1024)

    print(f"Loading Qwen tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    vocab = tok.get_vocab()
    tokens = list(vocab.keys())
    print(f"Loaded {len(tokens)} tokens.", flush=True)

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

    print(f"Starting parallel corpus scan with {n_workers} workers...", flush=True)
    pool = mp.Pool(processes=n_workers)
    tasks = []

    # Chunked reading
    if dataset_mode:
        # Hf dataset
        #"Geralt-Targaryen/C4-zh"
        print(f"Loading Huggingface Dataset: {corpus_path}", flush=True)
        dataset = load_dataset(corpus_path, split="train", streaming=True)

        # smaller batches keep memory low and parallelism high 
        buffer = bytearray()
        tasks = []

        # parallel encode batches -> bytes
        with mp.Pool(processes=n_workers) as encode_pool:  # Unlimited!
            for encoded_batch in tqdm(
                encode_pool.imap_unordered(assemble_batch, batch_iter(dataset, BATCH_SIZE)),
                desc="Encoding dataset batches"
            ):
                buffer.extend(encoded_batch)
                if len(buffer) >= CHUNK_SIZE:
                    tasks.append((bytes(buffer), token_bytes_list))
                    buffer.clear()

            if buffer:
                tasks.append((bytes(buffer), token_bytes_list))
    else:
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
    print("Aggregating results...", flush=True)
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
    print("Computing final metrics...", flush=True)
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

    print(f"✅ Done. Saved to {output_path}", flush=True)
    print(f"Meaningless tokens: {df['meaningless'].sum()} / {len(df)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a program aiming to find meaningless tokens in Qwen-2.5 tokenizer.")

    parser.add_argument("-d", "--dataset", action="store_true", help="Use hf dataset instead of local files.")  #action="store_true"
    parser.add_argument("-p", "--path", type=str, default="corpus.txt", help="Path to the dataset. If using hf dataset, this should be the dataset name.")
    parser.add_argument("-o", "--output", type=str, default="output.csv", help="Output path of the result.")
    parser.add_argument("-n", "--num_cpus", type=int, default=0, help="Numbers of cpus that's utilized. If this arg is not passed, the program will use all CPUs in the system.")
    parser.add_argument("-c", "--chunk_size", type=int, default=32*1024*1024, help="Chunks of memory allocations in the process.")

    # 解析参数
    args = parser.parse_args()

    n_workers = args.num_cpus if args.num_cpus != 0 else mp.cpu_count()
    main(args.dataset, args.path, args.output, n_workers, args.chunk_size)
