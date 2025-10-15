"""
Code generate by GPT-5
Detect meaningless tokens in a Qwen3 tokenizer.
Requirements:
    pip install transformers jieba tqdm pandas
Usage:
    python detect_meaningless_tokens.py corpus.txt output.csv
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import sys
import jieba
import pandas as pd
from tqdm import tqdm
from collections import Counter
from math import log2
from transformers import AutoTokenizer
import pickle

# ---------- Helper functions ----------
def entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts]
    return -sum(p * log2(p) for p in probs if p > 0)

# ---------- Main ----------
def main(corpus_path, output_path):
    print("Loading Qwen3 tokenizer ...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    vocab = tok.get_vocab()
    tokens = list(vocab.keys())

    print(f"Loaded {len(tokens)} tokens.")
    print("Reading corpus ...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    print("Computing token frequencies ...")
    if(os.path.exists("cache/token_freq.pkl") and os.path.exists("cache/standalone_freq.pkl")):
        with open('cache/token_freq.pkl', 'rb') as f:
            token_freq:Counter = pickle.load(f)
        with open('cache/standalone_freq.pkl', 'rb') as f:
            standalone_freq:Counter = pickle.load(f)

    else:
        token_freq = Counter()
        standalone_freq = Counter()

        # Chinese char range for regex
        zh_char = r"[\u4e00-\u9fa5]"

        for t in tqdm(tokens):
            token_freq[t] = corpus.count(t)
            standalone_freq[t] = len(
                re.findall(rf"(?<!{zh_char}){re.escape(t)}(?!{zh_char})", corpus)
            )

        with open('cache/token_freq.pkl', 'wb') as f:
            pickle.dump(data, f)
        with open('cache/standalone_freq.pkl', 'wb') as f:
            pickle.dump(data, f)
        

    print("Computing neighbor entropy ...")
    neighbor_ent = {}
    for t in tqdm(tokens):
        matches = list(re.finditer(t, corpus))
        left, right = Counter(), Counter()
        for m in matches:
            s, e = m.span()
            if s > 0 and re.match(zh_char, corpus[s - 1]):
                left[corpus[s - 1]] += 1
            if e < len(corpus) and re.match(zh_char, corpus[e]):
                right[corpus[e]] += 1
        neighbor_ent[t] = entropy(list(left.values()) + list(right.values()))

    print("Checking dictionary ...")
    dict_words = set(jieba.dt.FREQ.keys())

    data = []
    for t in tokens:
        f = token_freq[t]
        if f < 10 or len(t) <= 1:
            continue
        ratio = standalone_freq[t] / (f + 1e-6)
        in_dict = t in dict_words
        ent = neighbor_ent[t]
        data.append(
            {
                "token": t,
                "freq": f,
                "standalone_ratio": ratio,
                "entropy": ent,
                "in_dict": in_dict,
            }
        )

    df = pd.DataFrame(data)
    df.sort_values("freq", ascending=False, inplace=True)

    # Flag meaningless tokens
    df["meaningless"] = (
        (df["standalone_ratio"] < 0.05)
        & (~df["in_dict"])
        & (df["entropy"] < 1.0)
    )

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Done. Results saved to {output_path}")
    print(f"Meaningless tokens found: {df['meaningless'].sum()} / {len(df)}")

# ---------- Run ----------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python detect_meaningless_tokens.py corpus.txt output.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
