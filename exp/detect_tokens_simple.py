"""
Counts token frequencies in a large corpus NOT using manual multiprocessing.

Supports two modes:
1.  --dataset_mode: Stream a dataset from Hugging Face Hub (e.g., "Geralt-Targaryen/C4-zh").
2.  (Default): Read a large local text file (e.g., "my_corpus.txt").

The script processes the corpus in chunks, counts token frequencies in parallel,
and aggregates the results into a single JSON file.

Example Usage:

# 1. From a local file (e.g., a 1GB text file)
python detect_tokens_simple.py -p ../test/corpus.txt -c 4096 Qwen/Qwen2.5-7B
python detect_tokens_simple.py -p ../test/corpus.txt -c 4096 gpt5

# 2. From a Hugging Face streaming dataset
python3 detect_tokens_2.py "Geralt-Targaryen/C4-zh" \
    --dataset_mode \
    --output_file c4_zh_counts.json \
    --text_column "text" \
    --hf_batch_size 1024

"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import sys
from collections import Counter
from typing import Iterator, Dict, Any
import tiktoken
import time

from transformers import AutoTokenizer
from datasets import Dataset, load_dataset, ReadInstruction

from gpt_tokenizer import JSGPTTokenizer
try:
    from .utils import get_total_lines
except:
    from utils import get_total_lines

# --- Worker Function ---
class Worker:
    def __init__(self, tokenizer_name: str):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        self.tokenize_times = 0
        
    def _init_tokenizer(self):
        if "gpt" in self.tokenizer_name:
            """Loads JavaScript module for gpt-style tokenizer"""
            try:
                self.tokenizer = JSGPTTokenizer(js_filename="gpt_tokenizer.mjs",use_mp=True)
            except Exception as e:
                print(f"Frror occurred when loading JavaScript GPT-tokenizer module:{e}", file=sys.stderr)
                print("Falling back to tiktoken (gpt-4o).", file=sys.stderr)
                try:
                    encoding = tiktoken.encoding_for_model("gpt-4o")
                except KeyError:
                    print("Warning: model not found. Using cl100k_base encoding.")
                    encoding = tiktoken.get_encoding("cl100k_base")
                self.tokenizer = encoding
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=True, use_fast=True)
        
    def tokenize_and_count(self, text_chunk:list[str]) -> Counter:
        """
        Tokenizes a chunk of text and returns its token frequencies.
        """
        #print(f"The text chunk size is {len(text_chunk)}")
        if self.tokenizer is None:
            self._init_tokenizer()

        assert self.tokenizer is not None, "Tokenizer should be initialized."

        assert isinstance(text_chunk[0],str), "Flattened text chunk should be list[str]."

        if "gpt" in self.tokenizer_name:
            if not isinstance(self.tokenizer, tiktoken.Encoding):
                """Run JavaScript module for gpt-style tokenizer"""
                tokens = self.tokenizer.encode_batch(text_chunk)
            else:
                """load tiktoken gpt-style tokenizer(gpt-4o)"""
                tokens = self.tokenizer.encode_batch(text_chunk)
        else:
            tokens = self.tokenizer(text_chunk)['input_ids']

        if isinstance(tokens[0], list):
            # Flatten list of lists
            tokens = flatten_list(tokens)
        self.tokenize_times += 1

        return Counter(tokens)
    
    def close(self):
        """Gracefully close held resources."""
        if isinstance(self.tokenizer, JSGPTTokenizer):
            self.tokenizer.close()
        # Add cleanup for AutoTokenizer if needed
    
    def terminate(self):
        """Forcibly terminate held resources."""
        if isinstance(self.tokenizer, JSGPTTokenizer):
            self.tokenizer.terminate()
        # Add cleanup for AutoTokenizer if needed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when exiting the 'with' block.
        Passes the exception type to decide how to shut down.
        """
        if exc_type == KeyboardInterrupt:
            print("Worker caught interrupt, terminating...", file=sys.stderr)
            self.terminate() # Hard shutdown
        else:
            self.close() # Graceful shutdown
    
    def __del__(self):
        """
        Fallback destructor. Use terminate() as it's safer.
        """
        # Note: __del__ is unreliable, but this is better than nothing.
        self.terminate()


# --- Helper Generators ---
# These functions stream chunks of text.

def flatten_list(lst:list) -> list:
    return [item for sublist in lst for item in sublist]

def read_chunks_from_jsonl(file_path: str, chunk_size: int) -> Iterator[list[str]]:
    count = 0
    finished = 0
    print(f"Total length: {get_total_lines(file_path)}.")
    return_str = []
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            count += 1
            finished += 1
            try:
                return_str.append(json.loads(line)['conversations'][0]['content'])
            except Exception as e:
                print(f"Error try to load json: {e}")
            if count == chunk_size:
                count = 0
                print(f'Have finished: {finished}')
                yield return_str
                return_str = []
        if count != 0:
            yield return_str
            return_str = []
        return 

def read_chunks_from_file(file_path: str, chunk_size_bytes: int) -> Iterator[list[str]]:
    """
    Reads a local file in binary chunks and yields them as decoded strings.
    This avoids reading the entire file into memory.
    We concat 64kb of data into one single string, and batch these strings as lists.
    This is the best for AutoTokenizer to process.
    """
    SINGLE_CHUNK = 64*1024 # each str is 64kb
    try:
        iterations = int(chunk_size_bytes/SINGLE_CHUNK)
        with open(file_path, "r", encoding="utf-8") as f:
            batch = []
            chunk = ''
            while True:
                for _ in range(iterations):
                    chunk = f.read(SINGLE_CHUNK)
                    batch.append(chunk)
                yield batch
                batch = []

                if not chunk:
                    break
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

def batch_hf_dataset(dataset: Dataset, text_column: str, batch_size: int) -> Iterator[list[str]]:
    """
    Iterates a streaming HF dataset, batches documents,
    and yields a single concatenated text chunk.
    We concat 64kb of data into one single string, and batch these strings as lists.
    This is the best for AutoTokenizer to process.
    """
    SINGLE_CHUNK = 64*1024 #each strlen is 64kb
    batch_texts = []
    str_text = []
    str_size = 0
    lth = 0
    for i, item in enumerate(dataset):
        try:
            text = item[text_column]
            if text:  # Ensure text is not None or empty
                str_text.append(text)
                str_size += sys.getsizeof(text)
                if str_size>SINGLE_CHUNK:
                    batch_texts.append(' '.join(str_text))
                    str_text = []
                    str_size = 0
        except KeyError:
            print(
                f"Error: Text column '{text_column}' not found in dataset.",
                file=sys.stderr
            )
            print(f"Available columns: {list(item.keys())}", file=sys.stderr)
            sys.exit(1)
        except TypeError:
            print(f"Error: Item {i} is not a dictionary? Item: {item}", file=sys.stderr)
            continue # Skip this item

        if len(batch_texts) >= batch_size:
            lth += 1
            if lth == 1:
                print("The first one length is:",len(batch_texts[0]),flush=True)
            yield batch_texts
            batch_texts = []

    print("Total dataset items:",lth,flush=True)
    # Yield any remaining items in the last batch
    if batch_texts:
        yield batch_texts


# --- Main Logic ---

def save_counts(counts: Counter, output_path: str):
    """Saves the aggregated Counter to a JSON file, sorted by frequency."""
    print(f"\nSaving aggregated counts to {output_path}...")
    try:
        # Sort by frequency, descending
        sorted_counts = dict(counts.most_common())
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_counts, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully saved {len(sorted_counts)} unique tokens.")
        print("Top 20 most common tokens:")
        for token, count in list(sorted_counts.items())[:20]:
            print(f"  {token}: {count}")

    except Exception as e:
        print(f"Error saving file to {output_path}: {e}", file=sys.stderr)


def main(args: argparse.Namespace):
    """
    Main function to set up and run.
    """

    # This is the CHUNK_SIZE from your stub
    CHUNK_SIZE_BYTES = args.chunk_size_kb * 1024 
    
    total_counts = Counter()
    
    # Set up the text chunk iterator based on the mode
    if args.dataset_mode:
        if Dataset is None:
            print(
                "Error: --dataset_mode requires the `datasets` library.",
                file=sys.stderr
            )
            print("Please install it with `pip install datasets`", file=sys.stderr)
            sys.exit(1)
            
        print(f"Loading Huggingface Dataset: {args.corpus_path} (streaming)", flush=True)
        try:
            dataset = load_dataset(args.corpus_path, split="train", streaming=True)
            text_chunks_iterator = batch_hf_dataset(
                dataset, args.text_column, args.hf_batch_size
            )
        except Exception as e:
            print(f"Error loading dataset '{args.corpus_path}': {e}", file=sys.stderr)
            sys.exit(1)
    elif args.jsonl:
        text_chunks_iterator = read_chunks_from_jsonl(
            args.corpus_path, CHUNK_SIZE_BYTES
        )

    else:
        print(f"Reading local file: {args.corpus_path}", flush=True)
        print(f"Using chunk size: {args.chunk_size_kb} KB", flush=True)
        text_chunks_iterator = read_chunks_from_file(
            args.corpus_path, CHUNK_SIZE_BYTES
        )
        
    with Worker(tokenizer_name=args.tokenizer) as worker:
        tokenize_and_count = worker.tokenize_and_count

        try:
            print("Process started. Processing chunks and aggregating results...", flush=True)
            chunk_count = 0
            for i, counts_from_chunk in enumerate(text_chunks_iterator):
                total_counts.update(tokenize_and_count(counts_from_chunk))
                chunk_count = i + 1
                
                # Print a progress update every 100 chunks
                if chunk_count % 1 == 0:
                    print(
                        f"  ... Aggregated {chunk_count} chunks. "
                        f"Total unique tokens so far: {len(total_counts)}",
                        flush=True
                    )
            print(f"\nProcessing complete. Aggregated a total of {chunk_count} chunks.")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Stopping...", file=sys.stderr)
            print("Note: Results will be partial.", file=sys.stderr)
        except Exception as e:
            print(f"\nAn error occurred: {e}", file=sys.stderr)
    
    # --- Save Results ---
    if not total_counts:
        print("Warning: No tokens were counted. Is the corpus empty or the path correct?")
    else:
        save_counts(total_counts, args.output_file)


# --- Argument Parsing ---
if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="Count token frequencies in a large corpus NOT using multiprocessing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "tokenizer",
        type=str,
        help="The name or path of the tokenizer."
    )
    
    parser.add_argument(
        "-p","--corpus_path",
        type=str,
        default="../test/corpus.txt",
        help="Path to the large local .txt file OR name of the Hugging Face dataset (e.g., 'Geralt-Targaryen/C4-zh')."
    )
    
    parser.add_argument(
        "-d","--dataset_mode",
        action="store_true",
        help="Enable this flag if 'corpus_path' is a Hugging Face dataset name."
    )
    
    parser.add_argument(
        "-j", "--jsonl",
        action="store_true",
        help="read jsonl file."
    )
    
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        default="token_frequencies.json",
        help="Path to save the final counts. Default: token_frequencies.json"
    )

    parser.add_argument(
        "-c", "--chunk_size_kb",
        type=int,
        default=4096,
        help="Size of each text chunk (in kbytes) to read from a local file. Default: 64"
    )

    parser.add_argument(
        "-t","--text_column",
        type=str,
        default="text",
        help="The name of the column in the HF dataset that contains the text. Default: 'text'"
    )
    parser.add_argument(
        "-b","--hf_batch_size",
        type=int,
        default=1000,
        help="Number of documents to batch together from the HF dataset to form one 'chunk'. Default: 1000"
    )

    # --- Parse args and run ---
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    parsed_args = parser.parse_args()
    main(parsed_args)
