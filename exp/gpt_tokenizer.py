import execjs
import multiprocessing
import os
import sys
from typing import List

# --- Worker Functions ---
# These functions MUST be defined at the top level of the module so they
# can be pickled and sent to worker processes.

# A global variable to hold the tokenizer instance *within each worker process*.
# It will be None in the main process and an execjs object in the workers.
worker_tokenizer = None
count = 0

def _init_worker(js_filename: str):
    """
    Initializer function for each worker process.
    This function is called once per worker when the Pool is created.
    """
    global worker_tokenizer
    try:
        # Open and read the JavaScript file
        with open(js_filename, "r", encoding="utf-8") as f:
            js_code = f.read()
        
        # Compile the JS code and store it in the worker's global variable
        worker_tokenizer = execjs.compile(js_code)
        #print(f"Worker process {os.getpid()} initialized tokenizer.")
    except Exception as e:
        # Handle errors during worker initialization
        print(f"Worker process {os.getpid()} FAILED to initialize: {e}")
        worker_tokenizer = None

def _encode_worker(text: str) -> List[int]:
    """
    The task function that each worker process will execute.
    It uses the (per-process) globally initialized tokenizer.
    """
    global worker_tokenizer
    if worker_tokenizer:
        try:
            # Call the 'tokenize' function in the loaded JavaScript
            return worker_tokenizer.call("tokenize", text)
        except Exception as e:
            # Handle potential errors during tokenization
            print(f"Worker {os.getpid()} error tokenizing text: {e}")
            return [] # Return an empty list on failure
    else:
        # This will happen if _init_worker failed
        print(f"Worker {os.getpid()} has no tokenizer, returning empty list.")
        return []

# --- Main Tokenizer Class ---

class JSGPTTokenizer:
    """
    GPT Tokenizer driven by JavaScript and implemented multiprocessing in Python.
    Please first make sure you have install Node.js by typing "node -v" in the terminal.
    After that, type "npm install gpt-tokenizer" to install the tokenizer module.

    **Note:** When using use_mp=True, it is highly recommended to use
    this class as a context manager (with the 'with' statement) to ensure
    the multiprocessing pool is properly closed.

    TODO: The multiprocessing are not able to shutdown when pressing Ctrl+C.
    This should be fixed.

    Example:
        with JSGPTTokenizer(use_mp=True) as tokenizer:
            texts = ["hello world", "this is a test"]
            all_tokens = tokenizer.encode_batch(texts)
    """
    def __init__(self, js_filename: str = "gpt_tokenizer.mjs", use_mp: bool = False):
        self.js_filename = js_filename
        self.use_mp = use_mp
        self.pool = None

        # 1. Initialize the tokenizer in the *main process*
        # This is used for non-batch .encode() calls
        try:
            with open(js_filename, "r", encoding="utf-8") as f:
                js_code = f.read()
            self.tokenizer = execjs.compile(js_code)
        except Exception as e:
            print(f"Main process failed to initialize tokenizer: {e}")
            self.tokenizer = None

        # 2. Initialize the multiprocessing pool if requested
        if self.use_mp:
            # Using processes=None defaults to os.cpu_count()
            num_processes = None 
            
            print(f"Starting multiprocessing pool (processes={num_processes or os.cpu_count()})...")
            # Create the pool, passing our initializer function and its argument
            self.pool = multiprocessing.Pool(
                processes=num_processes,
                initializer=_init_worker,
                initargs=(self.js_filename,) # Arguments to pass to _init_worker
            )

    def encode(self, text: str) -> List[int]:
        """Encodes a single string using the main process's tokenizer."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer is not initialized in main process.")
        tokens = self.tokenizer.call("tokenize", text)
        return tokens
    
    def encode_batch(self, text: List[str]) -> List[List[int]]:
        """
        Encodes a batch of strings.
        Uses the multiprocessing pool if use_mp=True, otherwise runs serially.
        """
        #print(f"Encoding items of {len(text)}.")
        if self.use_mp and self.pool:
            # 3. Use the pool to map the _encode_worker function to the list of texts
            # This distributes the 'text' list among all worker processes.
            try:
                results = self.pool.map(_encode_worker, text)
                return results
            except Exception as e:
                print(f"Error during multiprocessing batch encoding: {e}")
                return []
        else:
            # 4. Fallback: Run serially in the main process
            print("Running encode_batch serially (use_mp=False or pool not initialized).")
            if not self.tokenizer:
                raise RuntimeError("Tokenizer is not initialized in main process.")
            return [self.tokenizer.call("tokenize", t) for t in text]

    def close(self):
        """
        Safely closes the multiprocessing pool (graceful shutdown).
        Waits for all tasks to finish.
        """
        if self.pool:
            print("Closing multiprocessing pool gracefully...")
            self.pool.close() # Prevents any more tasks
            self.pool.join()  # Waits for all worker processes to finish
            self.pool = None
            print("Pool closed.")

    def terminate(self):
        """
        Forcibly terminates the multiprocessing pool (hard shutdown).
        Used for KeyboardInterrupt.
        """
        if self.pool:
            print("Terminating multiprocessing pool immediately...", file=sys.stderr)
            self.pool.terminate() # Sends SIGTERM to all workers
            self.pool.join()      # Waits for processes to die
            self.pool = None
            print("Pool terminated.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when exiting the 'with' block.
        This is now interrupt-aware.
        """
        if exc_type == KeyboardInterrupt:
            # If the block was exited by Ctrl+C,
            # we MUST terminate, not close/join, to avoid hanging.
            self.terminate()
        else:
            # Normal exit or a different exception, try graceful shutdown.
            self.close()

    def __del__(self):
        """
        Fallback destructor. Use terminate() as it's safer
        during garbage collection than close().
        """
        if self.pool:
            print("Warning: JSGPTTokenizer was not closed. Terminating pool from __del__.", file=sys.stderr)
            self.terminate()

