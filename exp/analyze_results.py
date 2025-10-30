import json
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer
from typing import List, Iterable

# 这些范围基本覆盖常见汉字（简体/繁体）、扩展区、和常用中文标点
CHINESE_UNICODE_BLOCKS = [
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs (最常用汉字区)
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF), # Extension B
    (0x2A700, 0x2B73F), # Extension C
    (0x2B740, 0x2B81F), # Extension D
    (0x2B820, 0x2CEAF), # Extension E
    (0x2CEB0, 0x2EBEF), # Extension F
    (0x3000, 0x303F),   # CJK Symbols and Punctuation（全角空格、。、「」……）
    (0xFF00, 0xFFEF),   # Halfwidth and Fullwidth Forms（全角标点、全角字母数字）
]

def is_cjk_char(ch: str) -> bool:
    """按 Unicode (UTF 顺序) 判断单个字符是不是中文相关字符。"""
    cp = ord(ch)
    for start, end in CHINESE_UNICODE_BLOCKS:
        if start <= cp <= end:
            return True
    return False

def is_chinese_token(token: str) -> bool:
    """只要 token 里出现过至少一个中文相关字符，就判定为中文 token。"""
    return any(is_cjk_char(ch) for ch in token)

def filter_chinese_tokens(tokens: Iterable[str]) -> List[str]:
    """在一堆 token 里筛出中文 token。"""
    return [tok for tok in tokens if is_chinese_token(tok)]

def find_further_merge(tokenizer_json_path: str) -> dict[int, list[int]]:
    """在token merge层面找到扩展后的所有token。通过读取tokenizer.json中的merges实现"""
    include: dict[int, list[int]] = {}
    with open(tokenizer_json_path, 'r') as f:
        tokenizer_config = json.load(f)
        merges = tokenizer_config['model']['merges']
        vocab = tokenizer_config['model']['vocab']
    for m in merges:
        a, b= m[0], m[1]
        merge_ab = a + b
        int_a = vocab[a]
        int_b = vocab[b]
        int_merge = vocab[merge_ab]
        if int_a not in include:
            include[int_a] = []
        if int_b not in include:
            include[int_b] = []
        include[int_a].append(int_merge)
        include[int_b].append(int_merge)
    update_times = 0
    while True:
        new_include = {}
        for par, chd in include.items():
            new_include[par] = list(sorted(list(set(include[par] + sum([include[chd_ele] for chd_ele in chd if chd_ele in include], start=[])))))
        if new_include == include:
            break
        include = new_include
        update_times += 1
        print(f'update times: {update_times}')
        if update_times >= 100:
            raise ValueError('Too many times.')
    return new_include

def translate_to_str(merge: dict[int, list[int]], tokenizer: AutoTokenizer) -> dict[str, list[str]]:
    new = {}
    for key, value in merge.items():
        new[tokenizer.decode(key)] = [tokenizer.decode(i) for i in value]
    return new

def main(full_name: str):
    if full_name.endswith('/'):
        full_name = full_name[:-1]
    tokenizer = AutoTokenizer.from_pretrained(full_name)
    short_name = os.path.split(full_name)[-1]
    if os.path.exists(f'include_{short_name}.json') and os.path.exists(f'include_{short_name}_str.json'):
        include = json.load(open(f'include_{short_name}.json', encoding='utf-8'))
        include = {int(key): value for key, value in include.items()}
    else:
        include = find_further_merge(os.path.join(full_name, 'tokenizer.json'))
        with open(f'include_{short_name}.json', 'w', encoding='utf-8') as f:
            json.dump(include, f, indent=2, ensure_ascii=False)
        with open(f'include_{short_name}_str.json', 'w', encoding='utf-8') as f:
            json.dump(translate_to_str(include, tokenizer), f, indent=2, ensure_ascii=False)

    freq_file = f'token_frequencies_{short_name}' if os.path.exists(f'token_frequencies_{short_name}') else "token_frequencies.json"
    
    with open(freq_file, 'r', encoding='utf-8') as f:
        freq = json.load(f)
        
    freq_int = {int(key): value for key, value in freq.items()}
    # add other zero freq token
    for i in range(len(tokenizer)):
        if i not in freq_int:
            freq_int[i] = 0
            freq[str(i)] = 0
            
    freq_int = sorted(freq_int.items(), key=lambda item: item[1], reverse=False)
    freq_int = filter(lambda item: is_chinese_token(tokenizer.decode(item[0])), freq_int)
    freq_int = [(tokenizer.decode(item[0]), item[1], sum([freq[str(further_extend)] for further_extend in include[item[0]]]) if (item[0] in include) else 0) for item in freq_int]
    
    with open(f'token_filtered_chinese_{short_name}.json', 'w') as f:
        json.dump(freq_int, f, indent=2, ensure_ascii=False)
    freq_int_ratio = sorted(freq_int, key=lambda item: (item[1]/(item[1]+item[2]) if item[1]+item[2] != 0 else -100000000), reverse=False)
    with open(f'token_filtered_chinese_ratio_{short_name}.json', 'w') as f:
        json.dump(freq_int_ratio, f, indent=2, ensure_ascii=False)
        
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("name_or_path", type=str, help="The name or path of the tokenizer.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.name_or_path)
