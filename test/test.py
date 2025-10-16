import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

"""from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B', trust_remote_code=True)


tokenize_ids = tokenizer("我是一只猫，哈！")['input_ids']
tokenize_result = [tokenizer.decode(id) for id in tokenize_ids]
print(tokenize_result)"""
#tokenizer.decode([51461])


#converted_tokens = tokenizer.convert_ids_to_tokens([51461])


#b' \xe6\xa0'.decode("utf-8", errors='replace')


#tokenizer.decode([51461, 117])
#' 根'

#tokenizer.convert_ids_to_tokens([51461, 117])
#[b' \xe6\xa0', b'\xb9']

#b' \xe6\xa0\xb9'.decode("utf-8", errors='replace')
"""
a - low rate
b - low rate
a+b - high rate
"""


import jsonlines
import os

cor = "wiki_zh_2019\wiki_zh"

"""with open("corpus.txt", "a", encoding="utf-8") as corpus:
    for dir in os.scandir(cor):
        for file in os.scandir(dir):
            with jsonlines.open(cor+"/"+dir.name+"/"+file.name, "r") as reader:
                for item in reader:
                    corpus.writelines(item["text"])"""


