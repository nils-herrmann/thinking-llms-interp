#%%

import json
import re

from transformers import AutoTokenizer

# %%

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

# %%

# Load data
with open('../data/tasks.json', 'r') as f:
    tasks = json.load(f)

with open('../data/annotated_responses_deepseek-r1-distill-qwen-14b.json', 'r') as f:
    responses = json.load(f)['responses']

# %%

backtracking_suffix_phrases_counts = {}
category = "backtracking"
suffix_length = 1

for item in responses:
    text = item['annotated_response']
    pattern = r'\["([\w-]+)"\](.*?)\["end-section"\]'
    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        if match.group(1) == category:
            text = match.group(2).strip()
            # Tokenize the text
            tokens = tokenizer.encode(text)[1:]

            # get the first token of the suffix
            suffix = tokens[:suffix_length]

            suffix_text = tokenizer.decode(suffix)

            backtracking_suffix_phrases_counts[suffix_text] = backtracking_suffix_phrases_counts.get(suffix_text, 0) + 1

for phrase, count in sorted(backtracking_suffix_phrases_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{phrase}: {count}")

# %%
