# %%
import dotenv
dotenv.load_dotenv(".env")

import torch
import gc
from tqdm import tqdm
from utils import load_model_and_vectors, steering_config, custom_hybrid_generate
import argparse
from messages import validation_messages

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B")
args, _ = parser.parse_known_args()

model, tokenizer, base_model, base_tokenizer, feature_vectors = load_model_and_vectors(model_name=args.model_name, base_model_name=args.base_model_name)

# %%
data_idx = 0
input_ids = tokenizer.apply_chat_template([validation_messages[data_idx]], add_generation_prompt=True, return_tensors="pt").to("cuda")

base_output_ids, forced_positions, forced_labels, forced_tokens = custom_hybrid_generate(
    model,
    base_model,
    base_tokenizer,
    input_ids,
    max_new_tokens=500,
    feature_vectors=feature_vectors,
    steering_config=steering_config[args.model_name],
    coefficient=3,
    steer_positive=True,
    warmup=7,
    show_progress=True,
    color_output=True
)

# %%
