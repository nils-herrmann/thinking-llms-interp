# %%
import dotenv
dotenv.load_dotenv(".env")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from nnsight import NNsight
import matplotlib.pyplot as plt
from utils import chat, steering_config
import re
import numpy as np
from messages import validation_messages, labels
from tqdm import tqdm
import gc
import os
import utils

os.system('')  # Enable ANSI support on Windows

# %% Evaluation examples - 3 from each category
def load_model_and_vectors(model_name="deepseek-ai/DeepSeek-R1-Distill-8B"):
    model, tokenizer, feature_vectors = utils.load_model_and_vectors(compute_features=True, return_steering_vector_set=True, model_name=model_name)
    return model, tokenizer, feature_vectors

# %%
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Can be changed to use different models
model, tokenizer, feature_vectors = load_model_and_vectors(model_name)

# %% Get activations and response
data_idx = 1

# %%
print("Original response:")
input_ids = tokenizer.apply_chat_template([validation_messages[data_idx]], add_generation_prompt=True, return_tensors="pt").to("cuda")
output_ids = utils.custom_generate_with_projection_removal(
    model,
    tokenizer,
    input_ids,
    max_new_tokens=250,
    label="none", 
    feature_vectors=None,
    steering_config=steering_config[model_name],
)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
print("\n================\n")

# %%
for label in ['adding-knowledge', 'example-testing', 'backtracking', 'uncertainty-estimation']:

        for t in ["negative"]:

            print(f"Label: {label}, {t}")

            input_ids = tokenizer.apply_chat_template([validation_messages[data_idx]], add_generation_prompt=True, return_tensors="pt").to("cuda")
            output_ids = utils.custom_generate_with_projection_removal(
                model,
                tokenizer,
                input_ids,
                max_new_tokens=250,
                label=label,
                feature_vectors=feature_vectors,
                steering_config=steering_config[model_name],
                steer_positive=True if t == "positive" else False,
            )

            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(response)
            print("\n================\n")
# %%
for label in ['adding-knowledge', 'backtracking', 'example-testing', 'uncertainty-estimation']:
    unembed = torch.load(f"data/{model_name.split('/')[-1].lower()}_unembed.pt").to(torch.float32)
    unembed = unembed / unembed.norm(dim=-1, keepdim=True)

    features = feature_vectors[label][steering_config[model_name][label]["pos_layers"][-1]]
    features = features / features.norm(dim=-1, keepdim=True)

    max_sims = torch.topk(features @ unembed.T, k=20, dim=-1)

    print(f"Label: {label}")
    for i, (idx, sim) in enumerate(zip(max_sims.indices, max_sims.values)):
        print(f"{i+1}. Index: {tokenizer.decode(idx.item())}, Similarity: {sim.item()}")

# %%
# print norm of each feature vector
for label in feature_vectors:
    for i, feature in enumerate(feature_vectors[label]):
        print(f"Label: {label}, Feature {i+1}: {feature.norm()}")

# %%
