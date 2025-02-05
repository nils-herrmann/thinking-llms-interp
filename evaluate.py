# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from nnsight import NNsight
import matplotlib.pyplot as plt
from utils import chat
import re
import numpy as np
from messages import eval_messages, labels
from tqdm import tqdm
import gc
import random
import os

os.system('')  # Enable ANSI support on Windows

random.shuffle(eval_messages)

# %% Evaluation examples - 3 from each category
def load_model_and_vectors():
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", torch_dtype=torch.bfloat16)
    model = NNsight(model).to("cuda")
    
    # Load mean vectors
    mean_vectors_dict = torch.load("mean_vectors.pt")
    print(mean_vectors_dict.keys())
    
    # Compute feature vectors by subtracting overall mean
    overall_mean = mean_vectors_dict['overall']['mean']
    feature_vectors = {}
    
    for label in mean_vectors_dict:
        if label != 'overall' and label in labels:
            feature_vectors[label] = mean_vectors_dict[label]['mean'] - overall_mean
            
    return model, tokenizer, feature_vectors

def get_thinking_activations(model, tokenizer, message_idx):
    """Get activations for a specific evaluation example"""
    message = eval_messages[message_idx]
    
    # Generate response
    tokenized_messages = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
    output = model.generate(
        tokenized_messages,
        max_new_tokens=500,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract thinking process
    think_start = response.index("<think>") + len("<think>")
    try:
        think_end = response.index("</think>")
    except ValueError:
        think_end = len(response)
    thinking_process = response[think_start:think_end].strip()
    
    # Get activations
    layer_outputs = []
    with model.trace(output):
        for layer_idx in range(model.config.num_hidden_layers):
            layer_outputs.append(model.model.layers[layer_idx].output[0].save())
    
    layer_outputs = torch.cat([x.value.cpu().detach().to(torch.float32) for x in layer_outputs], dim=0)
    
    return layer_outputs, thinking_process, response

def plot_similarities(activations, feature_vectors, thinking_process, tokenizer):
    """Color-code text based on highest cosine similarity label for each token"""
    # Get the tokens for the thinking process instead of input
    thinking_tokens = tokenizer(thinking_process, return_tensors="pt").input_ids.to("cuda")
    
    # Reshape activations to match thinking process length
    # Find the start and end indices corresponding to the thinking process
    full_text = thinking_process
    
    # Get activations only for the thinking process portion
    activations = activations
    
    # Calculate cosine similarities for each position and label
    similarities = {}
    for label, feature_vector in feature_vectors.items():
        # Reshape activations to (num_tokens, num_layers, hidden_dim)
        reshaped_activations = activations.reshape(-1, 32, activations.size(-1))
        
        # Normalize feature vector (32, hidden_dim)
        feature_vector = feature_vector / torch.norm(feature_vector, dim=1, keepdim=True)
        
        # Normalize activations (num_tokens, 32, hidden_dim)
        normalized_activations = reshaped_activations / torch.norm(reshaped_activations, dim=2, keepdim=True)
        
        # Calculate similarity for each layer (num_tokens, 32)
        layer_similarities = torch.sum(normalized_activations * feature_vector.unsqueeze(0), dim=2)
        
        # Average over layers (num_tokens,)
        similarities[label] = layer_similarities[:,15:-5].mean(dim=1)
    
    # Convert similarities dict to tensor for easier comparison
    similarity_tensor = torch.stack([similarities[label] for label in feature_vectors.keys()])
    # Get the highest similarity label for each token
    max_label_indices = torch.argmax(similarity_tensor, dim=0)
    
    # Get list of labels for easier indexing
    label_list = list(feature_vectors.keys())
    
    # Create color map using ANSI escape codes
    # Using common terminal colors: red, green, blue, yellow, magenta, cyan
    ansi_colors = ['\033[31m', '\033[32m', '\033[34m', '\033[33m', '\033[35m', '\033[36m']
    reset_color = '\033[0m'
    
    # Cycle through colors if we have more labels than colors
    label_to_color = {label: ansi_colors[i % len(ansi_colors)] 
                     for i, label in enumerate(label_list)}
    
    # Get the tokens
    tokens = tokenizer.convert_ids_to_tokens(thinking_tokens[0])
    
    # Build colored text for console
    colored_text = []
    current_label = None
    current_text = []
    
    for token, label_idx in zip(tokens, max_label_indices):
        label = label_list[label_idx]
        token_text = tokenizer.convert_tokens_to_string([token])
        
        if label != current_label:
            if current_text:
                colored_text.append(f'{label_to_color[current_label]}{tokenizer.convert_tokens_to_string(current_text)}{reset_color}')
                current_text = []
            current_label = label
            
        current_text.append(token)
    
    # Add the last span
    if current_text:
        colored_text.append(f'{label_to_color[current_label]}{tokenizer.convert_tokens_to_string(current_text)}{reset_color}')
    
    # Print the legend and colored text
    print("Color legend:")
    for label, color in label_to_color.items():
        print(f'{color}{label}{reset_color}')
    print("\nColored text:")
    print("".join(colored_text))

def custom_generate_with_projection_removal(model, tokenizer, input_ids, max_new_tokens, label, feature_vectors):
    generated_ids = input_ids.clone().cpu()
    if label in feature_vectors:
        # Move feature vectors to GPU only once, outside the loop
        feature_vector = feature_vectors[label].to("cuda").to(torch.bfloat16)
        normalized_features = feature_vector
    else:
        normalized_features = None
            
    for k in tqdm(range(max_new_tokens), desc="Generating response"):
        # Clear cache at start of each iteration
        input_chunk = generated_ids.to("cuda")
        
        with torch.no_grad():  # Add this to reduce memory usage
            with model.trace(input_chunk) as trace:
                if normalized_features is not None:
                    for layer_idx in range(10,15):
                        hidden_states = model.model.layers[layer_idx].output[0]
                        # Compute projections more efficiently
                        #projection = torch.einsum('sh,h->s', hidden_states[0], normalized_features[layer_idx])
                        #projection_vector = projection[-1:].unsqueeze(-1) * normalized_features[layer_idx]  # Outer product
                        model.model.layers[layer_idx].output[0][:, -1] -= feature_vector[layer_idx].unsqueeze(0)

                        del hidden_states
                
                outputs = model.lm_head.output.save()
        
        next_token = outputs[:, -1, :].argmax(dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).cpu()], dim=1)

        # Explicitly delete tensors
        del trace, outputs, next_token, input_chunk
       
        torch.cuda.empty_cache()
        if k % 10 == 0:
            gc.collect()
    
    gc.collect()
    return generated_ids.cpu()


# %%
model, tokenizer, feature_vectors = load_model_and_vectors()

# %% Get activations and response
data_idx = 1
activations, thinking_process, full_response = get_thinking_activations(model, tokenizer, data_idx)

# %%
plot_similarities(activations, feature_vectors, thinking_process, tokenizer)

# %%
print("Original response:")
input_ids = tokenizer.apply_chat_template([eval_messages[data_idx]], add_generation_prompt=True, return_tensors="pt").to("cuda")
output_ids = custom_generate_with_projection_removal(
        model, 
        tokenizer, 
        input_ids, 
        max_new_tokens=500, 
        label="none",
        feature_vectors=feature_vectors
    )
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
print("\n================\n")

# %%
input_ids = tokenizer.apply_chat_template([eval_messages[data_idx]], add_generation_prompt=True, return_tensors="pt").to("cuda")
output_ids = custom_generate_with_projection_removal(
        model, 
        tokenizer, 
        input_ids, 
        max_new_tokens=500, 
        label="forward-reasoning", 
        feature_vectors=feature_vectors
    )
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
print("\n================\n")

# %%
