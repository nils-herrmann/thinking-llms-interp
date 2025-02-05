# %%
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import NNsight
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import utils

# %%
def load_model_and_vectors(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    model, tokenizer, mean_vectors_dict = utils.load_model_and_vectors(compute_features=False, model_name=model_name)
    return model, tokenizer, mean_vectors_dict

def find_label_positions(annotated_text, original_text, tokenizer, label):
    """Find the token positions in original_text for sentences labeled in annotated_text"""
    pattern = f'\\["{label}"\\]([^\\[]+?)(?=\\[|$)'
    matches = re.finditer(pattern, annotated_text)
    positions = []
    
    # Tokenize original text once
    original_tokens = tokenizer(original_text, return_tensors="pt").input_ids[0]
    
    for match in matches:
        # Get the labeled text and clean it
        labeled_text = match.group(1).strip()
        
        # Find this text in the original text
        if labeled_text in original_text:
            # Find start position in original text
            start_char = original_text.index(labeled_text)
            end_char = start_char + len(labeled_text)
            
            # Get text before the match to find token start position
            text_before = original_text[:start_char]
            before_tokens = tokenizer(text_before, return_tensors="pt").input_ids[0]
            
            # Get tokens for the labeled section
            section_tokens = tokenizer(labeled_text, return_tensors="pt").input_ids[0]
            
            start_pos = len(before_tokens) - 1  # -1 for BOS token
            end_pos = start_pos + len(section_tokens) - 1  # -1 for BOS token
            
            # Verify the token sequence matches
            predicted_tokens = original_tokens[start_pos:end_pos]
            if len(predicted_tokens) > 0:  # Only add if we found valid tokens
                positions.append({
                    'start': start_pos,
                    'end': end_pos,
                    'text': labeled_text  # Keep for debugging
                })
    return positions

def compute_cross_entropy_metric(logits):
    """Compute cross entropy between predicted distribution and detached version"""
    probs = F.softmax(logits, dim=-1)
    detached_probs = F.softmax(logits.detach(), dim=-1)
    return F.cross_entropy(logits, detached_probs.argmax(dim=-1))

def analyze_layer_effects(model, tokenizer, text, label, mean_vectors_dict, label_positions):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    layer_activations = []
    layer_gradients = []
    
    with model.trace() as tracer:
        with tracer.invoke(input_ids) as invoker:
            # Collect activations from each layer
            for layer_idx in range(model.config.num_hidden_layers):
                layer_activations.append(model.model.layers[layer_idx].output[0].save())
                layer_gradients.append(model.model.layers[layer_idx].output[0].grad.save())
            
            # Get logits for the endpoints
            logits = model.lm_head.output.save()
            
            # Compute cross entropy metric for each labeled section
            total_value = 0
            for pos in label_positions:
                if pos['end'] < logits.shape[1]:  # Ensure position is within sequence length
                    value = compute_cross_entropy_metric(logits[0, pos['end']])
                    total_value += value
            
            # Backward pass
            total_value.backward()
    
    # Wait for all computations to complete
    torch.cuda.synchronize()
                
    # Compute patching effects
    patching_effects = []
    overall_mean = mean_vectors_dict['overall']['mean']
    
    for layer_idx in range(model.config.num_hidden_layers):
        layer_effects = []
        for pos in label_positions:
            # Get activations and gradients for the entire labeled section
            activations = layer_activations[layer_idx].value[0, pos['start']:pos['end']].cpu()
            gradients = layer_gradients[layer_idx].value[0, pos['start']:pos['end']].cpu()
            mean_activation = overall_mean[layer_idx].cpu()
            
            # Compute patching effect for this section
            if activations.shape[0] > 0:  # Ensure we have tokens to analyze
                effect = torch.mean((activations - mean_activation.unsqueeze(0)) * gradients)
                layer_effects.append(effect.item())
        
        # Average effects across all labeled sections in this layer
        if layer_effects:
            patching_effects.append(np.abs(np.mean(layer_effects)))
        else:
            patching_effects.append(0.0)
    
    return patching_effects

def plot_layer_effects(layer_effects, model_name):
    plt.figure(figsize=(12, 8))
    
    # Set style
    plt.style.use('ggplot')
    
    # Color scheme
    colors = ['#2E86C1', '#E67E22', '#27AE60', '#C0392B']
    
    for (label, effects), color in zip(layer_effects.items(), colors):
        # Only plot if we have more than one example for this label
        if len(effects) > 1:
            mean_effects = np.mean(effects, axis=0)
            
            # Apply smoothing using convolution
            window_size = 3
            kernel = np.ones(window_size) / window_size
            smoothed_effects = np.convolve(mean_effects, kernel, mode='valid')
            
            # Adjust x-axis for smoothed data
            x = range(len(smoothed_effects))
            
            # Plot smoothed line with confidence band
            std_effects = np.std(effects, axis=0)
            std_smoothed = np.convolve(std_effects, kernel, mode='valid')
            plt.fill_between(x, 
                           smoothed_effects - std_smoothed,
                           smoothed_effects + std_smoothed,
                           alpha=0.2, 
                           color=color)
            
            plt.plot(x, smoothed_effects, 
                    label=f"{label.replace('-', ' ').title()} (n={len(effects)})",
                    color=color,
                    linewidth=2.5,
                    marker='o',
                    markersize=4)
    
    plt.xlabel('Layer', fontsize=12, fontweight='bold')
    plt.ylabel('Average Patching Effect', fontsize=12, fontweight='bold')
    plt.title('Layer-wise Attribution Effects Across Model', 
             fontsize=14, 
             fontweight='bold', 
             pad=20)
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize legend
    plt.legend(bbox_to_anchor=(1.05, 1), 
              loc='upper left', 
              borderaxespad=0.,
              frameon=True,
              fontsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Get model identifier for file naming
    model_id = model_name.split('/')[-1].lower()
    
    # Update save path
    plt.savefig(f'figures/layer_effects_{model_id}.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    plt.show()
    plt.close()

# %%
# Load model and data
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Can be changed to use different models
model, tokenizer, mean_vectors_dict = load_model_and_vectors(model_name)

# %%
with open('data/responses.json', 'r') as f:
    results = json.load(f)

# %%
labels = ['adding-knowledge', 'uncertainty-estimation', 'example-testing', 'backtracking']
n_examples = 30  # Number of examples to analyze per label

# Store results
layer_effects = {label: [] for label in labels}

# Analyze each label
for label in labels:
    print(f"Analyzing label: {label}")
    for example in tqdm(results[:n_examples]):
        original_text = example['thinking_process']
        annotated_text = example['annotated_thinking']

        
        # Find token positions of labeled sentences
        label_positions = find_label_positions(annotated_text, original_text, tokenizer, label)

        if label_positions:  # Only process if we found labeled sentences
            effects = analyze_layer_effects(
                model, 
                tokenizer, 
                original_text, 
                label, 
                mean_vectors_dict,
                label_positions
            )
            layer_effects[label].append(effects)

# %% Plot results
plot_layer_effects(layer_effects, model_name)

# %%