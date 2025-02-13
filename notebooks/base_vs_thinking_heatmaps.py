# %%
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import random
from typing import List, Dict, Any
from tiny_dashboard.visualization_utils import activation_visualization
from IPython.display import HTML, display

# %% Set model names
deepseek_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
original_model_name = "Qwen/Qwen2.5-14B"

# %% Load models

deepseek_tokenizer = AutoTokenizer.from_pretrained(deepseek_model_name)
deepseek_model = AutoModelForCausalLM.from_pretrained(
    deepseek_model_name,
    torch_dtype=torch.float16,  # Use float16 for memory efficiency
    device_map="auto"  # Automatically handle device placement
)

original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
original_model = AutoModelForCausalLM.from_pretrained(
    original_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# %% Load data

annotated_responses_json_path = f"../data/annotated_responses_{deepseek_model_name.split('/')[-1].lower()}.json"
original_messages_json_path = f"../data/base_responses_{deepseek_model_name.split('/')[-1].lower()}.json"

tasks_json_path = "../data/tasks.json"

if not os.path.exists(annotated_responses_json_path):
    raise FileNotFoundError(f"Annotated responses file not found at {annotated_responses_json_path}")
if not os.path.exists(original_messages_json_path):
    raise FileNotFoundError(f"Original messages file not found at {original_messages_json_path}")
if not os.path.exists(tasks_json_path):
    raise FileNotFoundError(f"Tasks file not found at {tasks_json_path}")

print(f"Loading existing annotated responses from {annotated_responses_json_path}")
with open(annotated_responses_json_path, 'r') as f:
    annotated_responses_data = json.load(f)["responses"]
random.shuffle(annotated_responses_data)

print(f"Loading existing original messages from {original_messages_json_path}")
with open(original_messages_json_path, 'r') as f:
    original_messages_data = json.load(f)["responses"]
random.shuffle(original_messages_data)

print(f"Loading existing tasks from {tasks_json_path}")
with open(tasks_json_path, 'r') as f:
    tasks_data = json.load(f)

# %% Pick a random response uuid
response_uuid = random.choice(annotated_responses_data)["response_uuid"]

# %% Prepare model input

def prepare_model_input(
    response_uuid: str,
    annotated_responses_data: List[Dict[str, Any]],
    tasks_data: List[Dict[str, Any]],
    original_messages_data: List[Dict[str, Any]],
    tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    """
    Prepare model input for a given response UUID.
    Returns the tokenized input ready for the model.
    
    Returns:
        Dict with keys:
            'prompt_and_response_ids': Tensor of shape (1, sequence_length)
            'annotated_response': str
    """
    # Fetch the relevant response data
    annotated_response_data = next((r for r in annotated_responses_data if r["response_uuid"] == response_uuid), None)
    if not annotated_response_data:
        raise ValueError(f"Could not find annotated response data for UUID {response_uuid}")
    
    task_data = next((t for t in tasks_data if t["task_uuid"] == annotated_response_data["task_uuid"]), None)
    if not task_data:
        raise ValueError(f"Could not find task data for UUID {annotated_response_data['task_uuid']}")
    
    base_response_data = next((m for m in original_messages_data if m["response_uuid"] == response_uuid), None)
    if not base_response_data:
        raise ValueError(f"Could not find base response data for UUID {response_uuid}")
    
    # Build prompt message
    prompt_message = [task_data["prompt_message"]]
    prompt_message_input_ids = tokenizer.apply_chat_template(
        conversation=prompt_message,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    # Process base response
    base_response_str = base_response_data["response_str"]
    if base_response_str.startswith("<think>"):
        base_response_str = base_response_str[len("<think>"):]
    base_response_input_ids = tokenizer.encode(
        text=base_response_str,
        add_special_tokens=False,
        return_tensors="pt"
    )
    
    return {
        'prompt_and_response_ids': torch.cat(
            tensors=[prompt_message_input_ids, base_response_input_ids],
            dim=1
        ),
        'annotated_response': annotated_response_data["annotated_response"]
    }

model_input = prepare_model_input(
    response_uuid=response_uuid,
    annotated_responses_data=annotated_responses_data,
    tasks_data=tasks_data,
    original_messages_data=original_messages_data,
    tokenizer=deepseek_tokenizer
)

# %% Feed the input to both models and get the logits for all tokens

# Get logits from both models
with torch.no_grad():
    # DeepSeek model logits
    deepseek_outputs = deepseek_model(
        input_ids=model_input['prompt_and_response_ids'].to(deepseek_model.device)
    )
    deepseek_logits = deepseek_outputs.logits
    
    # Original model logits
    original_outputs = original_model(
        input_ids=model_input['prompt_and_response_ids'].to(original_model.device)
    )
    original_logits = original_outputs.logits

# Move logits to CPU for easier processing
deepseek_logits = deepseek_logits.cpu()
original_logits = original_logits.cpu()

# Assert both logits have the same shape
assert deepseek_logits.shape == original_logits.shape

# %% Calculate the KL divergence between the logits

def calculate_kl_divergence(p_logits, q_logits):
    """
    Calculate KL divergence between two distributions given their logits.
    Uses PyTorch's built-in KL divergence function with log_softmax.
    """
    # Convert logits directly to log probabilities
    p_log = torch.nn.functional.log_softmax(p_logits, dim=-1)
    q_log = torch.nn.functional.log_softmax(q_logits, dim=-1)
    
    # Calculate KL divergence using PyTorch's function
    kl_div = torch.nn.functional.kl_div(
        p_log,      # input in log-space
        q_log,      # target in log-space
        reduction='none',
        log_target=True
    )

    # Sum over vocabulary dimension
    kl_div = kl_div.sum(dim=-1)
    
    # Print some debug information
    print(f"KL divergence stats - mean: {kl_div.mean().item():.4f}, max: {kl_div.max().item():.4f}")
    
    return kl_div.squeeze()

# Calculate KL divergence for each position
kl_divergence = calculate_kl_divergence(deepseek_logits, original_logits)
# Convert to float32 and ensure positive values
kl_divergence = kl_divergence.float()
kl_divergence = torch.clamp(kl_divergence, min=0.0)

# %% Create interactive visualization using activation_visualization

# Get the tokens for visualization
tokens = deepseek_tokenizer.convert_ids_to_tokens(
    model_input['prompt_and_response_ids'][0]
)

html = activation_visualization(
    tokens,
    kl_divergence,  # Already in correct shape (sequence_length,)
    deepseek_tokenizer,
    title="KL Divergence between Models",
    relative_normalization=False,
)
display(HTML(html))
