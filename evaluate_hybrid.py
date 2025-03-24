# %%
import dotenv
dotenv.load_dotenv(".env")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import utils
from collections import defaultdict

def load_models_and_vectors(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", base_model_name="meta-llama/Llama-3.1-8B"):
    """Load both models, tokenizers and feature vectors."""
    model, tokenizer, base_model, base_tokenizer, feature_vectors = utils.load_model_and_vectors(
        compute_features=True, 
        model_name=model_name,
        base_model_name=base_model_name
    )
    return model, tokenizer, base_model, base_tokenizer, feature_vectors

def extract_answer(response):
    """Extract the final answer from the model's response."""
    try:
        # Look for the answer after ####
        answer = response.split("</think>")[-1].strip()
        # Try to convert to float
        return answer
    except:
        return None

def evaluate_answer(question, model_answer, correct_answer):
    """Use chat API to evaluate if the answer is correct."""
    evaluation_prompt = f"""
    Consider the following question with the given correct answer:
    Question: {question}
    Correct answer: {correct_answer}

    Is the following written out response to the question arriving at the correct answer?
    Response: {model_answer}

    Respond with only "correct" or "incorrect".
    """
    
    response = utils.chat(evaluation_prompt)
    return response.strip().lower() == "correct"

def generate_and_evaluate_base(model, tokenizer, question):
    """Generate and evaluate using base model."""
    message = {"role": "user", "content": question}
    input_ids = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        with model.generate(input_ids, max_new_tokens=1000, pad_token_id=tokenizer.eos_token_id) as tracer:
            outputs = model.generator.output.save()
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    extracted_answer = extract_answer(response)
    
    return {
        "response": response,
        "extracted_answer": extracted_answer
    }

def generate_and_evaluate_thinking(model, tokenizer, question):
    """Generate and evaluate using thinking model with steering."""
    message = {"role": "user", "content": question}
    input_ids = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    output_ids = utils.custom_generate_with_projection_removal(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=1000,
        label="none",  # Using backtracking as default steering
        feature_vectors=None,
        steering_config=None,
        steer_positive=True
    )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    extracted_answer = extract_answer(response)
    
    return {
        "response": response,
        "extracted_answer": extracted_answer
    }

def generate_and_evaluate_hybrid(thinking_model, base_model, base_tokenizer, question, feature_vectors, steering_config):
    """Generate and evaluate using hybrid model."""
    message = {"role": "user", "content": question}
    input_ids = base_tokenizer.apply_chat_template([message], add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    output_ids, forced_positions, forced_labels, forced_tokens = utils.custom_hybrid_generate(
        thinking_model,
        base_model,
        base_tokenizer,
        input_ids,
        max_new_tokens=1000,
        feature_vectors=feature_vectors,
        steering_config=steering_config,
        coefficient=3,
        steer_positive=True,
        warmup=7,
        show_progress=False,
        color_output=False
    )
    
    response = base_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    extracted_answer = extract_answer(response)
    
    return {
        "response": response,
        "extracted_answer": extracted_answer,
        "forced_positions": forced_positions,
        "forced_labels": forced_labels,
        "forced_tokens": forced_tokens
    }

def calculate_thinking_length(response):
    """Calculate the length of thinking process between <think> and </think> tags."""
    start_idx = response.find("<think>")
    try:
        end_idx = response.find("</think>")
        if start_idx != -1 and end_idx != -1:
            thinking_text = response[start_idx + 7:end_idx].strip()
            return len(thinking_text.split())
    except:
        pass
    return len(response[start_idx + 7:].strip())

def plot_results(results, model_name):
    """Plot the evaluation results for all models."""
    os.makedirs('figures', exist_ok=True)
    model_id = model_name.split('/')[-1].lower()
    
    # Create subplots for accuracy and thinking length
    plt.style.use('seaborn-v0_8-white')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Calculate metrics for each model
    models = ['base', 'thinking', 'hybrid']
    accuracies = []
    thinking_lengths = []
    
    for model in models:
        # Calculate accuracy
        correct = sum(1 for r in results if r[model]["correct"])
        accuracy = correct / len(results)
        accuracies.append(accuracy)
        
        # Calculate average thinking length
        lengths = [calculate_thinking_length(r[model]["response"]) for r in results]
        avg_length = sum(lengths) / len(lengths)
        thinking_lengths.append(avg_length)
    
    # Plot accuracy bars
    x = np.arange(len(models))
    width = 0.35
    
    bars = ax1.bar(x, accuracies, width, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    # Plot thinking length bars
    bars = ax2.bar(x, thinking_lengths, width, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add length labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=10)
    
    # Improve styling for accuracy plot
    ax1.set_ylabel('Accuracy (%)', fontsize=16, labelpad=10)
    ax1.set_title(f'MATH Evaluation - {model_name}', fontsize=20, pad=20, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([model.title() for model in models], fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)
    
    # Improve styling for thinking length plot
    ax2.set_ylabel('Average Thinking Length (words)', fontsize=16, labelpad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([model.title() for model in models], fontsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f'figures/math_results_{model_id}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %%
# Parameters
n_examples = 1
random.seed(42)

# Create data directory
os.makedirs('data', exist_ok=True)

# Load models and vectors
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
args, _ = parser.parse_known_args()

model_name = args.model
base_model_name = args.base_model
model_id = model_name.split('/')[-1].lower()

# Load models and vectors
model, tokenizer, base_model, base_tokenizer, feature_vectors = load_models_and_vectors(model_name, base_model_name)

# %% Load MATH dataset
test_dataset = load_dataset("HuggingFaceH4/MATH-500", streaming=True).shuffle(seed=42)["test"]

# %% Randomly sample evaluation examples
results = []

# Evaluate each example
for idx, example in tqdm(enumerate(test_dataset), desc="Processing examples"):
    if idx > n_examples:
        break

    question = example["problem"]
    correct_answer = example["answer"]
    
    example_results = {}
    
    # Generate responses for each model
    # Base model
    base_response = generate_and_evaluate_base(base_model, tokenizer, question)
    is_correct = evaluate_answer(question, base_response["extracted_answer"], correct_answer)
    example_results["base"] = {
        "response": base_response["response"],
        "extracted_answer": base_response["extracted_answer"],
        "correct": is_correct
    }
    
    # Thinking model
    thinking_response = generate_and_evaluate_thinking(model, tokenizer, question)
    is_correct = evaluate_answer(question, thinking_response["extracted_answer"], correct_answer)
    example_results["thinking"] = {
        "response": thinking_response["response"],
        "extracted_answer": thinking_response["extracted_answer"],
        "correct": is_correct
    }
    
    # Hybrid model
    hybrid_response = generate_and_evaluate_hybrid(model, base_model, tokenizer, question, feature_vectors, utils.steering_config[model_name])
    is_correct = evaluate_answer(question, hybrid_response["extracted_answer"], correct_answer)
    example_results["hybrid"] = {
        "response": hybrid_response["response"],
        "extracted_answer": hybrid_response["extracted_answer"],
        "correct": is_correct,
        "forced_positions": hybrid_response["forced_positions"],
        "forced_labels": hybrid_response["forced_labels"],
        "forced_tokens": hybrid_response["forced_tokens"]
    }
    
    results.append(example_results)

# %% Save results
with open(f'data/math_evaluation_results_{model_id}.json', 'w') as f:
    json.dump(results, f, indent=2)

plot_results(results, model_name) 
# %%
