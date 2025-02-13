# %%
from transformers import AutoTokenizer
import json
import random
from tqdm import tqdm
import os
from deepseek_steering.utils import chat
import deepseek_steering.utils as utils
import re

def load_model_and_vectors(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    model, tokenizer, feature_vectors = utils.load_model_and_vectors(compute_features=True, model_name=model_name)
    return model, tokenizer, feature_vectors

def generate_response(model, tokenizer, message, feature_vectors):
    input_ids = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_text=True)
    
    output_ids = utils.custom_generate_with_projection_removal(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=5000,
        label="none",
        feature_vectors=feature_vectors,
        layers=[],
        coefficient=1,
        steer_positive=False,
        show_progress=False
    )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def extract_answer(response):
    """Extract the answer after </think> tag"""
    try:
        think_end = response.index("</think>")
        answer = response[think_end + len("</think>"):].strip()
        return answer
    except ValueError:
        return "THINKING_TIMEOUT"

def evaluate_answer(question, model_answer, correct_answer):
    """Use chat function to evaluate if the answer is correct"""
    prompt = f"""
    Given a question, the model's answer, and the correct answer, determine if the model's answer is correct.
    
    Question: {question}
    Model's Answer: {model_answer}
    Correct Answer: {correct_answer}
    
    Is the model's answer correct? Answer with just 'yes' or 'no'.
    """
    
    response = chat(prompt, max_tokens=10)
    return response.strip().lower() == "yes"

# %% Parameters
n_examples = 10  # Number of examples to evaluate
random.seed(42)
model_name = "Qwen/Qwen2.5-14B-Instruct"
model_id = model_name.split('/')[-1].lower()

# Create directories
os.makedirs('data', exist_ok=True)

# %% Load model and task data
print("Loading model and data...")
model, tokenizer, feature_vectors = load_model_and_vectors(model_name)

with open('data/answer_tasks.json', 'r') as f:
    tasks = json.load(f)

# Randomly sample tasks
selected_tasks = random.sample(tasks, n_examples)

# %% Generate and evaluate responses
results = []

for task in tqdm(selected_tasks, desc="Evaluating tasks"):
    question = task['prompt_message']
    correct_answer = task['answer']
    
    # Generate model response
    response = generate_response(model, tokenizer, question, feature_vectors)
    
    # Extract model's answer
    model_answer = extract_answer(response)
    
    # Evaluate correctness
    is_correct = False if model_answer == "THINKING_TIMEOUT" else evaluate_answer(question, model_answer, correct_answer)
    
    result = {
        "task_uuid": task['task_uuid'],
        "task_category": task['task_category'],
        "question": question["content"],
        "model_response": response,
        "model_answer": model_answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "thinking_timeout": model_answer == "THINKING_TIMEOUT"
    }
    results.append(result)

# %% Calculate and display statistics
total = len(results)
correct = sum(1 for r in results if r['is_correct'])
timeouts = sum(1 for r in results if r['thinking_timeout'])
accuracy = correct / total

category_stats = {}
for result in results:
    category = result['task_category']
    if category not in category_stats:
        category_stats[category] = {'correct': 0, 'total': 0}
    category_stats[category]['total'] += 1
    if result['is_correct']:
        category_stats[category]['correct'] += 1

print(f"\nOverall Accuracy: {accuracy:.2%}")
print(f"Thinking Timeouts: {timeouts}/{total} ({timeouts/total:.2%})")
print("\nAccuracy by Category:")
for category, stats in category_stats.items():
    cat_accuracy = stats['correct'] / stats['total']
    print(f"{category}: {cat_accuracy:.2%} ({stats['correct']}/{stats['total']})")

# Save results
output_file = f'data/performance_eval_results_{model_id}.json'
with open(output_file, 'w') as f:
    json.dump({
        'model_name': model_name,
        'overall_accuracy': accuracy,
        'category_stats': category_stats,
        'results': results
    }, f, indent=2)

# %%
