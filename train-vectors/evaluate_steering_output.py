# %%
import sys
import os
import torch
import gc
import argparse
import random
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path for imports
sys.path.append('..')

import utils
from utils import steering_opt

# Annotation patterns from optimize_steering_vectors.py
ANNOTATION_PATTERN = re.compile(r'\["([\d.]+):(\S+?)"\](.*?)\["end-section"\]', re.DOTALL)
CATEGORY_PATTERN = re.compile(r'\["[\d.]+:(\S+?)"\]')

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate individual steering vector effects')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.1-8B',
                      help='Model for base generation')
    parser.add_argument('--layer', type=int, default=6,
                      help='Layer to apply steering to')
    parser.add_argument('--n_examples', type=int, default=5,
                      help='Number of examples to evaluate')
    parser.add_argument('--max_new_tokens', type=int, default=200,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1e-19,
                      help='Temperature for sampling')
    parser.add_argument('--coefficient', type=float, default=1.0,
                      help='Steering coefficient')
    parser.add_argument('--steering_vector_idx', type=str, default=None,
                      help='Comma-separated indices of steering vectors to use (e.g., "0,2,5"). If not specified, uses all vectors.')
    parser.add_argument('--context_sentences', type=int, default=0,
                      help='Number of additional sentences to include after target completion')
    parser.add_argument('--steering_token_window', type=int, default=50,
                      help='Number of previous tokens in the target completion to apply the steering vector to')
    parser.add_argument('--test_examples_pct', type=float, default=0.30,
                      help='Percentage of examples to use for testing (rest would be used for training)')
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
    return parser.parse_args()

def get_label_positions(annotated_thinking, response_text, tokenizer, context_sentences=0):
    """Parse SAE annotations and find token positions for each label"""
    label_positions = {}
    
    # Use a pattern that captures labeled segments in the format [activation:category-name] text [end-section]
    # Now supporting activation strength values in the format [56.86:category-name]
    matches = list(ANNOTATION_PATTERN.finditer(annotated_thinking))
    
    # Create character to token mapping once
    char_to_token = utils.get_char_to_token_map(response_text, tokenizer)
    
    # Split response into sentences for context
    sentences = utils.split_into_sentences(response_text, min_words=0)
    
    for match in matches[:-1]:
        activation_str = match.group(1).strip()
        label = match.group(2).strip()
        text = match.group(3).strip()
        
        try:
            activation = float(activation_str)
        except ValueError:
            print(f"Warning: Could not parse activation value '{activation_str}' for category '{label}'")
            continue
            
        if not text:  # Skip empty text
            continue
            
        # Find this text in the original response
        pattern = r'(?:[.?!;\n]|\n\n)\s*(' + re.escape(text) + ')'
        match = re.search(pattern, response_text)
        text_pos = match.start(1) if match else -1
        if text_pos >= 0:
            # Get start and end token positions
            token_start = char_to_token.get(text_pos, None)
            token_end = char_to_token.get(text_pos + len(text) - 1, None)
            
            # Adjust token_end to include the entire token
            if token_end is not None:
                token_end += 1

            if token_start is None or token_end is None or token_start >= token_end:
                continue
            
            # Find the sentence containing our target text
            target_sentence_idx = -1
            for i, sentence in enumerate(sentences):
                if text in sentence:
                    target_sentence_idx = i
                    break
            
            if target_sentence_idx == -1:
                continue
                
            # Get additional context sentences if requested
            additional_context = ""
            if context_sentences > 0 and target_sentence_idx < len(sentences) - 1:
                end_idx = min(target_sentence_idx + context_sentences + 1, len(sentences))
                additional_sentences = sentences[target_sentence_idx + 1:end_idx]
                
                # Find the original whitespace between sentences
                if additional_sentences:
                    # Get the text up to the end of our target text
                    text_end_pos = text_pos + len(text)
                    # Get the text up to the start of the next sentence
                    next_sentence_start = response_text.find(additional_sentences[0], text_end_pos)
                    if next_sentence_start > text_end_pos:
                        # Extract the original whitespace
                        original_whitespace = response_text[text_end_pos:next_sentence_start]
                        # Use the original whitespace to join sentences
                        additional_context = original_whitespace + original_whitespace.join(additional_sentences)
                    else:
                        # Fallback to space if we can't find the original whitespace
                        additional_context = " " + " ".join(additional_sentences)
                
                # Update token_end to include additional context
                if additional_context:
                    context_end_pos = text_pos + len(text) + len(additional_context)
                    context_token_end = char_to_token.get(context_end_pos - 1, None)
                    if context_token_end is not None:
                        token_end = context_token_end + 1
            
            # If we found valid token positions
            if label not in label_positions:
                label_positions[label] = []
            label_positions[label].append((token_start, token_end, text + additional_context, activation, text_pos))
    
    return label_positions

def get_sorted_categories(responses_data):
    """Extract all unique categories from responses data and return them sorted alphabetically"""
    categories = set()
    
    for resp in responses_data:
        if not resp.get('annotated_thinking'):
            continue
            
        # Extract category names from annotated thinking
        matches = CATEGORY_PATTERN.finditer(resp['annotated_thinking'])
        for match in matches:
            category = match.group(1).strip()
            categories.add(category)
    
    return sorted(list(categories))

def extract_examples_for_category(responses_data, category_name, tokenizer, context_sentences=0):
    """Extract examples for a specific category from the responses data"""
    examples_for_category = []
    
    # Process each response to extract labeled segments for the specified category
    for resp in tqdm(responses_data, desc=f"Extracting examples for {category_name}"):
        if not resp.get('annotated_thinking'):
            continue

        full_text = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{resp['original_message']['content']}\n\nStep by step answer:\n{resp['thinking_process']}"
        
        # Look for the specific category in the annotated thinking
        if category_name not in resp['annotated_thinking']:
            continue
            
        label_positions = get_label_positions(resp['annotated_thinking'], full_text, tokenizer, context_sentences)
        
        if category_name in label_positions:
            for start, end, text, activation, text_pos in label_positions[category_name]:
                # Get the text up to this point using the saved text_pos
                context = full_text[:text_pos]

                # Check if context ends properly
                if context[-1] not in ['.', '?', '!', ';', '\n', '\n\n'] and context[-2] not in ['.', '?', '!', ';', '\n', '\n\n'] and context.strip()[-1] not in ['.', '?', '!', ';', '\n', '\n\n']:
                    continue
                
                examples_for_category.append({
                    'prompt': context,
                    'target_completion': text,
                    'original_question': resp['original_message']['content'],
                    'full_thinking': resp['thinking_process'],
                    'activation': activation
                })
    
    return examples_for_category

def load_steering_vectors(model_id):
    """Load steering vectors from train_vectors output"""
    vectors_path = f"results/vars/optimized_vectors_{model_id}.pt"
    if not os.path.exists(vectors_path):
        print(f"Warning: No steering vectors found at {vectors_path}")
        return {}
    return torch.load(vectors_path)

def test_steering_on_example(model, tokenizer, vector, layer, example, coefficient, max_new_tokens, temperature, steering_token_window):
    """Test the steering vector on an example, similar to test_on_unseen_example from optimize_steering_vectors.py"""
    prompt = example['prompt']
    target_completion = example['target_completion']
    
    print(f"PROMPT:\n{prompt}")
    print("\n--- Generated Completions ---")
    
    # Generate without steering
    generated_tokens = model.generate(
        **tokenizer(prompt, return_tensors='pt').to(model.device), 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=temperature > 0,
        suppress_tokens=[tokenizer.eos_token_id] if temperature <= 0 else None
    )
    unsteered_completion = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    unsteered_generated_part = unsteered_completion[len(prompt):]
    print("UNSTEERED:")
    print(unsteered_generated_part)
    print()
    
    # Figure out the correct slice for steering
    prompt_tok = tokenizer(prompt, return_tensors='pt')['input_ids'][0]
    target_tok = tokenizer(target_completion, return_tensors='pt')['input_ids'][0]
    prompt_len = len(prompt_tok)
    target_len = len(target_tok)
    if steering_token_window is None:
        steering_start = prompt_len
    else:
        steering_start = prompt_len + max(0, target_len - steering_token_window)
    steering_token_slice = slice(steering_start, None)
    
    # Generate with steering
    steering_hook = (layer, steering_opt.make_steering_hook_hf(vector * coefficient, token=steering_token_slice))
    with steering_opt.hf_hooks_contextmanager(model, [steering_hook]): 
        generated_tokens = model.generate(
            **tokenizer(prompt, return_tensors='pt').to(model.device), 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=temperature > 0
        )
        steered_completion = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        steered_generated_part = steered_completion[len(prompt):]
        print("STEERED:")
        print(steered_generated_part)
    
    print("\nTARGET:")
    print(target_completion)
    print("\n" + "-"*50 + "\n")
    
    return unsteered_completion, steered_completion

def evaluate_steering_vectors(args):
    """Main evaluation function"""    
    # Load model
    base_model_id = args.base_model.split('/')[-1].lower()
    print(f"Loading model {args.base_model}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        device_map="auto",
        load_in_8bit=args.load_in_8bit,
        torch_dtype=torch.bfloat16
    )

    torch.set_default_device(model.device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Set up paths for loading annotated responses
    thinking_model_name = utils.model_mapping.get(args.base_model, base_model_id)
    if thinking_model_name is None:
        thinking_model_name = base_model_id
    thinking_model_short = thinking_model_name.split('/')[-1].lower()
    
    responses_json_path = f"../generate-responses/results/vars/responses_{thinking_model_short}.json"
    annotated_responses_json_path = f"../generate-responses/results/vars/annotated_responses_{thinking_model_short}.json"
    
    if not os.path.exists(responses_json_path):
        raise FileNotFoundError(f"Responses file not found at {responses_json_path}")
    if not os.path.exists(annotated_responses_json_path):
        raise FileNotFoundError(f"Annotated responses file not found at {annotated_responses_json_path}")
    
    # Load responses and annotations
    print(f"Loading responses from {responses_json_path}")
    with open(responses_json_path, 'r') as f:
        responses_data = json.load(f)

    print(f"Loading annotated responses from {annotated_responses_json_path}")
    with open(annotated_responses_json_path, 'r') as f:
        annotated_responses_data = json.load(f)
    
    # Match responses with their annotations
    valid_responses = []
    for i, resp in enumerate(responses_data):
        if i < len(annotated_responses_data):
            annotated_resp = annotated_responses_data[i]
            # Verify that the responses match by question_id and dataset_name
            if (resp['question_id'] == annotated_resp['question_id'] and 
                resp['dataset_name'] == annotated_resp['dataset_name'] and
                annotated_resp.get('annotated_thinking')):
                # Create merged response with annotated_thinking
                merged_resp = resp.copy()
                merged_resp['annotated_thinking'] = annotated_resp['annotated_thinking']
                valid_responses.append(merged_resp)
    
    print(f"Found {len(valid_responses)} responses with annotations")
    
    # Load steering vectors
    print(f"Loading steering vectors...")
    all_steering_vectors = load_steering_vectors(base_model_id)
    
    if not all_steering_vectors:
        print("No steering vectors found! Please train vectors first.")
        return
    
    # Get all available categories
    all_categories = get_sorted_categories(valid_responses)
    print(f"Found {len(all_categories)} categories in annotated responses:")
    for i, category in enumerate(all_categories):
        print(f"  [{i}] {category}")
    
    print(f"Found {len(all_steering_vectors)} steering vectors:")
    for i, vector_name in enumerate(all_steering_vectors.keys()):
        print(f"  [{i}] {vector_name}")
    
    # Filter steering vectors based on indices if specified
    if args.steering_vector_idx is not None:
        try:
            indices = [int(idx.strip()) for idx in args.steering_vector_idx.split(',')]
            vector_list = list(all_steering_vectors.items())
            
            # Validate indices
            invalid_indices = [idx for idx in indices if idx < 0 or idx >= len(vector_list)]
            if invalid_indices:
                print(f"Warning: Invalid indices {invalid_indices}. Valid range is 0-{len(vector_list)-1}")
                indices = [idx for idx in indices if 0 <= idx < len(vector_list)]
            
            if not indices:
                print("No valid indices provided. Using all steering vectors.")
                steering_vectors = all_steering_vectors
            else:
                steering_vectors = {vector_list[idx][0]: vector_list[idx][1] for idx in indices}
                print(f"Using {len(steering_vectors)} selected steering vectors: {list(steering_vectors.keys())}")
                
        except ValueError as e:
            print(f"Error parsing steering vector indices: {e}")
            print("Using all steering vectors.")
            steering_vectors = all_steering_vectors
    else:
        steering_vectors = all_steering_vectors
    
    # Process examples for each steering vector
    for vector_name, vector in steering_vectors.items():
        print(f"\n{'='*80}")
        print(f"Processing steering vector: {vector_name}")
        print(f"{'='*80}")
        
        # Extract examples for this category
        examples = extract_examples_for_category(valid_responses, vector_name, tokenizer, args.context_sentences)
        
        if not examples:
            print(f"No examples found for category {vector_name}")
            continue
        
        print(f"Found {len(examples)} examples for {vector_name}")
        
        # Randomly sample examples if we have more than requested
        if len(examples) > args.n_examples:
            examples = random.sample(examples, args.n_examples)
            print(f"Randomly sampled {len(examples)} examples")
        
        # Test each example
        for i, example in enumerate(examples):
            print(f"\n===== Example {i+1}/{len(examples)} for {vector_name} =====")
            print(f"Original Question: {example['original_question']}")
            print(f"Activation Strength: {example['activation']:.2f}")
            
            try:
                test_steering_on_example(
                    model, tokenizer, vector, args.layer, example,
                    args.coefficient, args.max_new_tokens, args.temperature, 
                    args.steering_token_window
                )
                
            except Exception as e:
                print(f"Error: {str(e)}")
            
            # Clean up memory
            torch.cuda.empty_cache()
        
        # Clean up
        gc.collect()
    
    # Print final summary
    print(f"\n===== Final Summary =====")
    print(f"Evaluated examples from annotated responses")
    print(f"Tested {len(steering_vectors)} steering vectors: {list(steering_vectors.keys())}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    args = parse_args()
    evaluate_steering_vectors(args)

# %% 