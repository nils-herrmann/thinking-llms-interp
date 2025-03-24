from openai import OpenAI
import dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel
from tqdm import tqdm
import gc
import time
dotenv.load_dotenv(".env")

def chat(prompt, image=None):

    # try 3 times with 3 second sleep between attempts
    for _ in range(3):
        try:
            client = OpenAI(
                organization="org-E6iEJQGSfb0SNHMw6NFT1Cmi",
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                temperature=1e-19
            )
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(20)

    return None

def load_model_and_vectors(compute_features=True, return_steering_vector_set=False, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", base_model_name=None):
    """
    Load model, tokenizer and mean vectors. Optionally compute feature vectors.
    
    Args:
        compute_features (bool): If True, compute and return feature vectors by subtracting overall mean
        model_name (str): Name/path of the model to load
    """
    model = LanguageModel(model_name, dispatch=True, device_map="auto", torch_dtype=torch.bfloat16)
    
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.do_sample=False
    
    tokenizer = model.tokenizer

    if base_model_name is not None:
        base_model = LanguageModel(base_model_name, dispatch=True, device_map="auto", torch_dtype=torch.bfloat16)
    
        base_model.generation_config.temperature=None
        base_model.generation_config.top_p=None
        base_model.generation_config.do_sample=False
        
        base_tokenizer = base_model.tokenizer
    
    # Get model identifier for file naming
    model_id = model_name.split('/')[-1].lower()
    mean_vectors_dict = torch.load(f"data/mean_vectors_{model_id}.pt")
    
    if compute_features:
        # Compute feature vectors by subtracting overall mean
        feature_vectors = {}
        steering_vector_set = {label: {} for label in mean_vectors_dict}
        
        for label in mean_vectors_dict:
            all_directions = []

            if label != 'overall':

                for other_label in ["deduction", "adding-knowledge", "backtracking", "example-testing", "uncertainty-estimation"]:
                    if other_label != label:
                        all_directions.append(mean_vectors_dict[other_label]['mean'])
                        steering_vector_set[label][other_label] = mean_vectors_dict[label]['mean'] - mean_vectors_dict[other_label]['mean']

                feature_vectors[label] = mean_vectors_dict[label]['mean'] - torch.stack(all_directions).mean(0)
            
            else:
                feature_vectors["overall"] = mean_vectors_dict["overall"]['mean']

            for label in feature_vectors:
                feature_vectors[label] = feature_vectors[label] * (feature_vectors["overall"].norm(dim=-1, keepdim=True) / feature_vectors[label].norm(dim=-1, keepdim=True))

    if return_steering_vector_set and compute_features:
        feature_vectors["steering_vector_set"] = steering_vector_set
    elif return_steering_vector_set and not compute_features:
        mean_vectors_dict["steering_vector_set"] = steering_vector_set

    if base_model_name is not None and compute_features:
        return model, tokenizer, base_model, base_tokenizer, feature_vectors
    elif base_model_name is not None and not compute_features:
        return model, tokenizer, base_model, base_tokenizer, mean_vectors_dict
    elif base_model_name is None and compute_features:
        return model, tokenizer, feature_vectors
    else:
        return model, tokenizer, mean_vectors_dict

def custom_generate_with_projection_removal(model, tokenizer, input_ids, max_new_tokens, label, feature_vectors, steering_config, steer_positive=False):
    """
    Generate text while removing or adding projections of specific features.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        input_ids: Input token ids
        max_new_tokens: Maximum number of tokens to generate
        label: The label to steer towards/away from
        feature_vectors: Dictionary of feature vectors containing steering_vector_set
        steer_positive: If True, steer towards the label, if False steer away
    """
    model_layers = model.model.layers

    with model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    ) as tracer:
        # Apply .all() to model to ensure interventions work across all generations
        model_layers.all()

        if feature_vectors is not None:       
            vector_layer = steering_config[label]["vector_layer"]
            pos_layers = steering_config[label]["pos_layers"]
            neg_layers = steering_config[label]["neg_layers"]
            coefficient = steering_config[label]["pos_coefficient"] if steer_positive else steering_config[label]["neg_coefficient"]
     

            if steer_positive:
                feature_vector = feature_vectors[label][vector_layer].to("cuda").to(torch.bfloat16)
                for layer_idx in pos_layers:         
                    model.model.layers[layer_idx].output[0][:, :] += coefficient * feature_vector.unsqueeze(0).unsqueeze(0)
            else:
                feature_vector = feature_vectors[label][vector_layer].to("cuda").to(torch.bfloat16)
                for layer_idx in neg_layers:         
                    model.model.layers[layer_idx].output[0][:, :] -= coefficient * feature_vector.unsqueeze(0).unsqueeze(0)
        
        outputs = model.generator.output.save()
                    
    return outputs

steering_config = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "backtracking": {"vector_layer": 13, "pos_layers": [13], "neg_layers": [13], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 14, "pos_layers": [14], "neg_layers": [14], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 13, "pos_layers": [13], "neg_layers": [13], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 12, "pos_layers": [12], "neg_layers": [12], "pos_coefficient": 1, "neg_coefficient": 1}
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "backtracking": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1},
        "adding-knowledge": {"vector_layer": 27, "pos_layers": [27], "neg_layers": [27], "pos_coefficient": 1, "neg_coefficient": 1},
        "example-testing": {"vector_layer": 31, "pos_layers": [31], "neg_layers": [31], "pos_coefficient": 1, "neg_coefficient": 1},
        "uncertainty-estimation": {"vector_layer": 29, "pos_layers": [29], "neg_layers": [29], "pos_coefficient": 1, "neg_coefficient": 1}
    }
}

def custom_hybrid_generate(
        thinking_model, 
        base_model,
        base_tokenizer,
        input_ids, 
        max_new_tokens, 
        feature_vectors, 
        steering_config,
        coefficient=0.1, 
        steer_positive=False, 
        warmup=0,
        show_progress=True,
        color_output=False):

    base_generated_ids = input_ids.clone().cpu()
    
    # Convert feature vectors to cuda and bfloat16
    feature_vectors_cuda = {
        label: (feature_vectors[label][steering_config[label]["vector_layer"]].to("cuda").to(torch.bfloat16),
                steering_config[label]["vector_layer"])
        for label in ["backtracking", "uncertainty-estimation", "adding-knowledge", "example-testing"]
    }
    
    # Color codes for different labels
    label_colors = {
        "backtracking": "\033[92m",  # Green
        "uncertainty-estimation": "\033[94m",  # Blue
        "adding-knowledge": "\033[93m",  # Yellow
        "example-testing": "\033[95m",  # Magenta
    }
    
    iterator = range(max_new_tokens)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating response")
    
    # Track model usage and forced tokens
    base_model_tokens = 0
    thinking_model_tokens = 0
    forced_tokens = {}  # Dictionary to track token frequencies
    forced_positions = []  # List to track which positions were forced
    forced_labels = []  # List to track which label forced each token
    
    # Store projections
    projection_buffer = []
    label_buffer = []  # Store which label had max projection
            
    for k in iterator:
        base_input_chunk = base_generated_ids.to("cuda")
        
        # Run thinking model with base model's input
        with torch.no_grad():
            with thinking_model.trace(base_input_chunk) as trace:
                thinking_outputs = thinking_model.lm_head.output.save()
                
                # Get projections for each label in its respective layer
                projections = []
                for label, (vector, layer) in feature_vectors_cuda.items():
                    # Get hidden states for this layer
                    hidden_states = thinking_model.model.layers[layer].output[0][:, -1:, :]  # Only last token
                    
                    # Compute projection
                    projection = torch.einsum('bth,h->bt', hidden_states, vector).save()
                    projections.append((label, projection.save()))
        
        # Calculate max projection and label outside trace context
        max_projection = 0
        max_label = None
        for label, projection in projections:
            if abs(projection.item()) > max_projection:
                max_projection = abs(projection.item())
                max_label = label
        
        # Store projection and label in buffer
        projection_buffer.append(max_projection)
        label_buffer.append(max_label)
        
        # Now run base model
        with torch.no_grad():
            with base_model.trace(base_input_chunk) as trace:
                base_outputs = base_model.lm_head.output.save()
        
        # Get next tokens from both models
        base_next_token = base_outputs[:, -1, :].argmax(dim=-1)
        thinking_next_token = thinking_outputs[:, -1, :].argmax(dim=-1)

        # During warmup, use thinking model's predictions
        if k < warmup:
            next_token = thinking_next_token
            thinking_model_tokens += 1
            forced_positions.append(True)
            forced_labels.append("warmup")
        else:
            # Only force thinking model token if:
            # 1. Projection is above coefficient
            # 2. Thinking model predicts different token than base model
            if max_projection > coefficient and thinking_next_token != base_next_token:
                next_token = thinking_next_token
                thinking_model_tokens += 1
                # Track forced token
                token_text = base_tokenizer.decode(next_token[0])
                forced_tokens[token_text] = forced_tokens.get(token_text, 0) + 1
                forced_positions.append(True)
                forced_labels.append(max_label)
            else:
                next_token = base_next_token
                base_model_tokens += 1
                forced_positions.append(False)
                forced_labels.append(None)

        # Append token to sequence
        base_generated_ids = torch.cat([base_generated_ids, next_token.unsqueeze(0).cpu()], dim=1)
        
        # Check for end of sequence
        if next_token.item() == base_tokenizer.eos_token_id:
            break

        del trace, thinking_outputs, base_outputs, base_next_token, thinking_next_token, base_input_chunk
       
        torch.cuda.empty_cache()
        if k % 50 == 0:
            gc.collect()
    
    gc.collect()
    
    if color_output:
        # Print model usage statistics
        total_tokens = base_model_tokens + thinking_model_tokens
        print(f"\nModel Usage Statistics:")
        print(f"Base model tokens: {base_model_tokens} ({base_model_tokens/total_tokens*100:.1f}%)")
        print(f"Thinking model tokens: {thinking_model_tokens} ({thinking_model_tokens/total_tokens*100:.1f}%)")
        
        # Print top forced tokens
        print("\nTop 10 Most Frequent Forced Tokens:")
        sorted_tokens = sorted(forced_tokens.items(), key=lambda x: x[1], reverse=True)
        for token, freq in sorted_tokens[:10]:
            print(f"'{token}': {freq} times")
        
        # Print colored output
        print("\nColored Output (Colors indicate which label forced the token):")
        base_text = base_tokenizer.decode(base_generated_ids[0], skip_special_tokens=True)
        
        # Split into tokens and color them, skipping input tokens
        base_tokens = base_tokenizer.encode(base_text)
        input_length = len(base_tokenizer.encode(base_tokenizer.decode(input_ids[0], skip_special_tokens=True)))
        colored_base = []
        
        for i, token in enumerate(base_tokens):
            if i < input_length:
                colored_base.append(base_tokenizer.decode(token))
            else:
                token_idx = i - input_length
                if token_idx < len(forced_positions) and forced_positions[token_idx]:
                    label = forced_labels[token_idx]
                    if label == "warmup":
                        colored_base.append(f"\033[90m{base_tokenizer.decode(token)}\033[0m")  # Gray for warmup
                    else:
                        colored_base.append(f"{label_colors[label]}{base_tokenizer.decode(token)}\033[0m")
                else:
                    colored_base.append(base_tokenizer.decode(token))
        
        print("Base (with forced tokens colored by label):")
        print(base_tokenizer.convert_tokens_to_string(colored_base))
        
        # Print color legend
        print("\nColor Legend:")
        print("Gray: Warmup tokens")
        for label, color in label_colors.items():
            print(f"{color}{label}\033[0m")
    
    return base_generated_ids.cpu(), forced_positions, forced_labels, forced_tokens
