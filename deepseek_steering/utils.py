from openai import OpenAI
import dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import NNsight
from tqdm import tqdm
import gc
import asyncio
from openai import AsyncOpenAI
from typing import List

dotenv.load_dotenv(".env")

def chat(
    prompt,
    temperature=0.01,
    model="gpt-4o",
    max_tokens=5_000,
    top_p=0.90
):
    client = OpenAI(
        organization="org-E6iEJQGSfb0SNHMw6NFT1Cmi",
    )
    response = client.chat.completions.create(
        model=model,
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
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    return response.choices[0].message.content

async def _process_chat_request(
    client: AsyncOpenAI,
    prompt: str,
    temperature: float = 0.01,
    model: str = "gpt-4o",
    max_tokens: int = 5_000,
    top_p: float = 0.90
) -> str:
    """Process a single chat request asynchronously"""
    response = await client.chat.completions.create(
        model=model,
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
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    return response.choices[0].message.content

async def chat_batch(
    prompts: List[str],
    batch_size: int = 50,
    temperature: float = 0.01,
    model: str = "gpt-4o",
    max_tokens: int = 5_000,
    top_p: float = 0.90
) -> List[str]:
    """
    Process multiple chat requests in batches asynchronously.
    
    Args:
        prompts: List of prompts to process
        batch_size: Number of concurrent requests
        temperature: Temperature for generation
        model: Model to use
        max_tokens: Maximum tokens for generation
        top_p: Top p for generation
        
    Returns:
        List of responses in the same order as prompts
    """
    client = AsyncOpenAI(
        organization="org-E6iEJQGSfb0SNHMw6NFT1Cmi",
    )
    
    async def process_batch(batch_prompts: List[str]) -> List[str]:
        tasks = [
            _process_chat_request(
                client=client,
                prompt=prompt,
                temperature=temperature,
                model=model,
                max_tokens=max_tokens,
                top_p=top_p
            )
            for prompt in batch_prompts
        ]
        return await asyncio.gather(*tasks)

    results = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = await process_batch(batch)
            results.extend(batch_results)
            pbar.update(1)
    
    return results


def chat_batch_sync(
    prompts: List[str],
    batch_size: int = 50,
    **kwargs
) -> List[str]:
    """Synchronous wrapper for chat_batch"""
    return asyncio.run(chat_batch(prompts, batch_size, **kwargs))

def load_model_and_vectors(compute_features=True, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    """
    Load model, tokenizer and mean vectors. Optionally compute feature vectors.
    
    Args:
        compute_features (bool): If True, compute and return feature vectors by subtracting overall mean
        model_name (str): Name/path of the model to load
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = NNsight(model).to("cuda")
    
    # Get model identifier for file naming
    model_id = model_name.split('/')[-1].lower()
    mean_vectors_dict = torch.load(f"data/mean_vectors_{model_id}.pt")
    
    if compute_features:
        # Compute feature vectors by subtracting overall mean
        overall_mean = mean_vectors_dict['overall']['mean']
        feature_vectors = {}
        
        for label in mean_vectors_dict:
            if label != 'overall':
                feature_vectors[label] = mean_vectors_dict[label]['mean'] - overall_mean
        
        return model, tokenizer, feature_vectors
    
    return model, tokenizer, mean_vectors_dict

def custom_generate_with_projection_removal(model, tokenizer, input_ids, max_new_tokens, label, feature_vectors, layers=[10], coefficient=0.1, steer_positive=False, show_progress=True):
    """
    Generate text while removing or adding projections of specific features.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        input_ids: Input token ids
        max_new_tokens: Maximum number of tokens to generate
        label: The label to steer towards/away from
        feature_vectors: Dictionary of feature vectors
        steer_positive: If True, steer towards the label, if False steer away
        show_progress: If True, show progress bar
    """
    generated_ids = input_ids.clone().cpu()
    if label in feature_vectors:
        feature_vector = feature_vectors[label].to("cuda").to(torch.bfloat16)
    else:
        feature_vector = None
    
    iterator = range(max_new_tokens)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating response")
            
    for k in iterator:
        input_chunk = generated_ids.to("cuda")
        
        with torch.no_grad():
            with model.trace(input_chunk) as trace:
                # First run the model normally to get hidden states
                outputs = model.lm_head.output.save()
                
                if feature_vector is not None:
                    for layer_idx in layers:
                        
                        if steer_positive:
                            expanded_feature = feature_vector[layer_idx].unsqueeze(0).unsqueeze(0).expand(1, input_chunk.size(1)-1, -1)
                            model.model.layers[layer_idx].output[0][:, 1:] += coefficient * expanded_feature
                        else:
                            expanded_feature = feature_vector[layer_idx].unsqueeze(0).unsqueeze(0).expand(1, input_chunk.size(1)-1, -1)
                            model.model.layers[layer_idx].output[0][:, 1:] -= coefficient * expanded_feature
        
        next_token = outputs[:, -1, :].argmax(dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).cpu()], dim=1)

        del trace, outputs, next_token, input_chunk
       
        torch.cuda.empty_cache()
        if k % 50 == 0:
            gc.collect()
    
    gc.collect()
    return generated_ids.cpu()