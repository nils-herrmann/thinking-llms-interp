import json
import click
from deepseek_steering.utils import chat_batch_sync
import os
from typing import List


def get_annotated_responses_batch(thinking_processes: List[str], annotator_model_name: str, temperature: float, batch_size: int) -> List[str]:
    """Get annotated versions of thinking processes in batch using chat function"""
    prompts = [
        f"""
        Please split the following reasoning chain of an LLM into annotated parts using labels and the following format ["label"]...["end-section"]. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels.

        Available labels:
        0. initializing -> The model is rephrasing the given task and states initial thoughts.
        1. deduction -> The model is performing a deduction step based on its current approach and assumptions.
        2. adding-knowledge -> The model is enriching the current approach with recalled facts.
        3. example-testing -> The model generates examples to test its current approach.
        4. uncertainty-estimation -> The model is stating its own uncertainty.
        5. backtracking -> The model decides to change its approach.

        The reasoning chain to analyze:
        `{process}`

        Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
        """
        for process in thinking_processes
    ]
    
    return chat_batch_sync(
        prompts,
        batch_size=batch_size,
        model=annotator_model_name,
        temperature=temperature
    )

def extract_thinking_process(response_str):
    """Extract thinking process from response string"""
    try:
        # Find content between <think> tags
        think_start = response_str.index("<think>") + len("<think>")
        think_end = response_str.index("</think>")
        return response_str[think_start:think_end].strip()
    except ValueError:
        # If no think tags found, return None
        return None

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option(
    '--annotator-model-name',
    "-a",
    default="gpt-4o",
    help='Name of the model to use'
)
@click.option(
    '--temperature',
    "-t",
    default=0.01,
    help='Temperature for the annotator model'
)
@click.option(
    '--output-dir',
    default="data",
    help='Directory to save the annotated responses'
)
@click.option(
    '--batch-size',
    "-b",
    default=50,
    help='Number of concurrent requests to process'
)
def main(input_path: str, output_dir: str, annotator_model_name: str, temperature: float, batch_size: int):
    """Annotate responses in the input JSON file with reasoning labels."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input JSON
    print(f"Loading responses from {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Verify responses field exists
    if "responses" not in data:
        raise ValueError("Input JSON must contain a 'responses' field")
    
    # Extract thinking processes
    thinking_processes = []
    valid_responses = []
    
    for response in data["responses"]:
        thinking_process = extract_thinking_process(response["response_str"])
        if thinking_process:
            thinking_processes.append(thinking_process)
            valid_responses.append(response)
        else:
            print(f"Warning: No thinking process found for response {response['response_uuid']}")
    
    # Process in batches
    print(f"Processing {len(thinking_processes)} responses in batches of {batch_size}")
    annotated_texts = get_annotated_responses_batch(
        thinking_processes,
        annotator_model_name,
        temperature,
        batch_size
    )
    
    # Combine results
    annotated_responses = [
        {
            "response_uuid": response["response_uuid"],
            "task_uuid": response["task_uuid"],
            "annotated_response": annotated_text
        }
        for response, annotated_text in zip(valid_responses, annotated_texts)
    ]
    
    # Prepare output data
    output_data = {
        "model_name": data.get("model_name", "unknown"),
        "responses": annotated_responses,
        "annotator_model_name": annotator_model_name,
        "temperature": temperature
    }
    
    # Generate output filename
    input_filename = os.path.basename(input_path).replace("base_", "")
    output_filename = f"annotated_{input_filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(annotated_responses)} annotated responses to {output_path}")

if __name__ == "__main__":
    main() 