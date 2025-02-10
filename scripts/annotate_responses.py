import json
import click
from tqdm import tqdm
from deepseek_steering.utils import chat
import os

def get_annotated_response(thinking_process, annotator_model_name, temperature):
    """Get annotated version of thinking process using chat function"""
    annotated_response = chat(f"""
    Please split the following reasoning chain of an LLM into annotated parts using labels and the following format ["label"]...["end-section"]. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels.

    Available labels:
    0. initializing -> The model is rephrasing the given task and states initial thoughts.
    1. deduction -> The model is performing a deduction step based on its current approach and assumptions.
    2. adding-knowledge -> The model is enriching the current approach with recalled facts.
    3. example-testing -> The model generates examples to test its current approach.
    4. uncertainty-estimation -> The model is stating its own uncertainty.
    5. backtracking -> The model decides to change its approach.

    The reasoning chain to analyze:
    `{thinking_process}`

    Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
    """,
        model=annotator_model_name,
        temperature=temperature
    )
    
    return annotated_response

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
def main(input_path: str, output_dir: str, annotator_model_name: str, temperature: float):
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
    
    # Process each response
    annotated_responses = []
    for response in tqdm(data["responses"], desc="Annotating responses"):
        # Extract thinking process
        thinking_process = extract_thinking_process(response["response_str"])
        
        if thinking_process:
            # Get annotated version
            annotated_response = get_annotated_response(thinking_process, annotator_model_name, temperature)
            
            # Add to results
            annotated_responses.append({
                "response_uuid": response["response_uuid"],
                "task_uuid": response["task_uuid"],
                "annotated_response": annotated_response
            })
        else:
            print(f"Warning: No thinking process found for response {response['response_uuid']}")
    
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