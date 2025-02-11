# %%
import json
from glob import glob

# Load tasks
with open('../data/tasks.json', 'r') as f:
    tasks = json.load(f)

# Load all response files
annotated_responses_files = glob('../data/annotated_responses_*.json')
base_responses_files = glob('../data/base_responses_*.json')

# Load all responses into dictionaries with their model names as keys
annotated_responses = {}
base_responses = {}

for file_path in annotated_responses_files:
    with open(file_path, 'r') as f:
        data = json.load(f)
        annotated_responses[data['model_name']] = data

for file_path in base_responses_files:
    with open(file_path, 'r') as f:
        data = json.load(f)
        base_responses[data['model_name']] = data['responses']

# %%
# Filter out system-thinking tasks and their corresponding responses
system_thinking_tasks = [task for task in tasks if task['task_category'] == 'systems-thinking']
system_thinking_task_ids = {task['task_uuid'] for task in system_thinking_tasks}

# Filter tasks
filtered_tasks = [task for task in tasks 
                 if task['task_uuid'] not in system_thinking_task_ids]

# %%
# Filter and save annotated responses
for model_name, response_data in annotated_responses.items():
    model_id = model_name.split('/')[-1].lower()
    # Preserve all fields from the original response_data
    filtered_responses_data = response_data.copy()
    filtered_responses_data['responses'] = [resp for resp in response_data['responses'] 
                                          if resp['task_uuid'] not in system_thinking_task_ids]
    
    output_path = f'../data/annotated_responses_{model_id}.json'
    with open(output_path, 'w') as f:
        json.dump(filtered_responses_data, f, indent=2)

# Filter and save base responses
for model_name, responses_list in base_responses.items():
    model_id = model_name.split('/')[-1].lower()  # Add missing model_id definition
    filtered_responses_data = {
        "model_name": model_name,
        "responses": [resp for resp in responses_list 
                     if resp['task_uuid'] not in system_thinking_task_ids]
    }
    
    output_path = f'../data/base_responses_{model_id}.json'
    with open(output_path, 'w') as f:
        json.dump(filtered_responses_data, f, indent=2)

# Save filtered tasks
with open('../data/tasks.json', 'w') as f:
    json.dump(filtered_tasks, f, indent=2)

# %%
# Print statistics
print(f"Original number of tasks: {len(tasks)}")
print(f"Number of system-thinking tasks removed: {len(system_thinking_tasks)}")
print(f"Remaining tasks: {len(filtered_tasks)}")
print("\nAnnotated Response Files:")
for model_name, response_data in annotated_responses.items():
    original_count = len(response_data['responses'])
    filtered_count = len([r for r in response_data['responses'] 
                         if r['task_uuid'] not in system_thinking_task_ids])
    print(f"{model_name}:")
    print(f"  Original responses: {original_count}")
    print(f"  Remaining responses: {filtered_count}")

print("\nBase Response Files:")
for model_name, responses_list in base_responses.items():
    original_count = len(responses_list)
    filtered_count = len([r for r in responses_list 
                         if r['task_uuid'] not in system_thinking_task_ids])
    print(f"{model_name}:")
    print(f"  Original responses: {original_count}")
    print(f"  Remaining responses: {filtered_count}")