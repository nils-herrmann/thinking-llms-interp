# %%

import json

# %%

with open("../steering_evaluation_results_deepseek-r1-distill-llama-8b.json", "r") as f:
    results = json.load(f)

# %%

behavior_pattern = ["initializing", "deduction", "adding-knowledge", "example-testing", "uncertainty-estimation", "backtracking"]

# %%

for behavior in behavior_pattern:
    if behavior == "initializing":
        continue

    behavior_results = results[behavior]
    
    # Track the example with max change and its change value
    max_change_example = None
    max_total_change = 0
    
    for result in behavior_results:
        # Get fractions for each steering condition
        orig_frac = result["original"]["label_fractions"][behavior]
        pos_frac = result["positive"]["label_fractions"][behavior]
        neg_frac = result["negative"]["label_fractions"][behavior]
        
        # Calculate absolute changes
        pos_change = pos_frac - orig_frac
        neg_change = orig_frac - neg_frac
        total_change = pos_change + neg_change
        
        # Update if this example has larger total change
        if total_change > max_total_change:
            max_total_change = total_change
            max_change_example = result
    
    print(f"\n=== {behavior} ===")
    print(f"Original fraction: {max_change_example['original']['label_fractions'][behavior]:.3f}")
    print(f"Positive fraction: {max_change_example['positive']['label_fractions'][behavior]:.3f}")
    print(f"Negative fraction: {max_change_example['negative']['label_fractions'][behavior]:.3f}")
    print("Original response annotated:")
    print(max_change_example['original']['annotated_response'])
    print("\nPositive response annotated:")
    print(max_change_example['positive']['annotated_response'])
    print("\nNegative response annotated:")
    print(max_change_example['negative']['annotated_response'])
    print("\n")

# %%
