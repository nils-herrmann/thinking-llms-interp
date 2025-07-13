#!/usr/bin/env python3
"""
Script to merge partial optimized vector .pt files into a single comprehensive file.

Each partial file contains vectors for multiple categories, but we only want to keep
the categories corresponding to the indices specified in the filename.
"""

import os
import torch
import glob
import re
import json
from collections import OrderedDict

def load_category_mapping():
    """Load the index-to-category mapping from the hyperparameters file"""
    hp_file = "results/vars/steering_vector_hyperparams.json"
    
    if not os.path.exists(hp_file):
        raise FileNotFoundError(f"Hyperparameters file not found at {hp_file}")
    
    with open(hp_file, 'r') as f:
        all_hyperparams = json.load(f)
    
    # Get the mapping for llama-3.1-8b
    if "llama-3.1-8b" not in all_hyperparams:
        raise ValueError("No hyperparameters found for llama-3.1-8b")
    
    model_hyperparams = all_hyperparams["llama-3.1-8b"]
    
    # Create index-to-category mapping
    index_to_category = {}
    for idx_str, info in model_hyperparams.items():
        idx = int(idx_str)
        category = info["category"]
        index_to_category[idx] = category
    
    # Create ordered list of categories by index
    max_idx = max(index_to_category.keys())
    all_categories = []
    for i in range(max_idx + 1):
        if i in index_to_category:
            all_categories.append(index_to_category[i])
        else:
            all_categories.append(None)  # Missing category
    
    return all_categories, index_to_category

def main():
    # Load category mapping from hyperparameters file
    print("Loading category mapping from hyperparameters file...")
    all_categories, index_to_category = load_category_mapping()
    
    print(f"Found {len([c for c in all_categories if c])} categories:")
    for idx, category in enumerate(all_categories):
        if category:
            print(f"  [{idx}] {category}")
        else:
            print(f"  [{idx}] <missing>")
    
    # Define paths
    partial_files_pattern = "results/vars/optimized_vectors_llama-3.1-8b_idx_*.pt"
    output_file = "results/vars/optimized_vectors_llama-3.1-8b.pt"
    
    # Find all partial files
    partial_files = glob.glob(partial_files_pattern)
    
    if not partial_files:
        print(f"No partial files found matching pattern: {partial_files_pattern}")
        return
    
    # Sort files by the index range for consistent processing
    def extract_indices(filename):
        # Extract indices from filename like "optimized_vectors_llama-3.1-8b_idx_0_1.pt"
        match = re.search(r'idx_(\d+)_(\d+)\.pt$', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        raise ValueError(f"Could not extract indices from filename: {filename}")
    
    partial_files.sort(key=extract_indices)
    
    print(f"\nFound {len(partial_files)} partial files to merge:")
    for file in partial_files:
        start_idx, end_idx = extract_indices(file)
        print(f"  - {file} (indices {start_idx}-{end_idx})")
    
    # Merge only the relevant categories from each file
    merged_vectors = OrderedDict()
    
    for file_path in partial_files:
        print(f"\nProcessing {file_path}...")
        
        try:
            # Extract the indices for this file
            start_idx, end_idx = extract_indices(file_path)
            expected_indices = list(range(start_idx, end_idx + 1))
            
            # Get the expected category names for these indices
            expected_categories = []
            for idx in expected_indices:
                if idx in index_to_category:
                    expected_categories.append(index_to_category[idx])
                else:
                    print(f"  Warning: Index {idx} not found in hyperparameters")
            
            print(f"  Expected categories for indices {expected_indices}: {expected_categories}")
            
            # Load the partial file
            partial_vectors = torch.load(file_path, map_location='cpu')
            
            if not isinstance(partial_vectors, dict):
                print(f"  Warning: {file_path} does not contain a dictionary. Skipping.")
                continue
            
            print(f"  File contains {len(partial_vectors)} categories total")
            
            # Only keep the categories that match the expected indices
            kept_categories = []
            for category_name in expected_categories:
                if category_name in partial_vectors:
                    if category_name in merged_vectors:
                        print(f"  Warning: Category '{category_name}' already exists in merged vectors. Overwriting.")
                    
                    merged_vectors[category_name] = partial_vectors[category_name]
                    kept_categories.append(category_name)
                    print(f"    ✓ Kept category: {category_name}")
                else:
                    print(f"    ✗ Missing expected category: {category_name}")
            
            print(f"  Kept {len(kept_categories)} out of {len(expected_categories)} expected categories")
        
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
    
    if not merged_vectors:
        print("No vectors were successfully loaded. Exiting.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the merged vectors
    print(f"\nSaving merged vectors to {output_file}")
    print(f"Total categories: {len(merged_vectors)}")
    print("Categories included:")
    for i, category_name in enumerate(merged_vectors.keys()):
        # Find the index of this category in the mapping
        try:
            original_idx = next(idx for idx, cat in index_to_category.items() if cat == category_name)
            print(f"  [{original_idx}] {category_name}")
        except StopIteration:
            print(f"  [?] {category_name}")
    
    torch.save(dict(merged_vectors), output_file)
    print(f"\nSuccessfully merged {len(merged_vectors)} categories into {output_file}")
    
    # Verify the merged file
    print("\nVerifying merged file...")
    try:
        loaded_vectors = torch.load(output_file, map_location='cpu')
        print(f"Verification successful: {len(loaded_vectors)} categories loaded")
        
        # Print some statistics
        vector_shapes = {}
        for category_name, vector in loaded_vectors.items():
            shape = tuple(vector.shape)
            if shape not in vector_shapes:
                vector_shapes[shape] = 0
            vector_shapes[shape] += 1
        
        print("Vector shape distribution:")
        for shape, count in vector_shapes.items():
            print(f"  {shape}: {count} vectors")
            
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    main() 