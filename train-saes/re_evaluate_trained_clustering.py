# %%
import numpy as np
import argparse
import json
from tqdm import tqdm
import torch
import gc
from utils.utils import print_and_flush
from utils.clustering import (
    SUPPORTED_CLUSTERING_METHODS,
    load_trained_clustering_data, predict_clusters, evaluate_clustering_scoring_metrics, save_clustering_results
)
from utils import utils

# %%

parser = argparse.ArgumentParser(description="Evaluate saved clustering models")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="Model to analyze")
parser.add_argument("--layer", type=int, default=12,
                    help="Layer to analyze")
parser.add_argument("--n_examples", type=int, default=500,
                    help="Number of examples to analyze")
parser.add_argument("--clustering_methods", type=str, nargs='+', 
                    default=list(SUPPORTED_CLUSTERING_METHODS),
                    help="Clustering methods to evaluate")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--re_compute_cluster_labels", action="store_true", default=False,
                    help="Re-compute cluster labels and centers, and save them to the existing file")
parser.add_argument("--n_autograder_examples", type=int, default=100,
                    help="Number of examples from each cluster to use for autograding")
parser.add_argument("--description_examples", type=int, default=200,
                    help="Number of examples to use for generating cluster descriptions")
args, _ = parser.parse_known_args()

# %% Get model identifier for file naming
model_id = args.model.split('/')[-1].lower()

clustering_methods = [method for method in args.clustering_methods if method in SUPPORTED_CLUSTERING_METHODS]

# %%
def re_evaluate_clustering_method(model_id, layer, method, activations, all_texts, re_compute_cluster_labels):
    """
    Evaluate a clustering method across different numbers of clusters and update the existing JSON file.
    
    Parameters:
    -----------
    model_id : str
        Model identifier
    layer : int
        Layer number
    method : str
        Clustering method name
    activations : numpy.ndarray
        Activations to evaluate on (should be normalized)
    all_texts : list
        List of texts corresponding to the activations
    re_compute_cluster_labels : bool
        Whether to recompute cluster labels
        
    Returns:
    --------
    dict
        Dictionary containing evaluation results
    """
    print_and_flush(f"\nEvaluating {method.upper()} clustering method...")
    
    # Load existing JSON file
    results_json_path = f'results/vars/{method}_results_{model_id}_layer{layer}.json'
    
    with open(results_json_path, 'r') as f:
        existing_results = json.load(f)
    print_and_flush(f"Loaded existing results from {results_json_path}")
    
    cluster_sizes = list(existing_results.get("results_by_cluster_size", {}).keys())
    if not cluster_sizes:
        print_and_flush("No clustering results to process.")
        return existing_results

    print_and_flush(f"Testing {len(cluster_sizes)} different cluster sizes: {cluster_sizes}")
    
    # Process each cluster count
    eval_results_by_cluster_size = {}
    for cluster_size in tqdm(cluster_sizes, desc=f"{method.capitalize()} evaluation"):
        print_and_flush(f"Processing {cluster_size} clusters...")
        
        # Load the saved clustering model
        clustering_data = load_trained_clustering_data(model_id, layer, cluster_size, method)
        cluster_centers = clustering_data['cluster_centers']
        
        if re_compute_cluster_labels:
            # Predict cluster labels with new activations
            cluster_labels = predict_clusters(activations, clustering_data)
            clustering_data['cluster_labels'] = cluster_labels
        else:
            cluster_labels = clustering_data['cluster_labels']
        
        # Use evaluate_clustering_scoring_metrics for comprehensive evaluation with repetitions
        scoring_results = evaluate_clustering_scoring_metrics(
            all_texts, 
            cluster_labels, 
            cluster_size, 
            activations,
            cluster_centers,
            args.model,
            args.n_autograder_examples,
            args.description_examples,
            repetitions=5  # Use 5 repetitions for evaluation
        )
        
        eval_results_by_cluster_size[cluster_size] = scoring_results

    return save_clustering_results(model_id, layer, method, eval_results_by_cluster_size)


def print_evaluation_summary(results, method):
    """
    Print a summary of evaluation results.
    
    Parameters:
    -----------
    results : dict
        Evaluation results
    method : str
        Clustering method name
    """
    
    print_and_flush("\n" + "="*50)
    print_and_flush(f"{method.upper()} EVALUATION SUMMARY")
    print_and_flush("="*50)
    print_and_flush(f"Model: {model_id.upper()}, Layer: {args.layer}")
    print_and_flush(f"Optimal clusters: {results['optimal_n_clusters']}")
    print_and_flush(f"Optimal final score: {results.get('optimal_final_score', 0.0):.4f}")
    print_and_flush(f"Optimal accuracy: {results['optimal_accuracy']:.4f}")
    print_and_flush(f"Optimal precision: {results['optimal_precision']:.4f}")
    print_and_flush(f"Optimal recall: {results['optimal_recall']:.4f}")
    print_and_flush(f"Optimal F1: {results['optimal_f1']:.4f}")
    print_and_flush(f"Optimal assignment rate: {results['optimal_assignment_rate']:.4f}")
    print_and_flush(f"Optimal confidence: {results.get('optimal_confidence', 0.0):.4f}")
    print_and_flush(f"Optimal orthogonality: {results['optimal_orthogonality']:.4f}")
    print_and_flush(f"Optimal semantic orthogonality: {results['optimal_semantic_orthogonality']:.4f}")
    
    print_and_flush("\nMetrics for all cluster sizes:")
    print_and_flush(f"{'Clusters':<10} {'Final':<8} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'Assign%':<8} {'Confid':<8} {'Orthog':<8} {'SemOrth':<8}")
    print_and_flush(f"{'-'*10} {'-'*8} {'-'*10} {'-'*11} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    cluster_range = results['cluster_range']
    confidence_scores = results['confidence_scores']
    final_scores = results['final_scores']
    semantic_orthogonality_scores = results['semantic_orthogonality_scores']
    for i, n_clusters in enumerate(cluster_range):
        prefix = "* " if n_clusters == results['optimal_n_clusters'] else "  "
        print_and_flush(f"{prefix}{n_clusters:<8} "
                f"{final_scores[i]:<8.4f} {results['accuracy_scores'][i]:<10.4f} {results['precision_scores'][i]:<11.4f} "
                f"{results['recall_scores'][i]:<8.4f} {results['f1_scores'][i]:<8.4f} "
                f"{results['assignment_rates'][i]:<8.4f} {confidence_scores[i]:<8.4f} {results['orthogonality_scores'][i]:<8.4f} {semantic_orthogonality_scores[i]:<8.4f}")

# %% Load model and process activations
print_and_flush("Loading model and processing activations...")
model, tokenizer = utils.load_model(
    model_name=args.model,
    load_in_8bit=args.load_in_8bit
)

# %% Process saved responses
all_activations, all_texts, overall_mean = utils.process_saved_responses(
    args.model, 
    args.n_examples,
    model,
    tokenizer,
    args.layer
)

del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

# %% Center activations
all_activations = [x - overall_mean for x in all_activations]
all_activations = np.stack([a.reshape(-1) for a in all_activations])
norms = np.linalg.norm(all_activations, axis=1, keepdims=True)
all_activations = all_activations / norms

# %% Main evaluation loop
print_and_flush("\n" + "="*50)
print_and_flush("EVALUATING SAVED CLUSTERING MODELS")
print_and_flush("="*50)

all_results = {}

for method in clustering_methods:
    results = re_evaluate_clustering_method(model_id, args.layer, method, all_activations, all_texts, args.re_compute_cluster_labels)
    all_results[method] = results
    print_evaluation_summary(results, method)

# %% Print overall comparison
if len(all_results) > 1:
    print_and_flush("\n" + "="*50)
    print_and_flush("OVERALL COMPARISON")
    print_and_flush("="*50)
    print_and_flush(f"{'Method':<20} {'Optimal K':<10} {'Final':<8} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1':<8} {'Assign%':<8} {'Confid':<8} {'Orthog':<8} {'SemOrth':<8}")
    print_and_flush(f"{'-'*20} {'-'*10} {'-'*8} {'-'*10} {'-'*11} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for method, results in all_results.items():
        print_and_flush(f"{method.capitalize():<20} {results['optimal_n_clusters']:<10} "
              f"{results.get('optimal_final_score', 0.0):<8.4f} {results['optimal_accuracy']:<10.4f} "
              f"{results['optimal_precision']:<11.4f} {results['optimal_recall']:<8.4f} "
              f"{results['optimal_f1']:<8.4f} {results['optimal_assignment_rate']:<8.4f} "
              f"{results.get('optimal_confidence', 0.0):<8.4f} {results['optimal_orthogonality']:<8.4f} {results.get('optimal_semantic_orthogonality', 0.0):<8.4f}")

print_and_flush("\nEvaluation complete!") 