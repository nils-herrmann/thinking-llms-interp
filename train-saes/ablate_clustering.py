# %%
import numpy as np
import torch
import argparse
import json
import os
from tqdm import tqdm
from utils import utils
import gc
from utils.utils import print_and_flush
from utils.clustering import evaluate_clustering_scoring_metrics, save_clustering_results
from utils.clustering_methods import CLUSTERING_METHODS

# %%

parser = argparse.ArgumentParser(
    description="K-means clustering and autograding of neural activations"
)
parser.add_argument(
    "--model",
    type=str,
    default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    help="Model to analyze",
)
parser.add_argument("--layer", type=int, default=12, help="Layer to analyze")
parser.add_argument(
    "--n_examples", type=int, default=500, help="Number of examples to analyze"
)
parser.add_argument(
    "--clusters",
    type=str,
    default="5,10,15,20,25,30,35,40,45,50",
    help="Comma-separated list of cluster sizes to test",
)
parser.add_argument(
    "--load_in_8bit",
    action="store_true",
    default=False,
    help="Load model in 8-bit mode",
)
parser.add_argument(
    "--n_autograder_examples",
    type=int,
    default=100,
    help="Number of examples from each cluster to use for autograding",
)
parser.add_argument(
    "--description_examples",
    type=int,
    default=200,
    help="Number of examples to use for generating cluster descriptions",
)
parser.add_argument(
    "--clustering_methods",
    type=str,
    nargs="+",
    default=[
        "gmm",
        "pca_gmm",
        "spherical_kmeans",
        "pca_kmeans",
        "agglomerative",
        "pca_agglomerative",
        "sae_topk",
    ],
    help="Clustering methods to use",
)
parser.add_argument(
    "--clustering_pilot_size",
    type=int,
    default=50_000,
    help="Number of samples to use for pilot fitting with GMM",
)
parser.add_argument(
    "--clustering_pilot_n_init",
    type=int,
    default=10,
    help="Number of initializations for pilot fitting with GMM",
)
parser.add_argument(
    "--clustering_pilot_max_iter",
    type=int,
    default=100,
    help="Maximum iterations for pilot fitting with GMM",
)
parser.add_argument(
    "--clustering_full_n_init",
    type=int,
    default=1,
    help="Number of initializations for full fitting with GMM",
)
parser.add_argument(
    "--clustering_full_max_iter",
    type=int,
    default=100,
    help="Maximum iterations for full fitting with GMM",
)
args, _ = parser.parse_known_args()

# %%


def run_clustering_experiment(
    clustering_method, clustering_func, all_texts, activations, args, model_id=None
):
    """
    Run a clustering experiment using the specified clustering method.

    Parameters:
    -----------
    clustering_method : str
        Name of the clustering method
    clustering_func : function
        Function that implements the clustering algorithm
    all_texts : list
        List of texts to cluster
    activations : numpy.ndarray
        Normalized activation vectors
    args : argparse.Namespace
        Command line arguments
    model_id : str
        Model identifier for file naming

    Returns:
    --------
    dict
        Results of the clustering experiment
    """
    print_and_flush(f"\nRunning {clustering_method.upper()} clustering experiment...")

    # Define cluster range to test
    cluster_range = [int(c) for c in args.clusters.split(",")]

    print_and_flush(f"Testing {len(cluster_range)} different cluster counts...")

    # Process each cluster count
    eval_results_by_cluster_size = {}
    for n_clusters in tqdm(
        cluster_range, desc=f"{clustering_method.capitalize()} progress"
    ):
        print_and_flush(f"Processing {n_clusters} clusters...")

        # Perform clustering
        cluster_labels, cluster_centers = clustering_func(activations, n_clusters, args)

        # Evaluate clustering with repetitions
        evaluation_results = evaluate_clustering_scoring_metrics(
            all_texts,
            cluster_labels,
            n_clusters,
            activations,
            cluster_centers,
            args.model,
            args.n_autograder_examples,
            args.description_examples,
            repetitions=5,  # Use 5 repetitions for robust evaluation
        )

        eval_results_by_cluster_size[n_clusters] = evaluation_results

        # Save what we have so far
        results_data = save_clustering_results(args.model, args.layer, clustering_method, eval_results_by_cluster_size)

    # Final save just in case
    results_data = save_clustering_results(args.model, args.layer, clustering_method, eval_results_by_cluster_size)

    return results_data


# %% Load model and process activations
print_and_flush("Loading model and processing activations...")
model, tokenizer = utils.load_model(
    model_name=args.model, load_in_8bit=args.load_in_8bit
)

# %% Get model identifier for file naming
model_id = args.model.split("/")[-1].lower()

# %% Process saved responses
all_activations, all_texts, overall_mean = utils.process_saved_responses(
    args.model, args.n_examples, model, tokenizer, args.layer
)

del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

# %% Center activations
print_and_flush("Centering activations...")
all_activations = [x - overall_mean for x in all_activations]
all_activations = np.stack([a.reshape(-1) for a in all_activations])
norms = np.linalg.norm(all_activations, axis=1, keepdims=True)
all_activations = all_activations / norms

# %% Filter clustering methods based on args
clustering_methods = [
    method for method in args.clustering_methods if method in CLUSTERING_METHODS
]

# Run each clustering method
current_results = {}
for method in clustering_methods:
    try:
        clustering_func = CLUSTERING_METHODS[method]
        results = run_clustering_experiment(
            method, clustering_func, all_texts, all_activations, args, model_id
        )
        current_results[method] = results
    except Exception as e:
        print_and_flush(f"Error running {method}: {e}")
        import traceback

        print(traceback.format_exc())
