# %%
import json

# %%
total_sentences = 517662.0

layers = [6,10,14,18, 22, 26]
n_cluster_range = [10,20,30,40,50]

best_final_score = 0.0
best_layer_and_cluster_size = None

final_scores_by_layer_n_clusters = {} # (layer, n_clusters) -> final_score
avg_f1_by_layer_n_clusters = {} # (layer, n_clusters) -> avg_f1

for layer in layers:
    with open(f'../train-saes/results/vars/sae_topk_results_deepseek-r1-distill-llama-8b_layer{layer}.json') as f:
        results = json.load(f)

    clusters_detailed_results = results['detailed_results']
    # n_cluster_range = [int(n) for n in clusters_detailed_results.keys()]
    # n_cluster_range.sort()

    for n_clusters in n_cluster_range:
        print(f"=== {n_clusters} clusters ===")
        cluster_data = clusters_detailed_results[str(n_clusters)] # dict_keys(['accuracy', 'categories', 'orthogonality', 'assigned_fraction', 'category_counts', 'detailed_results'])
        accuracy = cluster_data['accuracy']
        orthogonality = cluster_data['orthogonality']
        completeness = cluster_data['assigned_fraction']
        
        print(f"Accuracy: {accuracy}")
        # print(f"Categories: {cluster_data['categories']}")
        print(f"Orthogonality: {orthogonality}")
        print(f"Completeness: {completeness}")
        # print(f"Category Counts: {cluster_data['category_counts']}")
        
        cluster_detailed_results = cluster_data['detailed_results'] # dict_keys(['title', 'description', 'size', 'precision', 'recall', 'accuracy', 'f1', 'examples'])
        avg_f1 = 0.0
        for cluster_id, data in cluster_detailed_results.items():
            cluster_size = data['size']
            cluster_percentage = cluster_size/total_sentences*100
            cluster_title = data['title']
            # print(f"Cluster {cluster_id}: {cluster_size} examples ({cluster_percentage:.2f}%) - {cluster_title}")
            avg_f1 += data['f1']
            # Print the first 10 sentences in the cluster
            # for sentence in data['examples'][:10]:
            #     print(f"  - {sentence}")
            # print()

        avg_f1 /= len(cluster_detailed_results)
        avg_f1_by_layer_n_clusters[(layer, n_clusters)] = avg_f1
        print(f"Average F1: {avg_f1}")

        cluster_score = (avg_f1 + completeness + orthogonality) / 3
        print(f"Cluster Score: {cluster_score}")

        final_scores_by_layer_n_clusters[(layer, n_clusters)] = cluster_score

print("=== Clusters sorted by Final score ===")
# worst first
sorted_clusters = sorted(final_scores_by_layer_n_clusters.items(), key=lambda x: x[1], reverse=False)
for (layer, n_clusters), cluster_score in sorted_clusters:
    print(f"Layer {layer}, Cluster {n_clusters}: {cluster_score}")

print("=== Clusters sorted by Avg F1 ===")
sorted_clusters = sorted(avg_f1_by_layer_n_clusters.items(), key=lambda x: x[1], reverse=False)
for (layer, n_clusters), avg_f1 in sorted_clusters:
    print(f"Layer {layer}, Cluster {n_clusters}: {avg_f1}")


# %%