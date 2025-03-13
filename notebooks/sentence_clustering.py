# %%
from sentence_transformers import SentenceTransformer
import json
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import re
import random
from datasets import load_dataset
# %%
model = SentenceTransformer("intfloat/multilingual-e5-large-instruct", trust_remote_code=True)

# %%
dataset = load_dataset("ServiceNow-AI/R1-Distill-SFT", "v1", split="train", streaming=True).shuffle(seed=42)

num_sentences = 10000

sentences = []
for response in dataset:
    if len(sentences) >= num_sentences:
        break
    response_str = response["reannotated_assistant_content"]
    response_str = response_str.replace("<think>", "").split("</think>")[0]
    response_str = response_str.replace("\n", " ")
    # Replace multiple spaces with a single space
    while "  " in response_str:
        response_str = response_str.replace("  ", " ")
    
    # Split on common sentence-ending punctuation
    sentence_splits = re.split(r'(?<=[.!?])\s+', response_str)
    
    # Clean and filter sentences
    for sentence in sentence_splits:
        sentence = sentence.strip()
        if sentence and len(sentence) > 3:  # Only keep non-empty sentences with meaningful content
            sentences.append(sentence)

random.shuffle(sentences)
print(len(sentences))
sentences = sentences[:10000]

# %%
sentence_embeddings = []
for sentence in tqdm(sentences):
    sentence_embeddings.append(model.encode(sentence))

# %% %%
sentence_embeddings = torch.tensor(sentence_embeddings)
sentence_embeddings_np = sentence_embeddings.numpy()

# Sweep over different k values
k_values = range(1, 16)
inertia_values = []

for k in tqdm(k_values, desc="Running K-means"):
    # Use cosine distance metric for K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='k-means++')
    # Normalize the vectors for cosine similarity
    normalized_embeddings = sentence_embeddings_np / np.linalg.norm(sentence_embeddings_np, axis=1, keepdims=True)
    kmeans.fit(normalized_embeddings)
    inertia_values.append(kmeans.inertia_)

# %% Compute the elbow point using second derivatives
def find_elbow_point(k_values, inertia_values):
    # Convert to numpy arrays for easier computation
    k_array = np.array(k_values)
    inertia_array = np.array(inertia_values)
    
    # Create line from first to last point
    first_point = np.array([k_array[0], inertia_array[0]])
    last_point = np.array([k_array[-1], inertia_array[-1]])
    line_vec = last_point - first_point
    
    # Normalize the line vector
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    # Compute perpendicular distances from each point to the line
    vec_from_first = np.array([(k, inertia) for k, inertia in zip(k_array, inertia_array)]) - first_point
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_on_line = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_on_line
    distances = np.sqrt(np.sum(vec_to_line**2, axis=1))
    
    # Find index of maximum distance
    elbow_index = np.argmax(distances)
    elbow_k = k_values[elbow_index]
    elbow_inertia = inertia_values[elbow_index]
    
    return elbow_k, elbow_inertia, elbow_index

# Find the elbow point
elbow_k, elbow_inertia, elbow_index = find_elbow_point(k_values, inertia_values)

# Plot the inertia values and mark the elbow point
plt.figure(figsize=(12, 8))
plt.plot(k_values, inertia_values, 'bo-', linewidth=2, markersize=8)
plt.plot(elbow_k, elbow_inertia, 'ro', markersize=12)
plt.annotate(f'Elbow point (k={elbow_k})', 
             xy=(elbow_k, elbow_inertia),
             xytext=(elbow_k+5, elbow_inertia+0.01),
             arrowprops=dict(facecolor='red', shrink=0.05),
             fontsize=12)

plt.title('Elbow Method for Optimal k', fontsize=16)
plt.xlabel('Number of clusters (k)', fontsize=14)
plt.ylabel('Inertia', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Print the optimal k value
print(f"Optimal number of clusters (k) based on elbow method: {elbow_k}")

# %% Run K-means with the optimal number of clusters
print(f"\nRunning K-means with k={elbow_k}...")
kmeans_optimal = KMeans(n_clusters=elbow_k, random_state=42, n_init=10, init='k-means++')
normalized_embeddings = sentence_embeddings_np / np.linalg.norm(sentence_embeddings_np, axis=1, keepdims=True)
kmeans_optimal.fit(normalized_embeddings)

# Get cluster assignments for each sentence
cluster_labels = kmeans_optimal.labels_

# %% Function to get top examples from each cluster
def get_top_examples_per_cluster(sentences, embeddings, labels, centroids, top_m=5):
    # Dictionary to store results
    clusters = {i: [] for i in range(len(centroids))}
    
    # Normalize centroids for cosine similarity
    normalized_centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    
    # Calculate distance to centroid for each sentence
    for idx, (sentence, embedding, label) in enumerate(zip(sentences, embeddings, labels)):
        # Normalize embedding for cosine similarity
        norm_embedding = embedding / np.linalg.norm(embedding)
        # Calculate cosine similarity (negative distance)
        similarity = np.dot(norm_embedding, normalized_centroids[label])
        clusters[label].append((sentence, similarity, idx))
    
    # Sort examples in each cluster by similarity to centroid (descending)
    for label in clusters:
        clusters[label].sort(key=lambda x: x[1], reverse=True)
    
    return clusters

# %% Get the top examples for each cluster
top_m = 30  # Number of examples to show per cluster
cluster_examples = get_top_examples_per_cluster(
    sentences,
    normalized_embeddings, 
    cluster_labels, 
    kmeans_optimal.cluster_centers_,
    top_m
)

# %% Print the top examples for each cluster
print(f"\nTop {top_m} examples from each of the {elbow_k} clusters:")
for cluster_id in sorted(cluster_examples.keys()):
    print(f"\nCluster {cluster_id}:")
    examples = cluster_examples[cluster_id][:top_m]
    for i, (sentence, similarity, idx) in enumerate(examples, 1):
        print(f"  {i}. {sentence}")
    
    # Print cluster size
    cluster_size = np.sum(cluster_labels == cluster_id)
    print(f"  Total sentences in cluster: {cluster_size}")

# %%
