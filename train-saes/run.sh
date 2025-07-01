MIN_CLUSTERS=4
MAX_CLUSTERS=20
N_EXAMPLES=100000  # all responses

# for LAYER in 4 8 12 16 20 24; do
#     python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --layer $LAYER --min_clusters $MIN_CLUSTERS --max_clusters $MAX_CLUSTERS
# done

python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --layer 6 --min_clusters $MIN_CLUSTERS --max_clusters $MAX_CLUSTERS --n_examples $N_EXAMPLES

# for LAYER in 6 10 14 18 22 26; do
#     python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --layer $LAYER --min_clusters $MIN_CLUSTERS --max_clusters $MAX_CLUSTERS
# done

# for LAYER in 8 14 20 26 32 38; do
#     python ablate_clustering.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --layer $LAYER --min_clusters $MIN_CLUSTERS --max_clusters $MAX_CLUSTERS
# done