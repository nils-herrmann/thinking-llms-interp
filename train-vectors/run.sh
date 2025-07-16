
#MINIBATCH_SIZE_PER_GPU=6
#NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
#TOTAL_MINIBATCHES=$((MINIBATCH_SIZE_PER_GPU * NUM_GPUS))
#echo "Total minibatches: $TOTAL_MINIBATCHES"

# Check if cluster IDs are provided as arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <cluster_id1> [cluster_id2] [cluster_id3] ..."
    echo "Example: $0 0 1 2"
    exit 1
fi

# Iterate over all provided cluster IDs
for cluster in "$@"; do
    echo "Processing cluster: $cluster"
    python optimize_steering_vectors.py \
        --model meta-llama/Llama-3.1-8B \
        --max_iters 20 \
        --n_training_examples 2048 \
        --minibatch_size 8 \
        --layer 14 \
        --has_bos_token True \
        --steering_vector_idx $cluster \
        --lr "1e-2" \
        --use_activation_perplexity_selection
        --use_wandb
done

python visualize_vector_losses.py --model meta-llama/Llama-3.1-8B