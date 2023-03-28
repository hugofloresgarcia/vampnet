#!/bin/bash

# Get the command to execute from the user
command_to_execute="$1"

# Get the maximum number of GPUs to use from the user
max_gpus="$2"

# Get the number of instances to start per GPU from the user
instances_per_gpu="$3"

# Set the CUDA_VISIBLE_DEVICES flag for each GPU
for gpu_id in $(seq 0 $(($max_gpus - 1))); do
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    # Start the specified number of instances for this GPU
    for i in $(seq 1 "$instances_per_gpu"); do
        # Run the command in the background
        $command_to_execute &
    done
done

# Wait for all instances to finish
wait