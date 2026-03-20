#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e  

# --- Configuration ---
MODEL="qwen3:4b"
VERSIONS=("v6.4" "v6.5" "v6.6")
TEMP=0.7
WORKERS=4
CHECKPOINT=8
PORT=11439
CTX_LEN=32000

# --- Execution ---
for VERSION in "${VERSIONS[@]}"; do
    echo "------------------------------------------"
    echo "Starting $MODEL for version: $VERSION"
    echo "------------------------------------------"
    
    python ollama_llm_executor.py \
        --file-type all \
        --model "$MODEL" \
        --version "$VERSION" \
        --workers $WORKERS \
        --model-temperature $TEMP \
        --checkpoint-size $CHECKPOINT \
        --ollama-server-port $PORT \
        --context-length $CTX_LEN

    echo "Completed $MODEL for version: $VERSION"
done

echo "All scripts completed successfully."