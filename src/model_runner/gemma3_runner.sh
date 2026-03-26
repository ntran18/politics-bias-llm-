#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e  

# --- Configuration ---
MODEL="gemma3:27b"
VERSIONS=("test")
FILE_TYPE="${1:-test_two_queries}"
TEMP=0.0
WORKERS=1
CHECKPOINT=8
PORT=11437
CTX_LEN=128000

# --- Execution ---
for VERSION in "${VERSIONS[@]}"; do
    echo "------------------------------------------"
    echo "Starting $MODEL for version: $VERSION"
    echo "------------------------------------------"
    
    python main.py \
        --provider ollama \
        --file-type "$FILE_TYPE" \
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