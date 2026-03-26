#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e  

# --- Configuration ---
MODEL="llama4:scout"
VERSIONS=("test")
FILE_TYPE="${1:-test_two_queries}"
TEMP=0.0
WORKERS=2
CHECKPOINT=8
PORT=11438
CTX_LEN=16000

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