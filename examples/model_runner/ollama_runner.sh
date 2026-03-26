#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python"
fi

MODEL="llama4:scout"
FILE_TYPE="${1:-test_two_queries}"
VERSIONS=("${2:-test}")
TEMP=0.0
WORKERS=2
CHECKPOINT=8
PORT=11438
CTX_LEN=16000

for VERSION in "${VERSIONS[@]}"; do
    echo "------------------------------------------"
    echo "Starting $MODEL for version: $VERSION"
    echo "------------------------------------------"

    "$PYTHON_BIN" "$REPO_ROOT/src/model_runner/main.py" \
        --provider ollama \
        --file-type "$FILE_TYPE" \
        --model "$MODEL" \
        --version "$VERSION" \
        --prompt-dir "$REPO_ROOT/data/prompts" \
        --output-dir "$REPO_ROOT/results" \
        --workers $WORKERS \
        --model-temperature $TEMP \
        --checkpoint-size $CHECKPOINT \
        --ollama-server-port $PORT \
        --context-length $CTX_LEN

    echo "Completed $MODEL for version: $VERSION"
done

echo "All scripts completed successfully."
