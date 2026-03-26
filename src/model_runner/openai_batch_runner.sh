#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
PYTHON_BIN="$REPO_ROOT/venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python"
fi

if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "Loaded environment from $ENV_FILE"
else
    echo "No .env found at $ENV_FILE (continuing with current shell env vars)"
fi
echo "Using Python interpreter: $PYTHON_BIN"

MODEL="gpt-4o-mini"
FILE_TYPE="${1:-test_two_queries}"
VERSIONS=("${2:-test}")
TEMP=0.0
CHECKPOINT=100
POLL_INTERVAL=60

for VERSION in "${VERSIONS[@]}"; do
    echo "------------------------------------------"
    echo "Starting OpenAI Batch model=$MODEL version=$VERSION file_type=$FILE_TYPE"
    echo "------------------------------------------"

    "$PYTHON_BIN" "$SCRIPT_DIR/main.py" \
        --provider openai \
        --openai-mode batch \
        --file-type "$FILE_TYPE" \
        --model "$MODEL" \
        --version "$VERSION" \
        --prompt-dir "$REPO_ROOT/data/prompts" \
        --output-dir "$REPO_ROOT/results" \
        --model-temperature $TEMP \
        --checkpoint-size $CHECKPOINT \
        --batch-poll-interval $POLL_INTERVAL

    echo "Completed OpenAI Batch model=$MODEL version=$VERSION file_type=$FILE_TYPE"
done

echo "All OpenAI batch jobs submitted/completed successfully for file_type=$FILE_TYPE."
