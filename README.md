# LLAM-political-bias

This is a project to find whether LLM can detect political bias in the media articles.

## Multi-provider model runner

The runner at `src/model_runner/ollama_llm_executor.py` now supports:

- Ollama local models (existing behavior)
- OpenAI Batch API (cost-optimized, async completion window up to 24h)
- Gemini live API

Input CSV format is unchanged (`article_id`, `index`, `prompt`) and output CSV format is unchanged (`llm_assessment`, `llm_confidence`, `llm_explanation`, `llm_model`).

### Setup

Install dependencies:

`pip install -r requirements.txt`

Set API keys if using cloud providers:

- OpenAI: `export OPENAI_API_KEY=...`
- Gemini: `export GEMINI_API_KEY=...` (or `GOOGLE_API_KEY`)

### Run examples

From `src/model_runner`:

- Ollama (auto-detected by model name):
  `python ollama_llm_executor.py --model llama3.2:3b --version v7.1 --file-type all`

- OpenAI Batch (cost-optimized):
  `python ollama_llm_executor.py --provider openai --openai-mode batch --model gpt-4o-mini --version v7.1 --file-type all --batch-poll-interval 60`

- Gemini:
  `python ollama_llm_executor.py --provider gemini --model gemini-2.0-flash-lite --version v7.1 --file-type all --workers 4`

## Running Tests

`python -m unittest article_fetcher_test.py`

## New analysis scripts

From repository root:

- Top explanation words by model + label:
  `venv/bin/python src/analysis/top_explanation_words_analysis.py --top-n 10`

- Interactive disagreement explorer (article-level):
  `venv/bin/python src/analysis/interactive_disagreement_explorer.py --top-n 20`

- Non-interactive preview mode:
  `venv/bin/python src/analysis/interactive_disagreement_explorer.py --top-n 20 --no-interactive`

Default output folders (driven by `GRAPHS_DIR` in `src/analysis/utils.py`):

- `graphs-2/top_explanation_words/`
- `graphs-2/interactive_disagreement/`
