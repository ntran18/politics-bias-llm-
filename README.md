# LLM Political Bias Runner

This repository runs political-bias classification prompts against multiple LLM providers and saves structured CSV outputs.

## Supported providers

- Ollama (local models)
- OpenAI Batch API
- Gemini API

## Input and output

Input prompt CSVs must include:

- `article_id`
- `index`
- `prompt`

Output CSV rows include:

- `llm_assessment`
- `llm_confidence`
- `llm_explanation`
- `llm_model`
- `llm_error`

## Setup

Install dependencies:

`pip install -r requirements.txt`

Set provider keys when needed:

- OpenAI: `export OPENAI_API_KEY=...`
- Gemini: `export GEMINI_API_KEY=...` or `export GOOGLE_API_KEY=...`

## Generate prompts

From `src/prompt_generation`: 

- Generate clean data + fetch article info + generate all prompts for a version:
  `python prompt_generator.py --clean --fetch --all-prompts --version v7`

- Generate only selected prompt files:
  `python prompt_generator.py --prompts articles_info politics sources source_politics source_pii politics_pii pii_combined_all --version v7`

- Limit prompt length (character truncation) during generation:
  `python prompt_generator.py --all-prompts --version v7 --context-length 32768`

Generated prompt CSVs are written to `data/prompts/<version>/`.

## Run from CLI

From `src/model_runner`:

- Ollama (auto provider):
  `python main.py --model llama3.2:3b --version v7.1 --file-type all`

- OpenAI Batch:
  `python main.py --provider openai --openai-mode batch --model gpt-4o-mini --version v7.1 --file-type all --batch-poll-interval 60`

- Gemini:
  `python main.py --provider gemini --model gemini-2.0-flash-lite --version v7.1 --file-type all --workers 4`

## Run with provided shell scripts

From `src/model_runner`:

- OpenAI batch: `bash openai_batch_runner.sh`
- Gemini: `bash gemini_runner.sh`
- Ollama models: `bash gemma3_runner.sh`, `bash llama_4_scout_runner.sh`, `bash qwen3_30_runner.sh`, `bash qwen3_4b_runner.sh`, `bash r1-1776_runner.sh`, `bash phi4_mini_runner.sh`

## Notes

- The runner writes outputs under `results/<version>/<model>/llm_outputs/`.
- `--provider auto` infers provider from model name.
