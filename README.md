# Political Bias in LLMs: Research Pipeline (Test Profile)

This repository evaluates political-bias judgments from multiple LLMs under different prompt conditions, then runs reproducible analysis workflows that produce research artifacts (CSV tables and figures).

This README is intentionally configured for test execution only. All commands below default to test-safe settings (`version=test`, `file-type=test_two_queries`).

## 1. Research Scope

### Core question

How stable are LLM bias judgments across model families, metadata framing, and reasoning strategy?

### Analysis modules

- Q1: Overall alignment with human labels (`most_aligned_models.py`)
- Q2: Alignment by political subgroup (`models_align_with_politics.py`)
- Q3: Inter-model agreement (`inter_model.py`)
- Q4: Metadata effects on outputs (`metadata_analysis.py`)
- Q5: Direct vs CoT vs Chained CoT (`cot_analysis.py`)
- Dataset-level diagnostics (`data_analysis.py`)

## 2. Repository Map

- `src/prompt_generation/`: data cleaning, article enrichment, prompt construction
- `src/model_runner/`: multi-provider inference runner (Ollama, OpenAI, Gemini)
- `src/analysis/`: statistical analysis scripts and report generation
- `data/`: cleaned datasets and generated prompts
- `results/`: model outputs by version/model
- `analysis_reports/`: final analysis CSVs and figures

## 3. Reproducible Environment Setup

From repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install python-dotenv matplotlib scikit-learn scipy statsmodels
```

Set API keys only for providers you use:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
# or
export GOOGLE_API_KEY="..."
```

## 4. Test Profile Defaults

Use these defaults everywhere unless you intentionally scale up:

- `version=test`
- `file-type=test_two_queries`
- small worker count (`--workers 1` or `2`)
- deterministic temperature (`--model-temperature 0.0`)

## 5. Step-by-Step Pipeline (Test-Only)

### Step A: Generate prompts

Run from `src/prompt_generation` (required because paths are relative in this module):

```bash
cd src/prompt_generation
python prompt_generator.py --clean --fetch --all-prompts --version test
```

Optional: generate only selected prompt families:

```bash
python prompt_generator.py --prompts articles_info politics sources source_politics source_pii politics_pii pii_combined_all --version test
```

Output location:

- `data/prompts/test/`

### Step B: Run model inference

Run from repository root with `PYTHONPATH=src`:

```bash
cd ../..
```

Ollama (test slice):

```bash
PYTHONPATH=src python src/model_runner/main.py \
  --provider ollama \
  --model llama4:scout \
  --version test \
  --file-type test_two_queries \
  --prompt-dir data/prompts \
  --output-dir results \
  --workers 1 \
  --model-temperature 0.0
```

OpenAI direct (test slice):

```bash
PYTHONPATH=src python src/model_runner/main.py \
  --provider openai \
  --openai-mode direct \
  --model gpt-5.4-mini \
  --version test \
  --file-type test_two_queries \
  --prompt-dir data/prompts \
  --output-dir results \
  --model-temperature 0.0
```

Gemini (test slice):

```bash
PYTHONPATH=src python src/model_runner/main.py \
  --provider gemini \
  --model gemini-2.5-flash-lite \
  --version test \
  --file-type test_two_queries \
  --prompt-dir data/prompts \
  --output-dir results \
  --workers 1 \
  --model-temperature 0.0
```

Output location:

- `results/test/<model>/llm_outputs/*.csv`

### Step D: Run analysis scripts

Run from `src/analysis`:

```bash
cd src/analysis
python data_analysis.py
python most_aligned_models.py
python models_align_with_politics.py
python inter_model.py
python metadata_analysis.py
python cot_analysis.py
```

Output location:

- `analysis_reports/dataset/`
- `analysis_reports/question1/`
- `analysis_reports/question2/`
- `analysis_reports/question3/`
- `analysis_reports/question4/`
- `analysis_reports/question5/`

## 6. Input/Output Data

### Prompt CSV required columns

- `article_id`
- `index`
- `prompt`

### Runner output columns (common)

- `llm_assessment`
- `llm_confidence`
- `llm_explanation` (when explanation is enabled)
- `llm_model`
- `llm_error`

These two identifiers are not expected to be identical.

## 9. For full analysis that presented to the paper

The complete set of prompts, model outputs, and analysis results used in this study are available on [Google Drive](https://drive.google.com/drive/folders/1AtnBNQbIRd-DJRad8vtg5F_psUXxRBmF?usp=sharing)
