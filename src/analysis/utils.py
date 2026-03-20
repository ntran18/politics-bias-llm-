import pandas as pd
import os
import numpy as np

# --- Configuration & Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
COMBINED_DATASETS_DIR = os.path.join(PROJECT_ROOT, 'results', 'combined_datasets')
GRAPHS_DIR = os.path.join(PROJECT_ROOT, 'graphs')
MODELS = ["qwen3:4b", "qwen3:30b", "phi4-mini", "llama3.2:3b", "gemma3:27b", "r1-1776:latest"]

DETAIL_LEVEL_MAP = {
    'prompt_article_info': 'article_only',
    'prompt_source_variants': '+source',
    'prompt_politics_variants': '+politics',
    'prompt_pii_combined_variants': '+all'
}

def get_age_group(age):
    """Categorizes age into research-friendly buckets."""
    try:
        age = float(age)
        if age < 30: return '18-29'
        if age < 50: return '30-49'
        return '50+'
    except (ValueError, TypeError):
        return 'Unknown'

def load_and_clean_data(version_filter: str = None) -> pd.DataFrame:
    """Load, merge, and clean LLM bias detection datasets."""
    
    if not os.path.exists(COMBINED_DATASETS_DIR):
        raise FileNotFoundError(f"Directory not found: {COMBINED_DATASETS_DIR}")

    # ========= Loading files ========= 
    files = [f for f in os.listdir(COMBINED_DATASETS_DIR) if f.endswith('_combined.csv') and not f.startswith('_')]
    if not files:
        print("No datasets found.")
        return pd.DataFrame()

    all_dfs = []
    for file in sorted(files):
        df = pd.read_csv(os.path.join(COMBINED_DATASETS_DIR, file))
        all_dfs.append(df)
        print(f"✓ Loaded {file}: {len(df)} rows")

    df = pd.concat(all_dfs, ignore_index=True)

    # Vectorized binary conversion
    label_map = {'is-biased': 1, 'is-not-biased': 0}
    df['llm_label_bin'] = df['llm_assessment'].map(label_map).fillna(-1).astype(int)
    df['human_label_bin'] = df['bias-question'].map(label_map).fillna(-1).astype(int)
    
    initial_count = len(df)
    df = df[df['llm_label_bin'] != -1]
    df = df[df['human_label_bin'] != -1]

    print(f"📊 Cleaned Data: {len(df)} rows (Dropped {initial_count - len(df)} invalid/empty rows)")

    # 3. Feature Engineering for Analysis
    df['match'] = (df['llm_label_bin'] == df['human_label_bin']).astype(int)
    df['detail_level'] = df['prompt_type'].map(DETAIL_LEVEL_MAP).fillna('unknown')
    df['age_group'] = df['age'].apply(get_age_group)
    
    if 'index' in df.columns:
        group_cols = ['llm_model', 'index', 'detail_level']
        df['llm_consensus_label'] = df.groupby(group_cols)['llm_label_bin'].transform(
            lambda x: x.mode()[0] if not x.mode().empty else np.nan
        )

    print(f"  Models: {df['llm_model'].nunique()}")
    print(f"  Versions: {df['version'].nunique()}")
    print(f"  Prompt types: {df['prompt_type'].nunique()}\n")
    print(f"  Columns: {', '.join(df.columns)}\n")
    return df