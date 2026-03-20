import pandas as pd
import numpy as np
import os
from utils import GRAPHS_DIR, load_and_clean_data

def final_qualitative_deep_dive(df, top_n=15):
    # 1. Identify the most controversial articles (Conflict Score)
    article_stats = df.groupby('index').agg({
        'llm_label_bin': 'mean',
        'human_label_bin': 'first'
    })
    article_stats['conflict_score'] = 0.5 - abs(article_stats['llm_label_bin'] - 0.5)
    conflict_indices = article_stats.sort_values('conflict_score', ascending=False).head(top_n).index

    # 2. Filter for +all context (where all metadata triggers were active)
    report_df = df[(df['index'].isin(conflict_indices)) & (df['detail_level'] == '+all')].copy()
    
    # Take the first version to avoid duplicates in the report
    first_v = sorted(report_df['version'].unique())[0]
    report_df = report_df[report_df['version'] == first_v]

    # 3. Pivot side-by-side reasoning
    pivot_report = report_df.pivot(index='index', columns='llm_model', values='llm_explanation')
    pivot_report = pivot_report.join(article_stats[['human_label_bin', 'llm_label_bin']])
    
    # 4. Keyword Analysis: What "Criteria" is each model using?
    # We define themes based on typical LLM bias justification patterns
    themes = {
        'Institutional': ['source', 'fox', 'cnn', 'outlet', 'reputation', 'brand'],
        'Linguistic': ['adjective', 'loaded', 'tone', 'wording', 'adverb', 'language', 'framing'],
        'Demographic': ['reader', 'liberal', 'conservative', 'identity', 'age', 'gender'],
        'Structural': ['omission', 'context', 'one-sided', 'balance', 'perspective'],
    }

    # Detect criteria for each model's explanations
    criteria_results = []
    for model in df['llm_model'].unique():
        all_text = " ".join(df[df['llm_model'] == model]['llm_explanation'].astype(str)).lower()
        counts = {theme: sum(all_text.count(word) for word in words) for theme, words in themes.items()}
        counts['llm_model'] = model
        criteria_results.append(counts)

    return pivot_report, pd.DataFrame(criteria_results)

# --- Execute ---
if __name__ == "__main__":
    full_df = load_and_clean_data()
    report, criteria = final_qualitative_deep_dive(full_df)

    # Save CSVs
    report.to_csv(os.path.join(GRAPHS_DIR, 'conflict_comparison_matrix.csv'))
    criteria.to_csv(os.path.join(GRAPHS_DIR, 'model_bias_criteria_counts.csv'))
    
    print("✓ Matrix and Criteria files generated in /graphs/")