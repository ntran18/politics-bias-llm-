import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import cohen_kappa_score
from utils import GRAPHS_DIR, load_and_clean_data, MODELS
from statsmodels.stats.contingency_tables import mcnemar

class KappaAnalyzer:
    def __init__(self, output_subdir='inter_model_analysis'):
        self.output_dir = os.path.join(GRAPHS_DIR, output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        
    def get_ensemble_df(self, df):
        """Creates a majority-vote consensus for each model/prompt combination."""
        return df.groupby(['index', 'prompt_type', 'llm_model']).agg({
            'llm_label_bin': lambda x: x.mode()[0],
            'human_label_bin': 'first'
        }).reset_index()

    def calculate_all_kappas(self, df):
        """Calculates Alignment, Inter-Model, and Self-Consistency metrics."""
        results = []
        models = df['llm_model'].unique()
        ensemble_df = self.get_ensemble_df(df)

        # 1. Ensemble Alignment (Model vs Human)
        consensus_model_df = ensemble_df.drop_duplicates(subset=['llm_model', 'index', 'prompt_type']).copy()
        for model in models:
            m_df = consensus_model_df[consensus_model_df['llm_model'] == model]
            kappa = cohen_kappa_score(m_df['llm_label_bin'], m_df['human_label_bin'])
            results.append({'Type': 'Human-Model Alignment', 'Comparison': f'{model} vs Human', 'Kappa': kappa})

        # 2. Inter-Model Agreement (Consensus A vs Consensus B)
        pivot_ensemble = consensus_model_df.pivot_table(
            index=['index', 'prompt_type'], columns='llm_model', values='llm_label_bin'
        ).dropna()
        
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i < j:
                    kappa = cohen_kappa_score(pivot_ensemble[m1], pivot_ensemble[m2])
                    results.append({'Type': 'Inter-Model Agreement', 'Comparison': f'{m1} vs {m2}', 'Kappa': kappa})

        # 3. Self-Consistency (Internal Agreement across Versions)
        for model in models:
            m_runs = df[df['llm_model'] == model].pivot_table(
                index=['index', 'prompt_type'], columns='version', values='llm_label_bin'
            ).dropna()
            if m_runs.shape[1] >= 2:
                kappa_self = cohen_kappa_score(m_runs.iloc[:, 0], m_runs.iloc[:, 1])
                results.append({'Type': 'Self-Consistency', 'Comparison': f'{model} Internal', 'Kappa': kappa_self})

        return pd.DataFrame(results)

    def plot_reliability_gap(self, kappa_df):
        """Graphs Alignment vs Self-Consistency to show the 'Subjectivity Wall'."""
        plt.figure(figsize=(12, 6))
        plot_df = kappa_df[kappa_df['Type'].isin(['Self-Consistency'])].copy()
        plot_df['Model'] = plot_df['Comparison'].str.split(' ').str[0]

        sns.barplot(data=plot_df, x='Model', y='Kappa', hue='Type', palette='viridis')
        
        # Reference lines for interpretation
        plt.axhline(y=0.41, color='gray', linestyle='--', alpha=0.6, label='Moderate Agreement')
        plt.axhline(y=0.81, color='green', linestyle='--', alpha=0.6, label='Almost Perfect')
        
        plt.title('Reliability Gap: Model-Model Alignment vs. Self-Consistency vs. Inter-Model Agreement', fontsize=14, fontweight='bold')
        plt.ylim(-0.2, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'reliability_gap_bars.png'))
        plt.close()

    def plot_inter_model_heatmap(self, kappa_df):
        """Creates a proper square matrix heatmap for Inter-Model agreement."""
        inter_df = kappa_df[kappa_df['Type'] == 'Inter-Model Agreement']
        
        # Extract individual model names from the comparison string
        models = sorted(list(set(inter_df['Comparison'].str.split(' vs ').str[0]) | 
                             set(inter_df['Comparison'].str.split(' vs ').str[1])))
        
        # Initialize square matrix
        matrix = pd.DataFrame(1.0, index=models, columns=models)
        
        for _, row in inter_df.iterrows():
            m1, m2 = row['Comparison'].split(' vs ')
            matrix.loc[m1, m2] = row['Kappa']
            matrix.loc[m2, m1] = row['Kappa']

        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, cmap='RdYlGn', vmin=0, vmax=1)
        plt.title("Inter-Model Agreement Matrix (Consensus Kappa)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'inter_model_matrix.png'))
        plt.close()
        
        
    def run_mcnemar_on_ensembles(self, ensemble_df, model_a, model_b, alpha=0.05):
        # Pivot to get a side-by-side comparison of the two models' majority votes
        pivot = ensemble_df.pivot_table(
            index=['index', 'prompt_type'], 
            columns='llm_model', 
            values='llm_label_bin'
        ).dropna()
        
        # Create the 2x2 contingency table
        # Cell 0,0: Both said Biased | Cell 0,1: A said Biased, B said Not
        # Cell 1,0: B said Biased, A said Not | Cell 1,1: Both said Not
        table = pd.crosstab(pivot[model_a], pivot[model_b])
        print(f"Contingency Table for {model_a} vs {model_b}:\n{table}\n")
        
        # Calculate McNemar
        result = mcnemar(table, exact=True)
        print(f"McNemar test result for {model_a} vs {model_b}: p-value = {result.pvalue}")
        return result.pvalue, result.statistic
    
    def run_mcnemar_on_ensembles_all_pairs(self, ensemble_df, models, alpha=0.05):
        pairs = [(m1, m2) for i, m1 in enumerate(models) for j, m2 in enumerate(models) if i < j]
        results = []
        for m1, m2 in pairs:
            p_value, stat = self.run_mcnemar_on_ensembles(ensemble_df, m1, m2, alpha)
            results.append({'Model A': m1, 'Model B': m2, 'Statistic': stat, 'p-value': p_value})
        return pd.DataFrame(results)

if __name__ == "__main__":
    clean_df = load_and_clean_data()
    analyzer = KappaAnalyzer()
    
    # Run analysis
    kappa_results = analyzer.calculate_all_kappas(clean_df)
    print(kappa_results.to_string(index=False))
    
    # Save results
    kappa_results.to_csv(os.path.join(analyzer.output_dir, 'kappa_results.csv'), index=False)
    
    # Generate Plots
    analyzer.plot_reliability_gap(kappa_results)
    analyzer.plot_inter_model_heatmap(kappa_results)
    analyzer.run_mcnemar_on_ensembles(analyzer.get_ensemble_df(clean_df), 'phi4-mini', 'llama3.2:3b')
    all_pairs_results = analyzer.run_mcnemar_on_ensembles_all_pairs(analyzer.get_ensemble_df(clean_df), MODELS)
    all_pairs_results.to_csv(os.path.join(analyzer.output_dir, 'mcnemar_all_pairs.csv'), index=False)
    