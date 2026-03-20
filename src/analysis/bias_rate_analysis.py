import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import GRAPHS_DIR, load_and_clean_data
from scipy.stats import chi2_contingency
import statsmodels.api as sm

class BiasAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        self.detail_order = ['article_only', '+source', '+politics', '+all']
    
    def _calculate_summary(self, df, group_cols):
        """Private helper to handle grouping, aggregation, and formatting."""
        # Core grouping and aggregation logic
        summary = df.groupby(group_cols).agg(
            llm_bias_rate=('llm_consensus_label', 'mean'),
            human_bias_rate=('human_label_bin', 'mean'),
            accuracy=('match', 'mean'),
            avg_confidence=('llm_confidence', 'mean')
        ).reset_index()

        # Consistent formatting for percentages
        pct_cols = ['llm_bias_rate', 'human_bias_rate', 'accuracy']
        summary[pct_cols] = (summary[pct_cols] * 100).round(2)
        
        return summary

    def generate_summaries(self, df: pd.DataFrame):
        """Generates both aggregate and granular summaries using the helper."""
        # Define the base groups required for every summary
        base_groups = ['llm_model', 'detail_level']

        agg_summary = self._calculate_summary(df, base_groups)
        source_summary = self._calculate_summary(df, base_groups + ['source'])
        politics_summary = self._calculate_summary(df, base_groups + ['politics'])
        age_summary = self._calculate_summary(df, base_groups + ['age_group'])
        gender_summary = self._calculate_summary(df, base_groups + ['gender'])
        
        
        
        return agg_summary, source_summary, politics_summary, age_summary, gender_summary

    def plot_base_metrics(self, summary_table):
        """Generates the three core research plots."""
        summary_table['detail_level'] = pd.Categorical(
            summary_table['detail_level'], categories=self.detail_order, ordered=True
        )

        # Plot 1: Accuracy Trend
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=summary_table, x='detail_level', y='accuracy', hue='llm_model', marker='o')
        plt.title('Human Alignment (Accuracy) by Information Level')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_trend.png'))

        # Plot 2: Bias Rate vs Human Baseline
        plt.figure(figsize=(12, 7))
        sns.barplot(data=summary_table, x='detail_level', y='llm_bias_rate', hue='llm_model')
        plt.axhline(y=summary_table['human_bias_rate'].iloc[0], color='red', linestyle='--', label='Human Ground Truth')
        plt.title('Bias Detection Rate: LLM vs Human')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'bias_rate_comparison.png'))

        # Plot 3: Calibration Scatter
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=summary_table, x='accuracy', y='avg_confidence', hue='llm_model', style='detail_level', s=100)
        plt.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect Calibration')
        plt.title('Confidence vs Accuracy Calibration')
        plt.savefig(os.path.join(self.output_dir, 'calibration_scatter.png'))
        plt.tight_layout()
        plt.close('all')
        
    def plot_bias_trend(self, df, category_col, filename=None):
        """
        Generalized visualization for Bias Labeling Rates.
        
        Args:
            df (pd.DataFrame): The input data.
            category_col (str): The column to use for 'hue' (e.g., 'source' or 'politics').
            filename (str): Optional custom filename.
        """
        df = df.copy()
        
        display_name = category_col.replace('_', ' ').title()
        if filename is None:
            filename = f'bias_rate_trend_by_{category_col}.png'

        # Only keep article_only and +source/+politics for this plot
        if category_col == 'source':
            df = df[df['detail_level'].isin(['article_only', '+source'])]
        elif category_col == 'politics':
            df = df[df['detail_level'].isin(['article_only', '+politics'])]

        if category_col in ['source', 'politics']:
            detail_order = ['article_only', f'+{category_col}']
        else:
            detail_order = self.detail_order
        df['detail_level'] = pd.Categorical(
            df['detail_level'], 
            categories=detail_order, 
            ordered=True
        )

        g = sns.FacetGrid(df, col="llm_model", col_wrap=3, height=4, aspect=1.2, sharey=True)

        g.map_dataframe(
            sns.barplot, 
            x="detail_level", 
            y="llm_consensus_label", 
            hue=category_col,
            errorbar=('ci', 95),
            capsize=.05,
            palette="viridis"
        )

        g.set_axis_labels("Information Detail Level", "Bias Labeling Rate (%)")
        g.set_titles(col_template="{col_name}")
        g.add_legend(title=display_name)
        
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f'Bias Labeling Sensitivity: Information Progression vs. {display_name}', 
                    fontsize=16, fontweight='bold')
        
        g.tight_layout()
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to: {save_path}")
    
    def _chi_squared_test(self, df, category_col):
        contingency_table = pd.crosstab(df[category_col], df['llm_consensus_label'])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        return p < 0.05, p
    
    def run_statistical_tests(self, df):
        results = {}
        
        # --- Test 1: Source vs Bias Rate (Chi-Squared) ---
        is_significant, p = self._chi_squared_test(df, 'source')
        results['Source_Effect'] = {'p-value': p, 'Significant': is_significant}

        # --- Test 2: Politics vs Bias Rate (Chi-Squared) ---
        is_significant, p_pol = self._chi_squared_test(df, 'politics')
        results['Politics_Effect'] = {'p-value': p_pol, 'Significant': is_significant}

        # --- Test 3: Age vs Bias Rate (Chi-Squared) ---
        df_age = df[df['age_group'] != 'Unknown'].copy()
        is_significant, p  = self._chi_squared_test(df_age, 'age_group')
        results['Age_Effect'] = {'p-value': p, 'Significant': is_significant}
        
        # --- Test 4: Gender vs Bias Rate (Chi-Squared) ---
        is_significant, p_gender = self._chi_squared_test(df, 'gender')
        results['Gender_Effect'] = {'p-value': p_gender, 'Significant': is_significant}
        
        # Save results to a text file
        results_file = os.path.join(self.output_dir, 'statistical_test_results.txt')
        with open(results_file, 'w') as f:
            f.write("Statistical Test Results:\n")
            for key, value in results.items():
                f.write(f"  {key}: p-value = {value['p-value']:.4f}, Significant = {value['Significant']}\n")
                
        # Print the bias rates for each model across detail levels
        print("\n--- LLM Bias Detection Rates (%) ---")
        pivot_table = agg_table.pivot(index='llm_model', columns='detail_level', values='llm_bias_rate')
        # Reorder columns to match your research flow
        pivot_table = pivot_table[['article_only', '+source', '+politics', '+all']]
        print(pivot_table)

        # Optional: Save to CSV for your paper's appendix
        pivot_table.to_csv(os.path.join(analysis_dir, 'model_bias_rates.csv'))

        return results
    
    def _test_metadata_impact(self, df, model_name, detail_level_to_test):
        """
        Compares 'article_only' vs a specific detail level (e.g., '+source') 
        to see if the metadata caused a significant shift.
        """
        model_df = df[df['llm_model'] == model_name]
        
        # Filter for the two levels we want to compare
        comparison_df = model_df[model_df['detail_level'].isin(['article_only', detail_level_to_test])]
        
        # Create the contingency table
        contingency = pd.crosstab(comparison_df['detail_level'], comparison_df['llm_consensus_label'])
        print(contingency)
        
        chi2, p, _, _ = chi2_contingency(contingency)
        return p, p < 0.05
    
    def run_granular_statistical_tests(self, df):
        """Runs Chi-Squared tests for each model to see what variables actually move the needle."""
        models = df['llm_model'].unique()
        print("models found for granular testing:", models)
        variables = ['+politics', '+source', '+all']
        
        granular_results = []

        for model in models:
            model_results = {'llm_model': model}
            
            for var in variables:
                # Drop 'Unknown' or NaNs for the specific test
                p, significant = self._test_metadata_impact(df, model, var)
                model_results[f'{var}_p_value'] = round(p, 4)
                model_results[f'{var}_significant'] = significant
            
            granular_results.append(model_results)
        
        return pd.DataFrame(granular_results)
    

if __name__ == "__main__":
    # Setup
    analysis_dir = os.path.join(GRAPHS_DIR, 'model_performance_summary')
    analyzer = BiasAnalyzer(analysis_dir)
    
    # Load and Process
    clean_df = load_and_clean_data()
    consensus_df = clean_df.drop_duplicates(subset=['llm_model', 'index', 'detail_level']).copy()
    agg_table, source_summary_table, politics_summary_table, age_summary_table, gender_summary_table = analyzer.generate_summaries(consensus_df)
    
    # Generate Visuals
    analyzer.plot_base_metrics(agg_table)
    
    # analyzer.plot_5_variable_grid(agg_table)
    analyzer.plot_bias_trend(consensus_df, category_col='source')
    analyzer.plot_bias_trend(consensus_df, category_col='politics')

    analyzer.plot_bias_trend(consensus_df, category_col='age_group')
    analyzer.plot_bias_trend(consensus_df, category_col='gender')
    results = analyzer.run_statistical_tests(consensus_df)
    print("Statistical Test Results:")
    for key, value in results.items():
        print(f"  {key}: p-value = {value['p-value']:.4f}, Significant = {value['Significant']}")
        
    print("\nGranular Statistical Test Results by Model:")
    granular_results = analyzer.run_granular_statistical_tests(consensus_df)
    for _, row in granular_results.iterrows():
        print(f"  {row['llm_model']}:")
        for var in ['+politics', '+source', '+all']:
            print(f"    {var}: p-value = {row[f'{var}_p_value']:.4f}, Significant = {row[f'{var}_significant']}")
        
    print(f"Analysis complete. Results saved to: {analysis_dir}")