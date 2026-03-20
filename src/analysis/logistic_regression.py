import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils import GRAPHS_DIR, load_and_clean_data

class MetadataSensitivityAnalyzer:
    def __init__(self, output_subdir='metadata_analysis'):
        self.output_dir = os.path.join(GRAPHS_DIR, output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def run_per_model_regression(self, df):
        """Calculates sensitivity weights and reports the top determining features."""
        # Deduplicate to get the consensus per article/context
        df_consensus = df.drop_duplicates(subset=['llm_model', 'index', 'detail_level']).copy()
        
        models = df_consensus['llm_model'].unique()
        features = ['source', 'politics', 'gender', 'age_group']
        all_weights = []

        print("\n" + "="*50)
        print("MODEL FEATURE DETERMINATION REPORT")
        print("="*50)

        for model_name in models:
            model_df = df_consensus[
                (df_consensus['llm_model'] == model_name) & 
                (df_consensus['detail_level'] == '+all')
            ]
            
            data = model_df[features + ['llm_consensus_label']].dropna()
            
            if len(data) < 20: 
                continue

            # Dummy Encoding
            X = pd.get_dummies(data[features], drop_first=True)
            y = data['llm_consensus_label']

            # Standardize (Crucial so we can compare 'Age' weight vs 'Politics' weight directly)
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_scaled, y)

            # Store Results
            weights = pd.DataFrame({
                'Feature': X.columns,
                'Weight': lr.coef_[0],
                'llm_model': model_name
            })
            
            top_features = weights.reindex(weights.Weight.abs().sort_values(ascending=False).index)
            primary_feature = top_features.iloc[0]
            
            direction = "INCREASE" if primary_feature['Weight'] > 0 else "DECREASE"
            
            print(f"\n[Model]: {model_name}")
            print(f"  - Primary Driver: {primary_feature['Feature']}")
            print(f"  - Impact: Presence of this feature causes an {direction} in bias labeling.")
            print(f"  - Top 3 Influencers: {', '.join(top_features.Feature.head(3).tolist())}")
            
            all_weights.append(weights)
            self.plot_sensitivity(weights, model_name)

        return pd.concat(all_weights) if all_weights else pd.DataFrame()

    def plot_sensitivity(self, weights_df, model_name):
        weights_df = weights_df.sort_values(by='Weight', key=abs, ascending=False).copy()
        
        weights_df['Direction'] = weights_df['Weight'].apply(
            lambda x: 'Increases Bias Verdict' if x > 0 else 'Decreases Bias Verdict'
        )

        plt.figure(figsize=(10, 8))
        
        custom_palette = {
            'Increases Bias Verdict': '#d65f5f',
            'Decreases Bias Verdict': '#5f9ed1'
        }

        # 4. Use 'Direction' as the hue
        sns.barplot(
            data=weights_df, 
            x='Weight', 
            y='Feature', 
            hue='Direction', 
            palette=custom_palette,
            dodge=False
        )
        
        plt.title(f'Feature Sensitivity (Consensus): {model_name}', fontsize=12, fontweight='bold')
        plt.xlabel('Standardized Coefficient (Importance)')
        plt.axvline(x=0, color='black', linewidth=1.2)
        
        # Move legend to the bottom so it doesn't cover the bars
        plt.legend(title='Effect on Model', loc='lower right')
        
        plt.tight_layout()
        clean_name = model_name.replace(":", "_")
        plt.savefig(os.path.join(self.output_dir, f'weights_{clean_name}.png'))
        plt.close()

if __name__ == "__main__":
    full_df = load_and_clean_data() 
    analyzer = MetadataSensitivityAnalyzer()
    weights_summary = analyzer.run_per_model_regression(full_df)
    
    if not weights_summary.empty:
        weights_summary.to_csv(os.path.join(analyzer.output_dir, 'model_weights_consensus.csv'), index=False)
        print(f"\nDetailed CSV saved to {analyzer.output_dir}")