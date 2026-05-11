import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.config import Constants
from core.utils import Utils

# Use centralized utils for paths, IO, and helpers
utils = Utils("question3")
PROMPT_FILE = "prompt_article_info.csv"

# Directories via utils
FIGURE_DIR = utils.figure_dir
CSV_DIR = utils.csv_dir

# Apply shared plot style
plt.rcParams.update(Constants.PLOT_STYLE)


# ============================================================
# 1. Load model predictions
# ============================================================
long_df = utils.load_model_outputs(Constants.REASONING_CONDITION_PATHS["Direct"], PROMPT_FILE)

utils.save_csv(long_df, "inter_model_predictions_long.csv", index=False)

print("\nLoaded model predictions:")
print(long_df.head())


# ============================================================
# 2. Convert to wide format
# Rows = shared article/index
# Columns = models
# Values = model predictions
# ============================================================
wide_df = long_df.pivot_table(
    index=["index", "article_id"],
    columns="Model",
    values="llm_label",
    aggfunc="first"
)

# Keep only rows where every model has a valid prediction
wide_complete = wide_df.dropna(axis=0, how="any").copy()
wide_complete = wide_complete.astype(int)

utils.save_csv(wide_complete, "inter_model_predictions_wide_complete.csv", index=False)

models = wide_complete.columns.tolist()

print("\nWide model prediction table:")
print(wide_complete.head())

print("\nNumber of shared rows across all models:", len(wide_complete))
print("Number of models:", len(models))


# ============================================================
# 3. Pairwise inter-model agreement
# ============================================================
kappa_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
raw_agreement_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
disagreement_matrix = pd.DataFrame(index=models, columns=models, dtype=float)

pairwise_rows = []

for i, model_a in enumerate(models):
    for j, model_b in enumerate(models):
        preds_a = wide_complete[model_a]
        preds_b = wide_complete[model_b]

        raw_agreement = (preds_a == preds_b).mean()
        disagreement = 1 - raw_agreement
        kappa = utils.safe_kappa_pair(preds_a, preds_b)

        raw_agreement_matrix.loc[model_a, model_b] = raw_agreement
        disagreement_matrix.loc[model_a, model_b] = disagreement
        kappa_matrix.loc[model_a, model_b] = kappa

        if i < j:
            pairwise_rows.append({
                "Model_A": model_a,
                "Model_B": model_b,
                "Model_A_Display": utils.pretty_model_name(model_a),
                "Model_B_Display": utils.pretty_model_name(model_b),
                "Raw_Agreement": raw_agreement,
                "Disagreement_Rate": disagreement,
                "Cohen_Kappa": kappa,
                "N_Shared": len(wide_complete)
            })

pairwise_df = pd.DataFrame(pairwise_rows)

kappa_matrix.to_csv(os.path.join(CSV_DIR, "inter_model_kappa_matrix.csv"), index=False)
utils.save_csv(kappa_matrix, "inter_model_kappa_matrix.csv", index=False)
utils.save_csv(raw_agreement_matrix, "inter_model_raw_agreement_matrix.csv", index=False)
utils.save_csv(disagreement_matrix, "inter_model_disagreement_matrix.csv", index=False)
utils.save_csv(pairwise_df, "inter_model_pairwise_results.csv", index=False)

print("\n=== Pairwise inter-model agreement ===")
print(pairwise_df.sort_values("Cohen_Kappa", ascending=False).head(20))


# ============================================================
# 4. Model-level summary
# ============================================================
model_summary = []

for model in models:
    other_models = [m for m in models if m != model]

    mean_kappa = kappa_matrix.loc[model, other_models].mean()
    mean_raw_agreement = raw_agreement_matrix.loc[model, other_models].mean()
    mean_disagreement = disagreement_matrix.loc[model, other_models].mean()
    bias_prediction_rate = wide_complete[model].mean()

    model_summary.append({
        "Model": model,
        "Model_Display": utils.pretty_model_name(model),
        "Mean_Kappa_With_Other_Models": mean_kappa,
        "Mean_Raw_Agreement_With_Other_Models": mean_raw_agreement,
        "Mean_Disagreement_With_Other_Models": mean_disagreement,
        "Bias_Prediction_Rate": bias_prediction_rate
    })

model_summary = pd.DataFrame(model_summary)

model_summary = model_summary.sort_values(
    "Mean_Kappa_With_Other_Models",
    ascending=False
)

utils.save_csv(model_summary, "inter_model_summary_by_model.csv", index=False)

print("\n=== Model-level inter-model agreement summary ===")
print(model_summary)


# ============================================================
# 5. Heatmaps
# ============================================================
utils.plot_heatmap(
    fig_size=(8, 6),
    df=kappa_matrix,
    model_order=models,
    cbar_label="",
    filename="q3_inter_model_kappa_heatmap",
    vmin=-0.2,
    vmax=1.0,
    cmap="coolwarm_r",
    triangle="lower"
)

utils.plot_heatmap(
    fig_size=(8, 6),
    df=disagreement_matrix,
    model_order=models,
    cbar_label="Disagreement rate",
    filename="q3_inter_model_disagreement_heatmap",
    vmin=0,
    vmax=1,
    cmap="coolwarm_r",
    triangle="lower"
)


# ============================================================
# 6. Bar chart: mean kappa with other models
# ============================================================
utils.plot_bar_chart(
    df=model_summary,
    x_col="Model_Display",
    y_col="Mean_Kappa_With_Other_Models",
    filename="mean_inter_model_kappa_by_model",
    ylabel="Mean Cohen's kappa with other models",
    color=Constants.COLORS["Default"],
    sort_by="Mean_Kappa_With_Other_Models",
    ascending=False,
    horizontal_line=0,
    value_format=".2f",
    text_offset=0.01,
)


# ============================================================
# 7. Bar chart: model bias prediction rate
# ============================================================
utils.plot_bar_chart(
    df=model_summary,
    x_col="Model_Display",
    y_col="Bias_Prediction_Rate",
    filename="bias_prediction_rate_by_model",
    ylabel="Proportion predicted biased",
    color=Constants.COLORS["Default"],
    sort_by="Bias_Prediction_Rate",
    ascending=False,
    ylim=(0, 1),
    value_format=".2f",
    text_offset=0.015,
)