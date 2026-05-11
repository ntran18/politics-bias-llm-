import pandas as pd
import os
from pathlib import Path
import statsmodels.formula.api as smf
from sklearn.metrics import cohen_kappa_score

from core.config import Constants
from core.utils import Utils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


utils = Utils("question1/")

# -------------------------
# 1. Load human data
# -------------------------
human_df = utils.load_human_data()

print("=" * 60)
print("Human row-level data:")
print(human_df[["index", "article_id", "bias-question", "human_label"]].head())


# -------------------------
# 2. Load LLM results
# -------------------------
llm_df = utils.load_model_outputs(Constants.REASONING_CONDITION_PATHS["Direct"], "prompt_article_info.csv")

print("=" * 60)
print("LLM row-level data:")
print(llm_df.head())


# -------------------------
# 3. Merge human and LLM rows by index
# -------------------------
row_level_df = human_df.merge(
    llm_df,
    on="index",
    how="inner",
    suffixes=("_human", "_llm")
)

print("=" * 60)
print("Merged row-level dataset:")
print(row_level_df[[
    "Model",
    "index",
    "article_id_human",
    "article_id_llm",
    "human_label",
    "llm_label"
]].head())


# -------------------------
# 4. Check article_id consistency
# -------------------------
mismatch_count = (
    row_level_df["article_id_human"] != row_level_df["article_id_llm"]
).sum()

print("=" * 60)
print(f"Article ID mismatches after index merge: {mismatch_count}")

if mismatch_count > 0:
    print("WARNING: Some rows have matching index but different article_id.")
    print(row_level_df[
        row_level_df["article_id_human"] != row_level_df["article_id_llm"]
    ][[
        "index",
        "article_id_human",
        "article_id_llm",
        "Model"
    ]].head())


row_level_df["article_id"] = row_level_df["article_id_human"]


# -------------------------
# 5. Compute row-level alignment
# -------------------------
row_level_df["Aligned"] = (
    row_level_df["llm_label"] == row_level_df["human_label"]
).astype(int)

print("=" * 60)
print("Row-level alignment sample:")
print(row_level_df[[
    "Model",
    "index",
    "article_id",
    "human_label",
    "llm_label",
    "Aligned"
]].head())


# -------------------------
# 6. Add article-level human bias rate as context variable
# -------------------------
human_article_stats = (
    human_df.groupby("article_id")
    .agg(
        human_bias_rate=("human_label", "mean"),
        num_raters=("human_label", "count")
    )
    .reset_index()
)

row_level_df = row_level_df.merge(
    human_article_stats,
    on="article_id",
    how="left"
)


# -------------------------
# 7. Logistic regression: Model only
# -------------------------
print("=" * 60)
print("Running row-level logistic regression with only Model as predictor...")

model_simple = smf.logit(
    "Aligned ~ C(Model)",
    data=row_level_df
).fit()

print(model_simple.summary())


# -------------------------
# 8. Logistic regression: Model + human_bias_rate
# -------------------------
print("=" * 60)
print("Running row-level logistic regression with Model and human_bias_rate...")

model_with_human_bias_rate = smf.logit(
    "Aligned ~ C(Model) + human_bias_rate",
    data=row_level_df
).fit()

print(model_with_human_bias_rate.summary())


# -------------------------
# 9. Row-level model ranking
# -------------------------
ranking = (
    row_level_df.groupby("Model")["Aligned"]
    .mean()
    .sort_values(ascending=False)
)

print("=" * 60)
print("Row-level Model Ranking:")
print(ranking)


# -------------------------
# 10. Weighted ranking by number of human raters
# -------------------------
weighted_acc = (
    row_level_df.groupby("Model")
    .apply(lambda x: (x["Aligned"] * x["num_raters"]).sum() / x["num_raters"].sum())
    .sort_values(ascending=False)
)

print("=" * 60)
print("Row-level Weighted Model Ranking:")
print(weighted_acc)


# -------------------------
# 11. Kappa scores
# -------------------------
print("=" * 60)
print("Kappa scores for each model:")

kappa_scores = (
    row_level_df.groupby("Model")
    .apply(lambda x: cohen_kappa_score(
        x["llm_label"],
        x["human_label"]
    ))
    .sort_values(ascending=False)
)

print("=" * 60)
print("Row-level Model Ranking (Kappa):")
print(kappa_scores)


# -------------------------
# 12. Human label distribution
# -------------------------
print("\n=== Human row-level label distribution ===")
human_dist = row_level_df["human_label"].value_counts(normalize=True).sort_index()
print(human_dist)

print("\nCounts:")
print(row_level_df["human_label"].value_counts().sort_index())


# -------------------------
# 13. Save outputs
# -------------------------
# Save outputs via utils (handles directories)
utils.save_csv(row_level_df, "q1_row_level_merged.csv", index=False)
utils.save_csv(ranking, "q1_row_level_ranking.csv")
utils.save_csv(weighted_acc, "q1_row_level_weighted_ranking.csv")
utils.save_csv(kappa_scores, "q1_row_level_kappa.csv")

with open(utils.csv_dir.parent / "q1_row_level_model_only_summary.txt", "w") as f:
    f.write(str(model_simple.summary()))

with open(utils.csv_dir.parent / "q1_row_level_model_human_bias_summary.txt", "w") as f:
    f.write(str(model_with_human_bias_rate.summary()))

print("=" * 60)
print(f"Saved outputs to {utils.csv_dir.parent}")


# -------------------------
# 14. Paper-style plot settings
# -------------------------
plt.rcParams.update(Constants.PLOT_STYLE)


def style_ax(ax, xlabel="", ylabel=""):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(axis="both", length=3, width=0.8)

    plt.tight_layout(pad=0.4)


def add_barh_labels(ax, bars, fmt="{:.2f}", offset=0.006):
    for bar in bars:
        width = bar.get_width()
        if width >= 0:
            x_pos = width + offset
            ha = "left"
        else:
            x_pos = width - offset
            ha = "right"

        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(width),
            va="center",
            ha=ha,
            fontsize=8
        )

# -------------------------
# 15. Accuracy plot
# -------------------------
acc_df = ranking.reset_index()
acc_df.columns = ["Model", "Accuracy"]
acc_df["Model_Label"] = acc_df["Model"].map(Constants.MODEL_NAME_MAP).fillna(acc_df["Model"])
acc_df = acc_df.sort_values("Accuracy", ascending=True)

fig, ax = plt.subplots(figsize=(3.8, 2.8))

bars = ax.barh(
    acc_df["Model_Label"],
    acc_df["Accuracy"],
    color=Constants.COLORS["Default"],
    edgecolor=Constants.COLORS["Edge"],
    linewidth=0.7
)

add_barh_labels(ax, bars)

style_ax(
    ax,
    xlabel="Accuracy",
    ylabel=""
)

ax.set_xlim(0, max(acc_df["Accuracy"]) + 0.06)

utils.save_figure(fig, "q1_accuracy_paper")


# -------------------------
# 16. Weighted accuracy plot
# -------------------------
wacc_df = weighted_acc.reset_index()
wacc_df.columns = ["Model", "Weighted_Accuracy"]
wacc_df["Model_Label"] = wacc_df["Model"].map(Constants.MODEL_NAME_MAP).fillna(wacc_df["Model"])
wacc_df = wacc_df.sort_values("Weighted_Accuracy", ascending=True)

fig, ax = plt.subplots(figsize=(3.8, 2.8))

bars = ax.barh(
    wacc_df["Model_Label"],
    wacc_df["Weighted_Accuracy"],
    color=Constants.COLORS["Default"],
    edgecolor=Constants.COLORS["Edge"],
    linewidth=0.7
)

add_barh_labels(ax, bars)

style_ax(
    ax,
    xlabel="Weighted accuracy",
    ylabel=""
)

ax.set_xlim(0, max(wacc_df["Weighted_Accuracy"]) + 0.06)

utils.save_figure(fig, "q1_weighted_accuracy_paper")


# -------------------------
# 17. Kappa plot
# -------------------------
kap_df = kappa_scores.reset_index()
kap_df.columns = ["Model", "Kappa"]
kap_df["Model_Label"] = kap_df["Model"].map(Constants.MODEL_NAME_MAP).fillna(kap_df["Model"])
kap_df = kap_df.sort_values("Kappa", ascending=True)

fig, ax = plt.subplots(figsize=(3.8, 2.8))

bars = ax.barh(
    kap_df["Model_Label"],
    kap_df["Kappa"],
    color=Constants.COLORS["Default"],
    edgecolor=Constants.COLORS["Edge"],
    linewidth=0.7
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=0.8,
    color="black"
)

add_barh_labels(ax, bars, offset=0.004)

style_ax(
    ax,
    xlabel="Cohen's kappa",
    ylabel=""
)

xmin = min(kap_df["Kappa"].min() - 0.03, -0.05)
xmax = max(kap_df["Kappa"].max() + 0.03, 0.06)
ax.set_xlim(xmin, xmax)

utils.save_figure(fig, "q1_kappa_paper")

# -------------------------
# 18. Bias prediction rate plot
# -------------------------
# Bias prediction rate = proportion of rows where the model predicts "is-biased"
model_bias_rate = (
    row_level_df.groupby("Model")["llm_label"]
    .mean()
    .sort_values(ascending=False)
)

model_bias_df = model_bias_rate.reset_index()
model_bias_df.columns = ["Model", "Bias_Prediction_Rate"]
model_bias_df["Model_Label"] = (
    model_bias_df["Model"]
    .map(Constants.MODEL_NAME_MAP)
    .fillna(model_bias_df["Model"])
)

model_bias_df = model_bias_df.sort_values("Bias_Prediction_Rate", ascending=True)

fig, ax = plt.subplots(figsize=(3.8, 2.8))

bars = ax.barh(
    model_bias_df["Model_Label"],
    model_bias_df["Bias_Prediction_Rate"],
    color=Constants.COLORS["Default"],
    edgecolor=Constants.COLORS["Edge"],
    linewidth=0.7
)

add_barh_labels(ax, bars)

style_ax(
    ax,
    xlabel="Proportion predicted as biased",
    ylabel=""
)

ax.set_xlim(0, 1)

utils.save_figure(fig, "q1_bias_prediction_rate_paper")

utils.save_csv(
    model_bias_rate,
    "q1_row_level_bias_prediction_rate.csv",
    header=["bias_prediction_rate"]
)

print("\nSaved figures to:")
print(utils.figure_dir)

print("\nSaved CSV files to:")
print(utils.csv_dir)