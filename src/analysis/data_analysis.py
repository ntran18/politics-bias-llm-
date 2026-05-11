from pathlib import Path

from core.config import Constants
from core.utils import Utils

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

matplotlib.use("Agg")

utils = Utils("dataset")


# =========================================================
# Constants
# =========================================================
BAR_WIDTH = 0.65

# =========================================================
# Matplotlib style
# =========================================================
plt.rcParams.update(Constants.PLOT_STYLE)


# =========================================================
# Load dataset
# =========================================================
human_df = utils.load_human_data()


# =========================================================
# 1. Distribution of raters per article
# =========================================================
voters_per_article = (
    human_df.groupby("article_id")["human_label"]
    .count()
)

fig, ax = plt.subplots(figsize=(3.35, 2.4))

ax.hist(
    voters_per_article,
    bins=15,
    color=Constants.COLORS["Default"],
    edgecolor=Constants.COLORS["Edge"],
    linewidth=0.6,
)

utils.finalize_plot(
    ax,
    xlabel="Number of Raters",
    ylabel="Number of Articles",
)

utils.save_figure(fig, "voters_distribution")


# =========================================================
# 2. Human label distribution
# =========================================================
rating_counts = (
    human_df["human_label"]
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(2.5, 2.5))

colors = [
    Constants.COLORS["Not Biased"],
    Constants.COLORS["Biased"]
]

utils.plot_pie_chart(fig, ax, rating_counts, Constants.BIAS_LABELS_DESCRIPTIONS, 2, colors, "human_label_distribution")

# =========================================================
# 3. Article-level majority distribution
# =========================================================
article_level = (
    human_df.groupby("article_id")["human_label"]
    .mean()
    .reset_index(name="human_bias_rate")
)

article_level["human_majority"] = (
    article_level["human_bias_rate"] > 0.5
).astype(int)

majority_counts = (
    article_level["human_majority"]
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(2.4, 2.4))

bars = ax.bar(
    Constants.BIAS_LABELS_DESCRIPTIONS,
    majority_counts.values,
    color=Constants.COLORS["Default"],
    edgecolor=Constants.COLORS["Edge"],
    linewidth=0.6,
    width=BAR_WIDTH,
)

utils.add_bar_percent_labels(ax, bars, majority_counts.values)

utils.finalize_plot(
    ax,
    ylabel="Number of Articles",
)

utils.save_figure(fig, "article_majority_distribution")


# =========================================================
# 4. Article-level bias rate distribution
# =========================================================
fig, ax = plt.subplots(figsize=(3.35, 2.4))

ax.hist(
    article_level["human_bias_rate"],
    bins=12,
    color=Constants.COLORS["Default"],
    edgecolor=Constants.COLORS["Edge"],
    linewidth=0.6,
)

utils.finalize_plot(
    ax,
    xlabel="Proportion labeled biased",
    ylabel="Number of Articles",
)

utils.save_figure(fig, "article_bias_rate_distribution")


# =========================================================
# 5. Human consensus distribution
# =========================================================
consensus_df = (
    human_df.groupby("article_id")
    .agg(
        num_raters=("human_label", "count"),
        bias_rate=("human_label", "mean"),
    )
    .reset_index()
)

consensus_df = consensus_df[
    consensus_df["num_raters"] >= 2
].copy()

consensus_df["consensus_rate"] = (
    consensus_df["bias_rate"]
    .apply(lambda x: max(x, 1 - x))
)

fig, ax = plt.subplots(figsize=(3.35, 2.4))

ax.hist(
    consensus_df["consensus_rate"],
    bins=10,
    color=Constants.COLORS["Default"],
    edgecolor=Constants.COLORS["Edge"],
    linewidth=0.6,
)

ax.set_xlim(0.5, 1.0)

utils.finalize_plot(
    ax,
    xlabel="Proportion agreeing with majority label",
    ylabel="Number of Articles",
)

utils.save_figure(fig, "human_consensus_distribution")


# =========================================================
# 6. Consensus category distribution
# =========================================================
def consensus_category(x):
    if x == 0.5:
        return "Tie"

    if x < 0.60:
        return "Low"

    if x < 0.80:
        return "Moderate"

    return "High"


consensus_df["consensus_category"] = (
    consensus_df["consensus_rate"]
    .apply(consensus_category)
)

category_order = [
    "Tie",
    "Low",
    "Moderate",
    "High",
]

consensus_counts = (
    consensus_df["consensus_category"]
    .value_counts()
    .reindex(category_order)
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(3.35, 2.4))

colors = [
    "#D9D9D9",  # Tie 
    "#C7D3E3",  # Low consensus
    "#7FA6D6",  # Moderate consensus
    "#2F5D9B",  # High consensus
]

utils.plot_pie_chart(fig, ax, consensus_counts, category_order, 2, colors, "human_consensus_categories")


# =========================================================
# 7. Political group distribution
# =========================================================
politics_clean = (
    human_df["politics"]
    .fillna("No response")
    .astype(str)
    .str.strip()
)

politics_clean = politics_clean.replace(
    {
        "": "No response",
        "nan": "No response",
        "NaN": "No response",
        "None": "No response",
    }
)

politics_order = [
    "Conservative",
    "Liberal",
    "Independent",
    "No response",
]

politics_counts = (
    politics_clean
    .value_counts()
    .reindex(politics_order)
    .dropna()
)

fig, ax = plt.subplots(figsize=(3.35, 2.4))

bars = ax.bar(
    politics_counts.index,
    politics_counts.values,
    color=[
        Constants.COLORS.get(x, "#999999")
        for x in politics_counts.index
    ],
    edgecolor=Constants.COLORS["Edge"],
    linewidth=0.6,
    width=BAR_WIDTH,
)

utils.add_bar_percent_labels(ax, bars, politics_counts.values)

utils.finalize_plot(
    ax,
    xlabel="",
    ylabel="Number of Annotations",
    rotate_x=30,
)

utils.save_figure(fig, "political_group_distribution")


# =========================================================
# 8. Human bias rate by political group
# =========================================================
politics_df = human_df.copy()

politics_df["politics_clean"] = politics_clean

politics_groups = [
    "Conservative",
    "Liberal",
    "Independent",
]

human_bias_by_politics = (
    politics_df[
        politics_df["politics_clean"].isin(
            politics_groups
        )
    ]
    .groupby("politics_clean")["human_label"]
    .mean()
    .reindex(politics_groups)
)

fig, ax = plt.subplots(figsize=(3.35, 2.4))

bars = ax.bar(
    human_bias_by_politics.index,
    human_bias_by_politics.values,
    color=Constants.COLORS["Default"],
    edgecolor=Constants.COLORS["Edge"],
    linewidth=0.6,
    width=BAR_WIDTH,
)

for bar, value in zip(
    bars,
    human_bias_by_politics.values,
):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.1%}",
        ha="center",
        va="bottom",
    )

ax.set_ylim(0, 0.85)
ax.yaxis.set_major_formatter(
    PercentFormatter(1.0, decimals=0)
)

utils.finalize_plot(
    ax,
    ylabel="Human Bias Detection Rate"
)

utils.save_figure(fig, "human_bias_rate_by_politics")


# =========================================================
# Save CSV outputs
# =========================================================
utils.save_csv(
    voters_per_article.describe(),
    "voters_summary.csv",
)

utils.save_csv(
    article_level,
    "article_level_bias_rates.csv",
    index=False,
)

utils.save_csv(
    consensus_df,
    "human_consensus_by_article.csv",
    index=False,
)

utils.save_csv(
    consensus_counts,
    "human_consensus_category_counts.csv",
    header=["count"],
)

utils.save_csv(
    politics_counts,
    "political_group_counts.csv",
    header=["count"],
)

utils.save_csv(
    human_bias_by_politics,
    "human_bias_rate_by_politics.csv",
    header=["bias_rate"],
)


print("\nSaved figures to:")
print(utils.figure_dir)

print("\nSaved CSV files to:")
print(utils.csv_dir)