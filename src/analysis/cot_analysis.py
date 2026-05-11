import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter

from core.utils import Utils
from core.config import Constants


# ============================================================
# CONFIG
# ============================================================
utils = Utils("question5")

CONDITION_PATHS = Constants.REASONING_CONDITION_PATHS
CONDITION_ORDER = Constants.REASONING_CONDITION_ORDER

PROMPT_FILE = "prompt_article_info.csv"


# ============================================================
# PLOT STYLE
# ============================================================
plt.rcParams.update(Constants.PLOT_STYLE)


# ============================================================
# HELPERS
# ============================================================
def load_condition_outputs(base_path, condition_name):
    # Use shared loader from Utils (handles normalization & label mapping)
    df = utils.load_model_outputs(base_path, PROMPT_FILE)
    if df.empty:
        return df
    df["Condition"] = condition_name
    return df


# ============================================================
# 1. LOAD HUMAN DATA
# ============================================================
human_row = utils.load_human_data()

print("\nHuman row-level data:")
print(human_row.head())


# ============================================================
# 2. LOAD MODEL OUTPUTS FOR ALL REASONING CONDITIONS
# ============================================================
all_conditions = []

for condition_name, base_path in CONDITION_PATHS.items():
    condition_df = load_condition_outputs(base_path, condition_name)

    if not condition_df.empty:
        all_conditions.append(condition_df)

all_llm = pd.concat(all_conditions, ignore_index=True)

utils.save_csv(
    all_llm,
    "q5_reasoning_all_outputs_long.csv",
    index=False
)

print("\nLoaded LLM outputs:")
print(all_llm.head())


# ============================================================
# 3. KEEP ONLY MATCHED ROWS ACROSS ALL CONDITIONS
# ============================================================
required_n_conditions = len(CONDITION_PATHS)

condition_counts = (
    all_llm.groupby(["Model", "article_id", "index"])["Condition"]
    .transform("nunique")
)

all_llm_matched = all_llm[condition_counts == required_n_conditions].copy()

print("\nMatched row-level subset:")
print("Rows:", len(all_llm_matched))
print("Unique articles:", all_llm_matched["article_id"].nunique())
print("Unique annotation rows:", all_llm_matched[["article_id", "index"]].drop_duplicates().shape[0])

print(
    all_llm_matched.groupby(["Model", "Condition"])
    .agg(
        n_rows=("index", "count"),
        n_articles=("article_id", "nunique")
    )
    .reset_index()
)


# ============================================================
# 4. BUILD ROW-LEVEL WIDE TABLE
# ============================================================
wide_parts = []

for condition in CONDITION_ORDER:
    tmp = all_llm_matched[
        all_llm_matched["Condition"] == condition
    ][[
        "Model",
        "article_id",
        "index",
        "llm_label",
        "llm_confidence"
    ]].copy()

    safe = condition.lower().replace(" ", "_")

    tmp = tmp.rename(columns={
        "llm_label": f"{safe}_output",
        "llm_confidence": f"{safe}_confidence"
    })

    wide_parts.append(tmp)

wide_df = wide_parts[0]

for part in wide_parts[1:]:
    wide_df = wide_df.merge(
        part,
        on=["Model", "article_id", "index"],
        how="inner"
    )

wide_df = wide_df.merge(
    human_row,
    on=["article_id", "index"],
    how="left"
)

wide_df = wide_df.dropna(subset=["human_label"]).copy()
wide_df["human_label"] = wide_df["human_label"].astype(int)

# Alignment columns
wide_df["direct_aligned"] = (
    wide_df["direct_output"] == wide_df["human_label"]
).astype(int)

wide_df["cot_aligned"] = (
    wide_df["cot_output"] == wide_df["human_label"]
).astype(int)

wide_df["chained_cot_aligned"] = (
    wide_df["chained_cot_output"] == wide_df["human_label"]
).astype(int)

utils.save_csv(wide_df, "q5_reasoning_row_level_wide.csv", index=False)

print("\nRow-level wide table:")
print(wide_df.head())

print("\nRows per model in wide table:")
print(wide_df.groupby("Model").size())


# ============================================================
# 5. PERFORMANCE SUMMARY BY CONDITION
# ============================================================
condition_output_cols = {
    "Direct": "direct_output",
    "CoT": "cot_output",
    "Chained CoT": "chained_cot_output",
}

condition_aligned_cols = {
    "Direct": "direct_aligned",
    "CoT": "cot_aligned",
    "Chained CoT": "chained_cot_aligned",
}

performance_rows = []

for model_name, sub in wide_df.groupby("Model"):
    for condition in CONDITION_ORDER:
        output_col = condition_output_cols[condition]
        aligned_col = condition_aligned_cols[condition]

        accuracy = sub[aligned_col].mean()
        kappa = utils.safe_kappa_pair(sub["human_label"], sub[output_col])
        bias_rate = sub[output_col].mean()
        mean_confidence = sub[f"{condition.lower().replace(' ', '_')}_confidence"].mean()

        performance_rows.append({
            "Model": model_name,
            "Model_Display": utils.pretty_model_name(model_name),
            "Condition": condition,
            "N": len(sub),
            "Accuracy": accuracy,
            "Kappa": kappa,
            "Bias_Prediction_Rate": bias_rate,
            "Mean_Confidence": mean_confidence,
        })

performance_df = pd.DataFrame(performance_rows)

utils.save_csv(performance_df, "q5_reasoning_performance_by_condition_row_level.csv", index=False)

print("\n=== Row-level performance by condition ===")
print(performance_df)


# ============================================================
# 6. DELTA PERFORMANCE VS DIRECT AND BETWEEN COT CONDITIONS
# ============================================================
delta_rows = []

for model_name, sub in wide_df.groupby("Model"):
    direct_acc = sub["direct_aligned"].mean()
    cot_acc = sub["cot_aligned"].mean()
    chained_acc = sub["chained_cot_aligned"].mean()

    direct_kappa = utils.safe_kappa_pair(sub["human_label"], sub["direct_output"])
    cot_kappa = utils.safe_kappa_pair(sub["human_label"], sub["cot_output"])
    chained_kappa = utils.safe_kappa_pair(sub["human_label"], sub["chained_cot_output"])

    direct_bias = sub["direct_output"].mean()
    cot_bias = sub["cot_output"].mean()
    chained_bias = sub["chained_cot_output"].mean()

    comparisons = [
        (Constants.REASONING_COMPARISONS["cot_vs_direct"], cot_acc, direct_acc, cot_kappa, direct_kappa, cot_bias, direct_bias),
        (Constants.REASONING_COMPARISONS["chained_vs_direct"], chained_acc, direct_acc, chained_kappa, direct_kappa, chained_bias, direct_bias),
        (Constants.REASONING_COMPARISONS["chained_vs_cot"], chained_acc, cot_acc, chained_kappa, cot_kappa, chained_bias, cot_bias),
    ]

    for comp, after_acc, before_acc, after_kappa, before_kappa, after_bias, before_bias in comparisons:
        delta_rows.append({
            "Model": model_name,
            "Model_Display": utils.pretty_model_name(model_name),
            "Comparison": comp,
            "Before_Accuracy": before_acc,
            "After_Accuracy": after_acc,
            "Delta_Accuracy": after_acc - before_acc,
            "Before_Kappa": before_kappa,
            "After_Kappa": after_kappa,
            "Delta_Kappa": after_kappa - before_kappa,
            "Before_Bias_Rate": before_bias,
            "After_Bias_Rate": after_bias,
            "Delta_Bias_Rate": after_bias - before_bias,
        })

delta_df = pd.DataFrame(delta_rows)

utils.save_csv(delta_df, "q5_reasoning_delta_metrics_row_level.csv", index=False)

print("\n=== Delta metrics ===")
print(delta_df)


# ============================================================
# 7. FLIP ANALYSIS + MCNEMAR TESTS
# ============================================================
prediction_comparisons = {
    Constants.REASONING_COMPARISONS["cot_vs_direct"]: ("direct_output", "cot_output"),
    Constants.REASONING_COMPARISONS["chained_vs_direct"]: ("direct_output", "chained_cot_output"),
    Constants.REASONING_COMPARISONS["chained_vs_cot"]: ("cot_output", "chained_cot_output"),
}

alignment_comparisons = {
    Constants.REASONING_COMPARISONS["cot_vs_direct"]: ("direct_aligned", "cot_aligned"),
    Constants.REASONING_COMPARISONS["chained_vs_direct"]: ("direct_aligned", "chained_cot_aligned"),
    Constants.REASONING_COMPARISONS["chained_vs_cot"]: ("cot_aligned", "chained_cot_aligned"),
}

flip_results = []
prediction_mcnemar_results = []
alignment_change_results = []
alignment_mcnemar_results = []

for comp_name, (before_col, after_col) in prediction_comparisons.items():
    flip_results.append(
        utils.summarize_prediction_flips(
            wide_df,
            before_col,
            after_col,
            context_fields={"Comparison": comp_name},
        )
    )

    prediction_mcnemar_results.append(
        utils.run_mcnemar_by_model(
            wide_df,
            before_col,
            after_col,
            context_fields={"Comparison": comp_name},
            test_type="Prediction change",
        )
    )

for comp_name, (before_col, after_col) in alignment_comparisons.items():
    alignment_change_results.append(
        utils.summarize_alignment_change(
            wide_df,
            before_col,
            after_col,
            context_fields={"Comparison": comp_name},
        )
    )

    alignment_mcnemar_results.append(
        utils.run_mcnemar_by_model(
            wide_df,
            before_col,
            after_col,
            context_fields={"Comparison": comp_name},
            test_type="Alignment change",
        )
    )

flip_df = pd.concat(flip_results, ignore_index=True)
prediction_mcnemar_df = pd.concat(prediction_mcnemar_results, ignore_index=True)
alignment_change_df = pd.concat(alignment_change_results, ignore_index=True)
alignment_mcnemar_df = pd.concat(alignment_mcnemar_results, ignore_index=True)

all_mcnemar_df = pd.concat(
    [prediction_mcnemar_df, alignment_mcnemar_df],
    ignore_index=True
)

utils.save_csv(flip_df, "q5_reasoning_prediction_flip_summary_row_level.csv", index=False)
utils.save_csv(prediction_mcnemar_df, "q5_reasoning_mcnemar_prediction_change_row_level.csv", index=False)
utils.save_csv(alignment_change_df, "q5_reasoning_alignment_change_summary_row_level.csv", index=False)
utils.save_csv(alignment_mcnemar_df, "q5_reasoning_mcnemar_alignment_change_row_level.csv", index=False)
utils.save_csv(all_mcnemar_df, "q5_reasoning_all_mcnemar_tests_row_level.csv", index=False)

print("\n=== Prediction flip summary ===")
print(flip_df)

print("\n=== McNemar prediction-change tests ===")
print(prediction_mcnemar_df)

print("\n=== Alignment change summary ===")
print(alignment_change_df)

print("\n=== McNemar alignment-change tests ===")
print(alignment_mcnemar_df)


# ============================================================
# 8. OVERALL SUMMARY TABLES
# ============================================================
overall_performance = (
    performance_df.groupby("Condition")
    .agg(
        Mean_Accuracy=("Accuracy", "mean"),
        Mean_Kappa=("Kappa", "mean"),
        Mean_Bias_Prediction_Rate=("Bias_Prediction_Rate", "mean"),
        Mean_Confidence=("Mean_Confidence", "mean"),
    )
    .reindex(CONDITION_ORDER)
    .reset_index()
)

overall_flip = (
    flip_df.groupby("Comparison")
    .agg(
        Mean_Flip_Rate=("Flip_Rate", "mean"),
        Mean_To_Biased_Rate=("To_Biased_Rate", "mean"),
        Mean_To_Not_Biased_Rate=("To_Not_Biased_Rate", "mean"),
    )
    .reindex(Constants.REASONING_COMPARISON_ORDER)
    .reset_index()
)

overall_delta = (
    delta_df.groupby("Comparison")
    .agg(
        Mean_Delta_Accuracy=("Delta_Accuracy", "mean"),
        Mean_Delta_Kappa=("Delta_Kappa", "mean"),
        Mean_Delta_Bias_Rate=("Delta_Bias_Rate", "mean"),
    )
    .reindex(Constants.REASONING_COMPARISON_ORDER)
    .reset_index()
)

utils.save_csv(overall_performance, "q5_reasoning_overall_performance_row_level.csv", index=False)
utils.save_csv(overall_flip, "q5_reasoning_overall_flip_summary_row_level.csv", index=False)
utils.save_csv(overall_delta, "q5_reasoning_overall_delta_metrics_row_level.csv", index=False)


# ============================================================
# 9. FIGURES
# ============================================================

# ------------------------------------------------------------
# Figure 1: Average prediction flip rate by comparison
# ------------------------------------------------------------
avg_flip = (
    flip_df.groupby("Comparison")["Flip_Rate"]
    .mean()
    .reindex(Constants.REASONING_COMPARISON_ORDER)
)

fig, ax = plt.subplots(figsize=(6.5, 4.2))

bars = ax.bar(
    avg_flip.index,
    avg_flip.values,
    color=[Constants.COLORS[x] for x in avg_flip.index],
    edgecolor="black",
    linewidth=0.7,
    width=0.65
)

for bar, value in zip(bars, avg_flip.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value * 100:.1f}%",
        ha="center",
        va="bottom",
    )

ax.set_ylabel("Prediction Flip Rate")
ax.set_title("")
ax.set_ylim(0, max(0.5, avg_flip.max() + 0.08))
ax.set_xticklabels(avg_flip.index, rotation=20, ha="right")

plt.tight_layout()
utils.save_figure(fig, "q5_avg_prediction_flip_rate_row_level")


# ------------------------------------------------------------
# Figure 2: Directional flips
# ------------------------------------------------------------
direction_overall = (
    flip_df.groupby("Comparison")[["To_Biased_Rate", "To_Not_Biased_Rate"]]
    .mean()
    .reindex(Constants.REASONING_COMPARISON_ORDER)
)

fig, ax = plt.subplots(figsize=(6.5, 4.4))

x = np.arange(len(direction_overall.index))

ax.bar(
    x,
    direction_overall["To_Biased_Rate"],
    label="Not biased → Biased",
    color=Constants.COLORS["Not biased → Biased"],
    edgecolor="black",
    linewidth=0.7
)

ax.bar(
    x,
    direction_overall["To_Not_Biased_Rate"],
    bottom=direction_overall["To_Biased_Rate"],
    label="Biased → Not biased",
    color=Constants.COLORS["Biased → Not biased"],
    edgecolor="black",
    linewidth=0.7
)

for i, row in enumerate(direction_overall.itertuples()):
    total = row.To_Biased_Rate + row.To_Not_Biased_Rate
    ax.text(
        i,
        total + 0.01,
        f"{total * 100:.1f}%",
        ha="center",
        va="bottom",
    )

ax.set_xticks(x)
ax.set_xticklabels(direction_overall.index, rotation=20, ha="right")
ax.set_ylabel("Directional flip rates")
ax.set_title("")
ax.legend(frameon=True)
ax.set_ylim(0, max(0.5, direction_overall.sum(axis=1).max() + 0.08))
ax.yaxis.set_major_formatter(PercentFormatter(1.0))

plt.tight_layout()
utils.save_figure(fig, "q5_directional_prediction_flips_row_level")


# ------------------------------------------------------------
# Figure 3: Accuracy by condition, per model
# ------------------------------------------------------------
accuracy_plot = (
    performance_df.pivot(index="Model_Display", columns="Condition", values="Accuracy")
    .reindex(columns=CONDITION_ORDER)
)

accuracy_plot = accuracy_plot.loc[
    accuracy_plot.mean(axis=1).sort_values(ascending=True).index
]

fig, ax = plt.subplots(figsize=(7.2, 5.2))

accuracy_plot.plot(
    kind="barh",
    ax=ax,
    color=[Constants.COLORS[c] for c in CONDITION_ORDER],
    edgecolor="black",
    linewidth=0.6,
    width=0.8
)

ax.set_xlabel("Accuracy")
ax.set_ylabel("")
ax.set_title("")
ax.legend(title="", loc=Constants.LEGEND_LOCATIONS["lower_right"])
ax.set_xlim(0, max(0.75, accuracy_plot.max().max() + 0.05))

plt.tight_layout()
utils.save_figure(fig, "q5_accuracy_by_reasoning_condition_row_level")


# ------------------------------------------------------------
# Figure 4: Kappa by condition, per model
# ------------------------------------------------------------
kappa_plot = (
    performance_df.pivot(index="Model_Display", columns="Condition", values="Kappa")
    .reindex(columns=CONDITION_ORDER)
)

kappa_plot = kappa_plot.loc[
    kappa_plot.mean(axis=1).sort_values(ascending=True).index
]

fig, ax = plt.subplots(figsize=(7.2, 5.2))

kappa_plot.plot(
    kind="barh",
    ax=ax,
    color=[Constants.COLORS[c] for c in CONDITION_ORDER],
    edgecolor="black",
    linewidth=0.6,
    width=0.8
)

ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel("Cohen's kappa")
ax.set_ylabel("")
ax.set_title("")
ax.legend(title="", loc=Constants.LEGEND_LOCATIONS["lower_right"])

plt.tight_layout()
utils.save_figure(fig, "q5_kappa_by_reasoning_condition_row_level")


# ------------------------------------------------------------
# Figure 5: Delta kappa by comparison + significance stars
# ------------------------------------------------------------

delta_kappa_plot = (
    delta_df.pivot(
        index="Model_Display",
        columns="Comparison",
        values="Delta_Kappa"
    )
    .reindex(columns=Constants.REASONING_COMPARISON_ORDER)
)

delta_kappa_plot = delta_kappa_plot.loc[
    delta_kappa_plot["Chained CoT vs Direct"]
    .sort_values(ascending=True)
    .index
]

# ------------------------------------------------------------
# Build significance lookup
# ------------------------------------------------------------

sig_lookup = (
    alignment_mcnemar_df[
        ["Model_Display", "Comparison", "significance"]
    ]
    .set_index(["Model_Display", "Comparison"])
    ["significance"]
    .to_dict()
)

fig, ax = plt.subplots(figsize=(8.0, 5.8))

delta_kappa_plot.plot(
    kind="barh",
    ax=ax,
    color=[Constants.COLORS[c] for c in delta_kappa_plot.columns],
    edgecolor="black",
    linewidth=0.6,
    width=0.82
)

ax.axvline(0, color="black", linewidth=1, linestyle="--")

# Axis labels
ax.set_xlabel(
    r"$\Delta$ Cohen's kappa",
)

ax.set_title("")
ax.set_ylabel("")

# Tick label sizes
ax.tick_params(
    axis="x",
    labelsize=14
)

# Legend
ax.legend(
    title="",
    loc=Constants.LEGEND_LOCATIONS["lower_right"],
)

# ------------------------------------------------------------
# Add significance stars
# ------------------------------------------------------------

comparisons = list(delta_kappa_plot.columns)

for container, comparison in zip(ax.containers, comparisons):

    for bar, model_name in zip(container, delta_kappa_plot.index):

        star = sig_lookup.get((model_name, comparison), "")

        # skip non-significant
        if star == "ns" or star == "NA":
            continue

        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        # place star slightly outside bar
        offset = 0.008

        if width >= 0:
            x = width + offset
            ha = "left"
        else:
            x = width - offset
            ha = "right"

        ax.text(
            x,
            y,
            star,
            va="center",
            ha=ha,
            fontweight="bold"
        )

# nicer limits
xmin = delta_kappa_plot.min().min()
xmax = delta_kappa_plot.max().max()

ax.set_xlim(
    xmin - 0.05,
    xmax + 0.07
)

plt.tight_layout()

utils.save_figure(
    fig,
    "q5_delta_kappa_by_reasoning_condition_row_level"
)


# ------------------------------------------------------------
# Figure 6: Delta accuracy by comparison
# ------------------------------------------------------------
delta_acc_plot = (
    delta_df.pivot(index="Model_Display", columns="Comparison", values="Delta_Accuracy")
    .reindex(columns=Constants.REASONING_COMPARISON_ORDER)
)

delta_acc_plot = delta_acc_plot.loc[
    delta_acc_plot["Chained CoT vs Direct"].sort_values(ascending=True).index
]

fig, ax = plt.subplots(figsize=(7.2, 5.2))

delta_acc_plot.plot(
    kind="barh",
    ax=ax,
    color=[Constants.COLORS[c] for c in delta_acc_plot.columns],
    edgecolor="black",
    linewidth=0.6,
    width=0.8
)

ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel("Δ accuracy")
ax.set_ylabel("")
ax.set_title("")
ax.legend(title="", loc=Constants.LEGEND_LOCATIONS["lower_right"])

plt.tight_layout()
utils.save_figure(fig, "q5_delta_accuracy_by_reasoning_condition_row_level")


# ------------------------------------------------------------
# Figure 7: Bias prediction rate by condition
# ------------------------------------------------------------
bias_plot = (
    performance_df.pivot(index="Model_Display", columns="Condition", values="Bias_Prediction_Rate")
    .reindex(columns=CONDITION_ORDER)
)

bias_plot = bias_plot.loc[
    bias_plot.mean(axis=1).sort_values(ascending=True).index
]

fig, ax = plt.subplots(figsize=(7.2, 5.2))

bias_plot.plot(
    kind="barh",
    ax=ax,
    color=[Constants.COLORS[c] for c in CONDITION_ORDER],
    edgecolor="black",
    linewidth=0.6,
    width=0.8
)

ax.set_xlabel("Proportion predicted biased")
ax.set_ylabel("")
ax.set_title("")
ax.legend(title="", loc=Constants.LEGEND_LOCATIONS["lower_right"])
ax.set_xlim(0, 1)

plt.tight_layout()
utils.save_figure(fig, "q5_bias_prediction_rate_by_reasoning_condition_row_level")