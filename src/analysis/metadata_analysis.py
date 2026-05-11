from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter

from core.config import Constants
from core.utils import Utils

# Use utils for IO, normalization, and helpers
utils = Utils("question3/question3_metadata_row_level")
FIG_DIR = utils.figure_dir
CSV_DIR = utils.csv_dir

# Apply shared plot style
plt.rcParams.update(Constants.PLOT_STYLE)
PROMPT_FILES = {
    "article_only": "prompt_article_info.csv",
    "source": "prompt_source_variants.csv",
    "politics": "prompt_politics_variants.csv",
    "source_politics": "prompt_source_politics_variants.csv",
    "politics_demographic": "prompt_politics_pii_variants.csv",
    "source_demographic": "prompt_source_pii_variants.csv",
    "source_politics_demographic": "prompt_pii_combined_variants.csv",
}

METADATA_ORDER = [
    "source",
    "politics",
    "source_politics",
    "source_demographic",
    "politics_demographic",
    "source_politics_demographic",
]

METADATA_LABELS = {
    "source": "Source",
    "politics": "Politics",
    "source_politics": "Source + Politics",
    "source_demographic": "Source + Demographic",
    "politics_demographic": "Politics + Demographic",
    "source_politics_demographic": "Source + Politics + Demographic",
}


# ============================================================
# 1. LOAD HUMAN DATA + METADATA
# ============================================================
human_df = utils.load_human_data()

# Keep only the row-level human columns used later
human_row = human_df[["article_id", "index", "human_label", "source", "politics", "age", "gender"]].copy()

print("\nHuman metadata sample:")
print(human_row.head())


# ============================================================
# 2. LOAD ALL MODEL OUTPUTS
# ============================================================
BASE_PATH = Path("../../results/v8/v8.1/")

all_rows = []

for prompt_name, filename in PROMPT_FILES.items():
    df = utils.load_model_outputs(BASE_PATH, filename)
    if df.empty:
        continue
    df["Prompt"] = prompt_name
    all_rows.append(df)

if len(all_rows) > 0:
    all_llm = pd.concat(all_rows, ignore_index=True)
else:
    all_llm = pd.DataFrame()

all_llm = all_llm.merge(human_row, on=["article_id", "index"], how="left")
all_llm = all_llm.dropna(subset=["human_label"]).copy()
all_llm["human_label"] = all_llm["human_label"].astype(int)

utils.save_csv(all_llm, "q3_metadata_all_outputs_long_row_level.csv", index=False)

print("\nLoaded LLM row-level outputs:")
print(all_llm.head())


# ============================================================
# 3. BUILD ROW-LEVEL BASELINE
# ============================================================
baseline = all_llm[all_llm["Prompt"] == "article_only"][
    [
        "Model",
        "article_id",
        "index",
        "llm_label",
        "llm_confidence",
        "human_label",
        "source",
        "politics",
        "age",
        "gender",
    ]
].copy()

baseline = baseline.rename(columns={
    "llm_label": "article_only_output",
    "llm_confidence": "article_only_confidence",
})

baseline["article_only_aligned"] = (
    baseline["article_only_output"] == baseline["human_label"]
).astype(int)

baseline.to_csv(
    CSV_DIR / "q3_metadata_article_only_baseline_row_level.csv",
    index=False
)


# ============================================================
# 4. COMPARE EACH METADATA CONDITION AGAINST ARTICLE-ONLY
# ============================================================
compare_frames = []
flip_frames = []
alignment_frames = []
prediction_mcnemar_frames = []
alignment_mcnemar_frames = []
delta_rows = []

for metadata_condition in METADATA_ORDER:
    prompt_df = all_llm[all_llm["Prompt"] == metadata_condition][
        [
            "Model",
            "article_id",
            "index",
            "llm_label",
            "llm_confidence",
        ]
    ].copy()

    prompt_df = prompt_df.rename(columns={
        "llm_label": "metadata_output",
        "llm_confidence": "metadata_confidence",
    })

    compare = baseline.merge(
        prompt_df,
        on=["Model", "article_id", "index"],
        how="inner"
    )

    compare["Metadata_Condition"] = metadata_condition
    compare["Metadata_Label"] = METADATA_LABELS.get(metadata_condition, metadata_condition)

    compare["metadata_aligned"] = (
        compare["metadata_output"] == compare["human_label"]
    ).astype(int)

    compare["flipped"] = (
        compare["article_only_output"] != compare["metadata_output"]
    ).astype(int)

    compare["to_biased"] = (
        (compare["article_only_output"] == 0) &
        (compare["metadata_output"] == 1)
    ).astype(int)

    compare["to_not_biased"] = (
        (compare["article_only_output"] == 1) &
        (compare["metadata_output"] == 0)
    ).astype(int)

    compare_frames.append(compare)

    # Summary tables
    flip_frames.append(
        utils.summarize_prediction_flips(
            compare,
            "article_only_output",
            "metadata_output",
            context_fields={
                "Metadata_Condition": metadata_condition,
                "Metadata_Label": METADATA_LABELS.get(metadata_condition, metadata_condition),
            },
        )
    )

    alignment_frames.append(
        utils.summarize_alignment_change(
            compare,
            "article_only_aligned",
            "metadata_aligned",
            context_fields={
                "Metadata_Condition": metadata_condition,
                "Metadata_Label": METADATA_LABELS.get(metadata_condition, metadata_condition),
            },
        )
    )

    # McNemar prediction-change test
    prediction_mcnemar_frames.append(
        utils.run_mcnemar_by_model(
            compare,
            before_col="article_only_output",
            after_col="metadata_output",
            context_fields={
                "Metadata_Condition": metadata_condition,
                "Metadata_Label": METADATA_LABELS.get(metadata_condition, metadata_condition),
            },
            test_type="Prediction change"
        )
    )

    # McNemar alignment-change test
    alignment_mcnemar_frames.append(
        utils.run_mcnemar_by_model(
            compare,
            before_col="article_only_aligned",
            after_col="metadata_aligned",
            context_fields={
                "Metadata_Condition": metadata_condition,
                "Metadata_Label": METADATA_LABELS.get(metadata_condition, metadata_condition),
            },
            test_type="Alignment change"
        )
    )

    # Delta metrics
    for model_name, sub in compare.groupby("Model"):
        base_acc = sub["article_only_aligned"].mean()
        meta_acc = sub["metadata_aligned"].mean()

        base_kappa = utils.safe_kappa_pair(sub["human_label"], sub["article_only_output"])
        meta_kappa = utils.safe_kappa_pair(sub["human_label"], sub["metadata_output"])

        base_bias = sub["article_only_output"].mean()
        meta_bias = sub["metadata_output"].mean()

        delta_rows.append({
            "Model": model_name,
            "Model_Display": utils.pretty_model_name(model_name),
            "Metadata_Condition": metadata_condition,
            "Metadata_Label": METADATA_LABELS.get(metadata_condition, metadata_condition),
            "N": len(sub),
            "Baseline_Accuracy": base_acc,
            "Metadata_Accuracy": meta_acc,
            "Delta_Accuracy": meta_acc - base_acc,
            "Baseline_Kappa": base_kappa,
            "Metadata_Kappa": meta_kappa,
            "Delta_Kappa": meta_kappa - base_kappa,
            "Baseline_Bias_Rate": base_bias,
            "Metadata_Bias_Rate": meta_bias,
            "Delta_Bias_Rate": meta_bias - base_bias,
        })


compare_df = pd.concat(compare_frames, ignore_index=True)
flip_df = pd.concat(flip_frames, ignore_index=True)
alignment_change_df = pd.concat(alignment_frames, ignore_index=True)
prediction_mcnemar_df = pd.concat(prediction_mcnemar_frames, ignore_index=True)
alignment_mcnemar_df = pd.concat(alignment_mcnemar_frames, ignore_index=True)
delta_df = pd.DataFrame(delta_rows)

all_mcnemar_df = pd.concat(
    [prediction_mcnemar_df, alignment_mcnemar_df],
    ignore_index=True
)

utils.save_csv(compare_df, "q3_metadata_compare_to_article_only_row_level.csv", index=False)
utils.save_csv(flip_df, "q3_metadata_prediction_flip_summary_row_level.csv", index=False)
utils.save_csv(alignment_change_df, "q3_metadata_alignment_change_summary_row_level.csv", index=False)
utils.save_csv(prediction_mcnemar_df, "q3_metadata_mcnemar_prediction_change_row_level.csv", index=False)
utils.save_csv(alignment_mcnemar_df, "q3_metadata_mcnemar_alignment_change_row_level.csv", index=False)
utils.save_csv(all_mcnemar_df, "q3_metadata_all_mcnemar_tests_row_level.csv", index=False)
utils.save_csv(delta_df, "q3_metadata_delta_metrics_row_level.csv", index=False)


# ============================================================
# 5. OVERALL SUMMARY TABLES
# ============================================================
overall_flip = (
    flip_df.groupby(["Metadata_Condition", "Metadata_Label"]) 
    .agg(
        Mean_Flip_Rate=("Flip_Rate", "mean"),
        Mean_To_Biased_Rate=("To_Biased_Rate", "mean"),
        Mean_To_Not_Biased_Rate=("To_Not_Biased_Rate", "mean"),
    )
    .reset_index()
)

overall_delta = (
    delta_df.groupby(["Metadata_Condition", "Metadata_Label"]) 
    .agg(
        Mean_Delta_Accuracy=("Delta_Accuracy", "mean"),
        Mean_Delta_Kappa=("Delta_Kappa", "mean"),
        Mean_Delta_Bias_Rate=("Delta_Bias_Rate", "mean"),
    )
    .reset_index()
)

overall_flip["Metadata_Condition"] = pd.Categorical(
    overall_flip["Metadata_Condition"],
    categories=METADATA_ORDER,
    ordered=True
)

overall_delta["Metadata_Condition"] = pd.Categorical(
    overall_delta["Metadata_Condition"],
    categories=METADATA_ORDER,
    ordered=True
)

overall_flip = overall_flip.sort_values("Metadata_Condition")
overall_delta = overall_delta.sort_values("Metadata_Condition")

utils.save_csv(overall_flip, "q3_metadata_overall_flip_summary_row_level.csv", index=False)
utils.save_csv(overall_delta, "q3_metadata_overall_delta_summary_row_level.csv", index=False)

print("\n=== Overall metadata flip summary ===")
print(overall_flip)

print("\n=== Overall metadata delta summary ===")
print(overall_delta)


# ============================================================
# 6. FIGURES
# ============================================================

# ------------------------------------------------------------
# Figure 1: Average metadata flip rate
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.5))

plot_flip = overall_flip.set_index("Metadata_Condition").reindex(METADATA_ORDER).reset_index()

bars = ax.bar(
    plot_flip["Metadata_Label"],
    plot_flip["Mean_Flip_Rate"],
    color=[Constants.COLORS[c] for c in plot_flip["Metadata_Condition"]],
    edgecolor="black",
    linewidth=0.7,
    width=0.65
)

for bar, value in zip(bars, plot_flip["Mean_Flip_Rate"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12
    )

ax.set_yticks([])
ax.set_ylabel("")
ax.set_title("")
ax.set_ylim(0, max(0.4, plot_flip["Mean_Flip_Rate"].max() + 0.08))
ax.set_xticklabels(plot_flip["Metadata_Label"], rotation=25, ha="right", fontsize=12)

plt.tight_layout()
utils.save_figure(fig, "q3_metadata_average_flip_rate")


# ------------------------------------------------------------
# Figure 2: Directional metadata flips
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.8))

plot_direction = plot_flip.copy()

x = np.arange(len(plot_direction))

ax.bar(
    x,
    plot_direction["Mean_To_Biased_Rate"],
    label="Not biased → Biased",
    color=Constants.COLORS["Not biased → Biased"],
    edgecolor="black",
    linewidth=0.7
)

ax.bar(
    x,
    plot_direction["Mean_To_Not_Biased_Rate"],
    bottom=plot_direction["Mean_To_Biased_Rate"],
    label="Biased → Not biased",
    color=Constants.COLORS["Biased → Not biased"],
    edgecolor="black",
    linewidth=0.7
)

for i, row in plot_direction.iterrows():
    total = row["Mean_To_Biased_Rate"] + row["Mean_To_Not_Biased_Rate"]
    ax.text(
        i,
        total + 0.01,
        f"{total * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12
    )

ax.set_xticks(x)
ax.set_xticklabels(plot_direction["Metadata_Label"], rotation=25, ha="right")
ax.set_ylabel("Directional flip rates")
ax.set_title("")
ax.legend(frameon=True)
ax.set_ylim(0, max(0.3, plot_direction["Mean_Flip_Rate"].max() + 0.08))
ax.yaxis.set_major_formatter(PercentFormatter(1.0))

plt.tight_layout()
utils.save_figure(fig, "q3_metadata_directional_flips")


# ------------------------------------------------------------
# Figure 3: Delta kappa by metadata condition and model
# ------------------------------------------------------------
delta_kappa_plot = (
    delta_df.pivot(
        index="Model_Display",
        columns="Metadata_Label",
        values="Delta_Kappa"
    )
)

ordered_labels = [METADATA_LABELS[c] for c in METADATA_ORDER]
delta_kappa_plot = delta_kappa_plot.reindex(columns=ordered_labels)

delta_kappa_plot = delta_kappa_plot.loc[
    delta_kappa_plot.mean(axis=1).sort_values(ascending=True).index
]

# Alignment McNemar stars
sig_lookup = (
    alignment_mcnemar_df[
        ["Model_Display", "Metadata_Label", "significance"]
    ]
    .set_index(["Model_Display", "Metadata_Label"])
    ["significance"]
    .to_dict()
)

fig, ax = plt.subplots(figsize=(7,8))

delta_kappa_plot.plot(
    kind="barh",
    ax=ax,
    color=[Constants.COLORS[c] for c in METADATA_ORDER],
    edgecolor="black",
    linewidth=0.9,
    width=0.9
)

ax.axvline(0, color="black", linewidth=1, linestyle="--")

ax.set_xlabel(r"$\Delta$ Cohen's kappa")
ax.set_ylabel("")
ax.set_title("")

ax.legend(
    title="",
    loc="lower right",
    fontsize=12
)

for container, metadata_label in zip(ax.containers, ordered_labels):
    for bar, model_name in zip(container, delta_kappa_plot.index):
        star = sig_lookup.get((model_name, metadata_label), "")

        if star in ["ns", "NA", ""]:
            continue

        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2 - 0.06
        offset = 0.001

        if width >= 0:
            x_pos = width + offset
            ha = "left"
        else:
            x_pos = width - offset
            ha = "right"

        ax.text(
            x_pos,
            y,
            star,
            va="center",
            ha=ha,
            fontsize=12,
            fontweight="bold"
        )

xmin = delta_kappa_plot.min().min()
xmax = delta_kappa_plot.max().max()

ax.set_xlim(xmin - 0.02, xmax + 0.02)

plt.tight_layout()
utils.save_figure(fig, "q3_metadata_delta_kappa_by_model")


# ------------------------------------------------------------
# Figure 4: Delta bias prediction rate by metadata condition
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.5))

plot_delta = overall_delta.set_index("Metadata_Condition").reindex(METADATA_ORDER).reset_index()

bars = ax.bar(
    plot_delta["Metadata_Label"],
    plot_delta["Mean_Delta_Bias_Rate"],
    color=[Constants.COLORS[c] for c in plot_delta["Metadata_Condition"]],
    edgecolor="black",
    linewidth=0.7,
    width=0.65
)

ax.axhline(0, color="black", linewidth=1, linestyle="--")

for bar, value in zip(bars, plot_delta["Mean_Delta_Bias_Rate"]):
    label_y = value + 0.01 if value >= 0 else value - 0.03
    va = "bottom" if value >= 0 else "top"

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        label_y,
        f"{value * 100:+.1f}%",
        ha="center",
        va=va,
        fontsize=12
    )

ax.set_ylabel(r"$\Delta$ bias prediction rate")
ax.set_title("Change in Bias Prediction Rate Across Metadata Conditions")
ax.set_xticklabels(plot_delta["Metadata_Label"], rotation=25, ha="right")

y_min = min(-0.15, plot_delta["Mean_Delta_Bias_Rate"].min() - 0.05)
y_max = max(0.15, plot_delta["Mean_Delta_Bias_Rate"].max() + 0.05)
ax.set_ylim(y_min, y_max)

plt.tight_layout()
utils.save_figure(fig, "q3_metadata_delta_bias_prediction_rate")


# ------------------------------------------------------------
# Figure 5: Heatmap model x metadata flip rate
# ------------------------------------------------------------
heatmap_flip = (
    flip_df.pivot(
        index="Model_Display",
        columns="Metadata_Label",
        values="Flip_Rate"
    )
    .reindex(columns=ordered_labels)
)

heatmap_flip = heatmap_flip.loc[
    heatmap_flip.mean(axis=1).sort_values(ascending=True).index
]

fig, ax = plt.subplots(figsize=(7.5, 5.6))

im = ax.imshow(
    heatmap_flip.values,
    aspect="auto",
    cmap="coolwarm",
    vmin=0,
    vmax=max(0.35, np.nanmax(heatmap_flip.values))
)

ax.set_xticks(np.arange(len(heatmap_flip.columns)))
ax.set_xticklabels(heatmap_flip.columns, rotation=30, ha="right")

ax.set_yticks(np.arange(len(heatmap_flip.index)))
ax.set_yticklabels(heatmap_flip.index)

ax.set_title("")

for i in range(heatmap_flip.shape[0]):
    for j in range(heatmap_flip.shape[1]):
        val = heatmap_flip.iloc[i, j]
        label = "NA" if pd.isna(val) else f"{val * 100:.0f}%"
        ax.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="white" if not pd.isna(val) and val > 0.25 else "black"
        )

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("")

for spine in ax.spines.values():
    spine.set_visible(False)

ax.tick_params(axis="both", length=0)

plt.tight_layout()
utils.save_figure(fig, "q3_metadata_model_by_condition_flip_heatmap")


# ============================================================
# 7. FINAL PRINT
# ============================================================
print("\nSaved Q3 metadata row-level outputs to:", utils.csv_dir.parent)

print("\nMain CSV files:")
print("- q3_metadata_compare_to_article_only_row_level.csv")
print("- q3_metadata_prediction_flip_summary_row_level.csv")
print("- q3_metadata_alignment_change_summary_row_level.csv")
print("- q3_metadata_mcnemar_prediction_change_row_level.csv")
print("- q3_metadata_mcnemar_alignment_change_row_level.csv")
print("- q3_metadata_delta_metrics_row_level.csv")
print("- q3_metadata_overall_flip_summary_row_level.csv")
print("- q3_metadata_overall_delta_summary_row_level.csv")

print("\nMain figures:")
print("- q3_metadata_average_flip_rate.png / .pdf")
print("- q3_metadata_directional_flips.png / .pdf")
print("- q3_metadata_delta_kappa_by_model.png / .pdf")
print("- q3_metadata_delta_bias_prediction_rate.png / .pdf")
print("- q3_metadata_model_by_condition_flip_heatmap.png / .pdf")