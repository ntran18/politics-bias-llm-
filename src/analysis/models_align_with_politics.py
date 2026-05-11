from pathlib import Path
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

from matplotlib.ticker import PercentFormatter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.config import Constants
from core.utils import Utils

# =========================================================
# Configuration
# =========================================================
BASE_PATH = Path("../../results/v8/v8.1/")
PROMPT_FILE = "prompt_article_info.csv"

BAR_WIDTH = 0.65
HIST_COLOR = "0.65"

utils = Utils("question2")

# ============================================================
# 1. Load human data
# ============================================================
human_df = utils.load_human_data()
human_df = human_df[human_df["politics"].isin(Constants.POLITICS_GROUPS)].copy()

print("\nHuman data:")
print(human_df[["index", "article_id", "politics", "bias-question", "human_label"]].head())


# ============================================================
# 2. Load LLM outputs
# ============================================================
llm_df = utils.load_model_outputs(BASE_PATH, PROMPT_FILE)

print("\nLLM data:")
print(llm_df.head())


# ============================================================
# 3. Merge human and LLM rows
# ============================================================
merged_df = human_df.merge(
    llm_df,
    on="index",
    how="inner",
    suffixes=("_human", "_llm")
)

mismatch_count = (
    merged_df["article_id_human"] != merged_df["article_id_llm"]
).sum()

print("\nArticle ID mismatches:", mismatch_count)

if mismatch_count > 0:
    print(
        merged_df.loc[
            merged_df["article_id_human"] != merged_df["article_id_llm"],
            ["Model", "index", "article_id_human", "article_id_llm"]
        ].head()
    )

merged_df["article_id"] = merged_df["article_id_human"]

print("\nMissing labels before cleaning:")
print(merged_df[["human_label", "llm_label"]].isna().sum())

merged_df = merged_df.dropna(subset=["human_label", "llm_label"]).copy()

merged_df["human_label"] = merged_df["human_label"].astype(int)
merged_df["llm_label"] = merged_df["llm_label"].astype(int)

merged_df["Aligned"] = (
    merged_df["human_label"] == merged_df["llm_label"]
).astype(int)

merged_df["politics"] = pd.Categorical(
    merged_df["politics"],
    categories=Constants.POLITICS_GROUPS,
    ordered=True
)

merged_df["Model_Display"] = merged_df["Model"].map(utils.pretty_model_name)


# ============================================================
# 4. Summary metrics by model and politics
# ============================================================
summary = (
    merged_df
    .groupby(["Model", "Model_Display", "politics"], observed=True)
    .agg(
        N=("Aligned", "count"),
        Accuracy=("Aligned", "mean"),
        Bias_Prediction_Rate=("llm_label", "mean"),
        Human_Bias_Rate=("human_label", "mean"),
    )
    .reset_index()
)

kappa_df = (
    merged_df
    .groupby(["Model", "politics"], observed=True)
    .apply(utils.safe_kappa)
    .reset_index(name="Kappa")
)

summary = summary.merge(kappa_df, on=["Model", "politics"], how="left")

print("\n=== Summary by model and politics ===")
print(summary)


# ============================================================
# 5. Logistic regression per model
# Conservative is baseline
# ============================================================
coef_rows = []
overall_rows = []

for model_name in sorted(merged_df["Model"].unique()):
    df_model = merged_df[merged_df["Model"] == model_name].copy()

    df_model["politics"] = pd.Categorical(
        df_model["politics"],
        categories=Constants.POLITICS_GROUPS,
        ordered=True
    )

    # Full model: accuracy depends on politics
    full_model = smf.logit(
        "Aligned ~ C(politics, Treatment(reference='Conservative'))",
        data=df_model
    ).fit(disp=0)

    # Null model: no politics effect
    null_model = smf.logit(
        "Aligned ~ 1",
        data=df_model
    ).fit(disp=0)

    lr_stat = 2 * (full_model.llf - null_model.llf)
    df_diff = full_model.df_model - null_model.df_model
    overall_p = chi2.sf(lr_stat, df_diff)

    overall_rows.append({
        "Model": model_name,
        "Model_Display": utils.pretty_model_name(model_name),
        "LR_statistic": lr_stat,
        "df": df_diff,
        "overall_p_value": overall_p,
        "overall_significance": utils.significance_star(overall_p),
        "interpretation": (
            "Accuracy differs by political group"
            if overall_p < 0.05
            else "No significant overall political-group difference"
        )
                })

    coef_table = full_model.summary2().tables[1].reset_index()
    coef_table = coef_table.rename(columns={
        "index": "Term",
        "Coef.": "Coefficient",
        "Std.Err.": "Std_Error",
        "P>|z|": "p_value",
        "[0.025": "CI_2.5",
        "0.975]": "CI_97.5"
    })

    for _, row in coef_table.iterrows():
        term = row["Term"]

        if term == "Intercept":
            comparison = "Conservative baseline"
            group = "Conservative"
        elif "Liberal" in term:
            comparison = "Liberal vs Conservative"
            group = "Liberal"
        elif "Independent" in term:
            comparison = "Independent vs Conservative"
            group = "Independent"
        else:
            comparison = term
            group = term

        odds_ratio = np.exp(row["Coefficient"])
        ci_low = np.exp(row["CI_2.5"])
        ci_high = np.exp(row["CI_97.5"])

        coef_rows.append({
            "Model": model_name,
            "Model_Display": utils.pretty_model_name(model_name),
            "Group": group,
            "Comparison": comparison,
            "Coefficient_log_odds": row["Coefficient"],
            "Odds_Ratio": odds_ratio,
            "OR_CI_2.5": ci_low,
            "OR_CI_97.5": ci_high,
            "p_value": row["p_value"],
            "significance": utils.significance_star(row["p_value"]),
            "direction": (
                "higher accuracy than Conservative"
                if row["Coefficient"] > 0
                else "lower accuracy than Conservative"
            )
        })

    with open(
        os.path.join(utils.csv_dir, f"logit_summary_{model_name}.txt"),
        "w"
    ) as f:
        f.write(str(full_model.summary()))


logit_coef_df = pd.DataFrame(coef_rows)
logit_overall_df = pd.DataFrame(overall_rows)

print("\n=== Logistic regression overall tests ===")
print(logit_overall_df)

print("\n=== Logistic regression coefficients ===")
print(logit_coef_df)


# ============================================================
# 6. Prepare plot labels
# ============================================================
plot_df = summary.copy()

# Significance labels for Liberal and Independent vs Conservative
sig_lookup = {}

for _, row in logit_coef_df.iterrows():
    if row["Group"] in ["Liberal", "Independent"]:
        sig_lookup[(row["Model"], row["Group"])] = row["significance"]

plot_df["Sig_vs_Conservative"] = plot_df.apply(
    lambda r: "" if r["politics"] == "Conservative"
    else sig_lookup.get((r["Model"], r["politics"]), ""),
    axis=1
)

overall_lookup = dict(
    zip(logit_overall_df["Model"], logit_overall_df["overall_significance"])
)

plot_df["Overall_Sig"] = plot_df["Model"].map(overall_lookup)


# ============================================================
# 7. Clean grouped bar chart: Accuracy by politics
# ============================================================
model_order = (
    summary.groupby("Model")["Accuracy"]
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)

model_display_order = [utils.pretty_model_name(m) for m in model_order]

x = np.arange(len(model_order))
bar_width = 0.30

fig, ax = plt.subplots(figsize=(10, 4.8))

for i, group in enumerate(Constants.POLITICS_GROUPS):
    group_data = (
        plot_df[plot_df["politics"] == group]
        .set_index("Model")
        .reindex(model_order)
        .reset_index()
    )

    offset = (i - 1) * bar_width

    bars = ax.bar(
        x + offset,
        group_data["Accuracy"],
        width=bar_width,
        label=group,
        color=Constants.COLORS[group],
        edgecolor="black",
        linewidth=0.4
    )

    for bar, (_, row) in zip(bars, group_data.iterrows()):
        value = row["Accuracy"]

        # Accuracy label
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value*100:.1f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

        # Significance stars only for Liberal/Independent vs Conservative
        if group != "Conservative":
            sig = row["Sig_vs_Conservative"]

            if sig in ["*", "**", "***"]:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.05,
                    sig,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold"
                )

ax.yaxis.set_major_formatter(
    PercentFormatter(1.0, decimals=0)
)
ax.legend(
    title="Political group",
    ncol=1,
    loc="upper right",
)

ax.text(
    0.5,
    1,
    "* p < .05, ** p < .01, *** p < .001\n"
    "Stars compare each group with Conservative baseline.",
    transform=ax.transAxes,
    fontsize=12,
    va="top",
    ha="center"
)

ax.set_xticks(x)
ax.set_xticklabels(model_display_order, rotation=35, ha="right")
utils.finalize_plot(ax, ylabel="Accuracy", rotate_x=35, ylim=(0, 0.75))

utils.save_figure(fig, "q2_accuracy_by_politics")




# ============================================================
# 8. Heatmaps: Kappa and bias prediction rate by politics
# ============================================================
# Cohen's kappa heatmap: model x political group
kappa_by_group = (
    summary
    .pivot(index="Model", columns="politics", values="Kappa")
    .reindex(columns=Constants.POLITICS_GROUPS)
)

utils.plot_heatmap(
    fig_size=(4, 4.8),
    df=kappa_by_group,
    model_order=model_order,
    cbar_label="",
    filename="q2_kappa_heatmap_by_politics",
    vmin=-0.10,
    vmax=0.20,
    cmap="coolwarm_r",
    column_order=Constants.POLITICS_GROUPS
)

# Bias prediction rate heatmap: model x political group
bias_rate_by_group = (
    summary
    .pivot(index="Model", columns="politics", values="Bias_Prediction_Rate")
    .reindex(columns=Constants.POLITICS_GROUPS)
)

utils.plot_heatmap(
    fig_size=(4, 4.8),
    df=bias_rate_by_group,
    model_order=model_order,
    cbar_label="",
    filename="q2_bias_prediction_rate_heatmap_by_politics",
    vmin=0,
    vmax=1,
    cmap="coolwarm_r",
    fmt="percent_no_symbol",
    column_order=Constants.POLITICS_GROUPS
)

# ============================================================
# 9. Save CSV files for all summary tables
# ============================================================
utils.save_csv(
    merged_df,
    "q2_logistic_long_data.csv",
    index=False
)

utils.save_csv(
    summary,
    "q2_summary_accuracy_kappa_biasrate.csv",
    index=False
)

utils.save_csv(
    logit_coef_df,
    "q2_logistic_coefficients_vs_conservative.csv",
    index=False
)

utils.save_csv(
    logit_overall_df,
    "q2_logistic_overall_politics_effect.csv",
    index=False
)


utils.save_csv(
    kappa_by_group,
    "q2_kappa_by_model_and_politics.csv"
)

utils.save_csv(
    bias_rate_by_group,
    "q2_bias_prediction_rate_by_model_and_politics.csv"
)

# ============================================================
# 10. Print final notes
# ============================================================
print("\nSaved figures to:")
print(utils.figure_dir)

print("\nSaved CSV files to:")
print(utils.csv_dir)