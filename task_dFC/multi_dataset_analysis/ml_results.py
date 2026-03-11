import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper_functions import (  # pyright: ignore[reportMissingImports]
    boldify_axes,
    build_experiment_display_info,
    draw_labeled_legend_panel,
    relabel_heatmap_rows,
    savefig_pub,
    setup_pub_style,
)

LEVEL = "group_lvl"
KEYS_NOT_TO_INCLUDE = [
    "Logistic regression permutation p_value",
    "Logistic regression permutation score mean",
    "Logistic regression permutation score std",
    "SVM permutation p_value",
    "SVM permutation score mean",
    "SVM permutation score std",
]
GROUP = "test"
TARGETS = [
    ("PCA", "Logistic regression balanced accuracy"),
    ("PLS", "Logistic regression balanced accuracy"),
    ("PCA", "SVM balanced accuracy"),
    ("PLS", "SVM balanced accuracy"),
    ("PCA", "SI"),
    ("PLS", "SI"),
]


def parse_args():
    helptext = """
    Script to make figures/tables from multi-dataset ML results.
    """
    parser = argparse.ArgumentParser(description=helptext)
    parser.add_argument(
        "--multi_dataset_info", type=str, help="path to multi-dataset info file"
    )
    parser.add_argument(
        "--simul_or_real", type=str, help="Specify 'simulated' or 'real' data"
    )
    return parser.parse_args()


def read_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def get_analysis_config(multi_dataset_info, simul_or_real):
    if simul_or_real == "real":
        return multi_dataset_info["real_data"]
    if simul_or_real == "simulated":
        return multi_dataset_info["simulated_data"]
    raise ValueError(f"Invalid simul_or_real: {simul_or_real}")


def get_classification_input_dir(ml_root, dataset_info):
    sessions = dataset_info.get("SESSIONS") or [None]
    session = sessions[0]
    if session is None:
        return f"{ml_root}/classification"
    return f"{ml_root}/classification/{session}"


def filter_ml_scores(ml_scores_new, tasks_to_include):
    filtered_scores = {
        key: [] for key in ml_scores_new[LEVEL].keys() if key not in KEYS_NOT_TO_INCLUDE
    }

    for index, task in enumerate(ml_scores_new[LEVEL]["task"]):
        if task not in tasks_to_include:
            continue
        for key in filtered_scores:
            filtered_scores[key].append(ml_scores_new[LEVEL][key][index])

    return filtered_scores


def merge_ml_scores(all_ml_scores, ml_scores_new_updated):
    if all_ml_scores is None:
        return ml_scores_new_updated

    for key, values in ml_scores_new_updated.items():
        if key in all_ml_scores:
            all_ml_scores[key].extend(values)
    return all_ml_scores


def collect_all_ml_scores(main_root, datasets, tasks_to_include):
    all_ml_scores = None

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        dataset_info_file = f"{main_root}/{dataset}/codes/dataset_info.json"
        ml_root = f"{main_root}/{dataset}/derivatives/ML"
        dataset_info = read_json(dataset_info_file)
        input_dir = get_classification_input_dir(ml_root, dataset_info)

        if not os.path.exists(input_dir):
            print(
                f"Input directory {input_dir} does not exist. Skipping dataset {dataset}."
            )
            continue

        all_ml_scores_files = [
            filename
            for filename in os.listdir(input_dir)
            if "ML_scores_classify_" in filename
        ]

        for filename in all_ml_scores_files:
            try:
                ml_scores_new = np.load(
                    f"{input_dir}/{filename}", allow_pickle=True
                ).item()
                filtered_scores = filter_ml_scores(ml_scores_new, tasks_to_include)
                all_ml_scores = merge_ml_scores(all_ml_scores, filtered_scores)
            except Exception as error:
                print(f"Error loading {filename}: {error}")
                continue

    return all_ml_scores


def validate_score_lengths(all_ml_scores):
    if all_ml_scores is None:
        return

    lengths = [len(values) for values in all_ml_scores.values()]
    if len(set(lengths)) != 1:
        print(
            "Warning: Not all keys have the same length in ALL_ML_SCORES. "
            f"key and length pairs: {dict(zip(all_ml_scores.keys(), lengths))}"
        )


def save_all_ml_scores(all_ml_scores, output_root, simul_or_real):
    os.makedirs(output_root, exist_ok=True)
    np.save(f"{output_root}/ALL_ML_SCORES_{simul_or_real}.npy", all_ml_scores)


def prepare_metric_dataframe(
    all_ml_scores, tasks_to_include, embedding, metric, simul_or_real
):
    df = pd.DataFrame.from_dict(all_ml_scores)
    df = df[df["task"].isin(tasks_to_include)]
    df = df[(df["embedding"] == embedding) & (df["group"] == GROUP)].copy()

    method_order = sorted(df["dFC method"].unique(), key=lambda method: method.lower())
    df["dFC method"] = pd.Categorical(
        df["dFC method"], categories=method_order, ordered=True
    )

    task_order, task_to_experiment, experiment_order, experiment_palette = (
        build_experiment_display_info(
            df["task"].unique(),
            task_reference_order=tasks_to_include,
            simul_or_real=simul_or_real,
        )
    )
    df["experiment"] = df["task"].map(task_to_experiment)

    return (
        df,
        method_order,
        task_order,
        task_to_experiment,
        experiment_order,
        experiment_palette,
    )


def build_best_and_multi_tables(df, metric):
    counts_task = df.groupby("task")["run"].nunique()
    multi_tasks = counts_task[counts_task > 1].index
    df_multi = df[df["task"].isin(multi_tasks)].copy()

    df_best = (
        df.sort_values(["task", "dFC method", metric], ascending=[True, True, False])
        .drop_duplicates(subset=["task", "dFC method"], keep="first")
        .rename(columns={metric: "score"})
    )

    return df_best, df_multi


def get_pointplot_limits(metric):
    if metric == "SI":
        return -1.0, 1.0
    return 0.5, 1.0


def get_heatmap_limits(metric):
    if metric == "SI":
        return None, 1.0, 0.0
    return 0.5 - 1e-6, 1.0, 0.5


def style_boxplot(ax, box_edge):
    for artist in ax.artists:
        artist.set_edgecolor(box_edge)
        facecolor = artist.get_facecolor()
        artist.set_facecolor((facecolor[0], facecolor[1], facecolor[2], 0.12))
    for line in ax.lines:
        line.set_color(box_edge)
        line.set_alpha(0.5)
        line.set_zorder(1)


def overlay_method_means(ax, df_best, lower, upper):
    means = df_best.groupby("dFC method", observed=True)["score"].mean()
    xticks = ax.get_xticks()
    xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
    x_positions = {label: xticks[index] for index, label in enumerate(xticklabels)}

    halfwidth = 0.1
    for method, mean_score in means.items():
        if method not in x_positions or pd.isna(mean_score):
            continue
        mean_score = min(upper, max(lower, mean_score))
        x_position = x_positions[method]
        ax.hlines(
            mean_score,
            x_position - halfwidth,
            x_position + halfwidth,
            colors="#050505",
            lw=2.4,
            zorder=3,
        )


def finalize_marker_edges(ax):
    for line in ax.lines:
        try:
            line.set_markeredgecolor("#222222")
            line.set_markeredgewidth(0.8)
        except Exception:
            pass


def plot_best_pointplot(
    df_best,
    method_order,
    experiment_order,
    experiment_palette,
    output_root,
    embedding,
    metric,
    simul_or_real,
):
    figure = plt.figure(figsize=(max(10, 0.6 * len(method_order)) + 5.0, 7.0))
    grid_spec = figure.add_gridspec(
        ncols=2, nrows=1, width_ratios=[1.0, 0.5], wspace=0.05
    )
    ax = figure.add_subplot(grid_spec[0, 0])
    ax_leg = figure.add_subplot(grid_spec[0, 1])

    box_face = to_rgba("#DE9995", 0.18)
    box_edge = "#730800"

    sns.boxplot(
        data=df_best,
        x="dFC method",
        y="score",
        order=method_order,
        whis=(5, 95),
        fliersize=0,
        linewidth=1.0,
        width=0.2,
        color=box_face,
        ax=ax,
        zorder=1,
    )
    style_boxplot(ax, box_edge)

    lower, upper = get_pointplot_limits(metric)
    overlay_method_means(ax, df_best, lower, upper)

    sns.pointplot(
        data=df_best,
        x="dFC method",
        y="score",
        hue="experiment",
        order=method_order,
        hue_order=experiment_order,
        dodge=0.4,
        errorbar=None,
        linestyles="",
        markers="o",
        palette=experiment_palette,
        ax=ax,
        zorder=6,
    )
    finalize_marker_edges(ax)

    ax.set_xlabel("dFC method")
    ax.set_ylabel(metric)
    if metric == "SI":
        ax.set_ylim(top=1.02)
    else:
        ax.set_ylim(0.48, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    sns.despine(ax=ax, top=True, right=True)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    if ax.legend_:
        ax.legend_.remove()

    draw_labeled_legend_panel(
        ax_leg,
        experiment_order,
        experiment_palette,
        ncols=2 if simul_or_real == "real" else 1,
        fontsize=8,
        markersize=5,
    )
    ax_leg.set_title("Experiment", fontsize=9, pad=4, fontweight="bold")
    box = ax_leg.get_position()
    ax_leg.set_position([box.x0, box.y0 - 0.03, box.width, box.height])

    boldify_axes(ax, xlabel="dFC method", ylabel=metric)
    figure.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])

    savefig_pub(
        f"{output_root}/ML_scores_{embedding}_{metric}_{LEVEL}_{simul_or_real}_best.png"
    )
    plt.close(figure)


def plot_best_heatmap(
    df_best,
    method_order,
    task_order,
    task_to_experiment,
    output_root,
    embedding,
    metric,
    simul_or_real,
):
    matrix_best = df_best.pivot(index="task", columns="dFC method", values="score")
    annot_best = df_best.assign(
        label=lambda df_plot: df_plot["score"].map(lambda value: f"{value:.2f}")
    ).pivot(index="task", columns="dFC method", values="label")

    matrix_best, annot_best, _ = relabel_heatmap_rows(
        matrix_best,
        annot_best,
        task_reference_order=task_order,
        task_to_experiment=task_to_experiment,
    )
    col_order = [method for method in method_order if method in matrix_best.columns]

    if simul_or_real == "real":
        width = max(10, 0.65 * len(col_order))
        height = max(6.0, 0.30 * len(matrix_best.index))
    else:
        width = max(11, 11 / 7 * len(col_order))
        height = max(7.0, 0.35 * len(matrix_best.index))

    figure, ax = plt.subplots(figsize=(width, height))
    vmin, vmax, center = get_heatmap_limits(metric)
    heatmap = sns.heatmap(
        matrix_best.loc[:, col_order],
        vmin=vmin,
        vmax=vmax,
        center=center,
        cmap="coolwarm",
        annot=annot_best.loc[:, col_order],
        fmt="",
        annot_kws={"fontsize": 9, "fontweight": "bold", "linespacing": 1.15},
        cbar_kws={"shrink": 0.7, "pad": 0.02},
        ax=ax,
    )
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label(metric, fontsize=10, fontweight="bold")
    colorbar.ax.tick_params(labelsize=9)

    boldify_axes(ax, xlabel="dFC method", ylabel="Experiment", rotate_xticks=35)
    ax.set_xlabel("dFC method")
    ax.set_ylabel("Experiment")
    plt.setp(ax.get_xticklabels(), fontweight="bold", rotation=35, ha="right")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    savefig_pub(
        f"{output_root}/ML_scores_heatmap_{embedding}_{metric}_{LEVEL}_{simul_or_real}_best.png"
    )
    plt.close(figure)


def build_across_heatmap_data(df_multi, metric, task_order, task_to_experiment):
    summary = (
        df_multi.groupby(["task", "dFC method"], observed=True)[metric]
        .agg(n="count", med="median", vmin="min", vmax="max")
        .reset_index()
    )

    matrix_across = summary.pivot(index="task", columns="dFC method", values="med")
    annot_across = summary.assign(
        label=lambda df_plot: df_plot["vmin"].map(lambda value: f"{value:.2f}")
        + "\u2013"
        + df_plot["vmax"].map(lambda value: f"{value:.2f}")
        + "\n"
        + df_plot["n"].map(lambda value: f"n={value}")
    ).pivot(index="task", columns="dFC method", values="label")

    return relabel_heatmap_rows(
        matrix_across,
        annot_across,
        task_reference_order=task_order,
        task_to_experiment=task_to_experiment,
    )


def plot_across_heatmap(
    df_multi,
    method_order,
    task_order,
    task_to_experiment,
    output_root,
    embedding,
    metric,
    simul_or_real,
):
    if df_multi.empty:
        print(
            f"[ACROSS-RUN] No tasks with ≥2 runs for {embedding} / {metric} — skipping across-run figures."
        )
        return

    matrix_across, annot_across, _ = build_across_heatmap_data(
        df_multi,
        metric,
        task_order,
        task_to_experiment,
    )
    col_order = [method for method in method_order if method in matrix_across.columns]
    width = max(9.0, 11 / 7 * len(col_order))
    height = max(7.0, 7 / 20 * len(matrix_across.index))

    figure, ax = plt.subplots(figsize=(width, height))
    vmin, vmax, center = get_heatmap_limits(metric)
    heatmap = sns.heatmap(
        matrix_across.loc[:, col_order],
        vmin=vmin,
        vmax=vmax,
        center=center,
        cmap="coolwarm",
        annot=annot_across.loc[:, col_order],
        fmt="",
        annot_kws={"fontsize": 9, "fontweight": "bold", "linespacing": 1.15},
        cbar_kws={"shrink": 0.7, "pad": 0.02},
        ax=ax,
    )

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label(metric, fontsize=10, fontweight="bold")
    boldify_axes(ax, xlabel="dFC method", ylabel="Experiment", rotate_xticks=35)
    ax.set_xlabel("dFC method")
    ax.set_ylabel("Experiment")
    plt.setp(ax.get_xticklabels(), fontweight="bold", rotation=35, ha="right")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    savefig_pub(
        f"{output_root}/ML_scores_heatmap_{embedding}_{metric}_{LEVEL}_{simul_or_real}_across.png"
    )
    plt.close(figure)


def generate_all_plots(all_ml_scores, tasks_to_include, output_root, simul_or_real):
    sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 1.2})
    sns.set_style("darkgrid")

    for embedding, metric in TARGETS:
        (
            df,
            method_order,
            task_order,
            task_to_experiment,
            experiment_order,
            experiment_palette,
        ) = prepare_metric_dataframe(
            all_ml_scores,
            tasks_to_include,
            embedding,
            metric,
            simul_or_real,
        )
        df_best, df_multi = build_best_and_multi_tables(df, metric)

        plot_best_pointplot(
            df_best,
            method_order,
            experiment_order,
            experiment_palette,
            output_root,
            embedding,
            metric,
            simul_or_real,
        )
        plot_best_heatmap(
            df_best,
            method_order,
            task_order,
            task_to_experiment,
            output_root,
            embedding,
            metric,
            simul_or_real,
        )
        plot_across_heatmap(
            df_multi,
            method_order,
            task_order,
            task_to_experiment,
            output_root,
            embedding,
            metric,
            simul_or_real,
        )


def main():
    args = parse_args()
    setup_pub_style()

    multi_dataset_info = read_json(args.multi_dataset_info)
    analysis_config = get_analysis_config(multi_dataset_info, args.simul_or_real)
    main_root = analysis_config["main_root"]
    datasets = analysis_config["DATASETS"]
    tasks_to_include = analysis_config["TASKS_to_include"]
    output_root = f"{multi_dataset_info['output_root']}/ML_results"

    all_ml_scores = collect_all_ml_scores(main_root, datasets, tasks_to_include)
    validate_score_lengths(all_ml_scores)
    save_all_ml_scores(all_ml_scores, output_root, args.simul_or_real)
    generate_all_plots(
        all_ml_scores,
        tasks_to_include,
        output_root,
        args.simul_or_real,
    )


if __name__ == "__main__":
    main()
