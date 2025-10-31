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
    add_domains_between_ylabel_and_ticks,
    boldify_axes,
    build_task_order_and_palette,
    domain_sorted_rows,
    draw_grouped_legend_panel,
    get_cog_domain_info,
    savefig_pub,
    setup_pub_style,
    task_domain_real,
    task_domain_simul,
)

level = "group_lvl"
keys_not_to_include = [
    "Logistic regression permutation p_value",
    "Logistic regression permutation score mean",
    "Logistic regression permutation score std",
    "SVM permutation p_value",
    "SVM permutation score mean",
    "SVM permutation score std",
]

#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to make figures/tables from multi-dataset ML results.
    """

    setup_pub_style()
    parser = argparse.ArgumentParser(description=HELPTEXT)

    parser.add_argument(
        "--multi_dataset_info", type=str, help="path to multi-dataset info file"
    )
    parser.add_argument(
        "--simul_or_real", type=str, help="Specify 'simulated' or 'real' data"
    )

    args = parser.parse_args()

    multi_dataset_info = args.multi_dataset_info
    simul_or_real = args.simul_or_real

    # Read dataset info
    with open(multi_dataset_info, "r") as f:
        multi_dataset_info = json.load(f)

    if simul_or_real == "real":
        main_root = multi_dataset_info["real_data"]["main_root"]
        DATASETS = multi_dataset_info["real_data"]["DATASETS"]
        TASKS_to_include = multi_dataset_info["real_data"]["TASKS_to_include"]
    elif simul_or_real == "simulated":
        main_root = multi_dataset_info["simulated_data"]["main_root"]
        DATASETS = multi_dataset_info["simulated_data"]["DATASETS"]
        TASKS_to_include = multi_dataset_info["simulated_data"]["TASKS_to_include"]

    output_root = f"{multi_dataset_info['output_root']}/ML_results"

    ALL_ML_SCORES = None
    for dataset in DATASETS:
        print(f"Processing dataset: {dataset}")
        dataset_info_file = f"{main_root}/{dataset}/codes/dataset_info.json"
        ML_root = f"{main_root}/{dataset}/derivatives/ML"

        # Read dataset info
        with open(dataset_info_file, "r") as f:
            dataset_info = json.load(f)

        if "SESSIONS" in dataset_info:
            SESSIONS = dataset_info["SESSIONS"]
        else:
            SESSIONS = None
        if SESSIONS is None:
            SESSIONS = [None]

        TASKS = dataset_info["TASKS"]

        if "RUNS" in dataset_info:
            RUNS = dataset_info["RUNS"]
        else:
            RUNS = None
        if RUNS is None:
            RUNS = {task: [None] for task in TASKS}

        # find all ML_scores_classify_dFC-id.npy in the ML_root/classfication/ folder
        # for now we will only use the first session
        session = SESSIONS[0]
        if session is None:
            input_dir = f"{ML_root}/classification"
        else:
            input_dir = f"{ML_root}/classification/{session}"
        if not os.path.exists(input_dir):
            print(
                f"Input directory {input_dir} does not exist. Skipping dataset {dataset}."
            )
            continue
        ALL_ML_SCORES_FILES = os.listdir(input_dir)
        ALL_ML_SCORES_FILES = [
            f for f in ALL_ML_SCORES_FILES if "ML_scores_classify_" in f
        ]
        for f in ALL_ML_SCORES_FILES:
            try:
                ML_scores_new = np.load(f"{input_dir}/{f}", allow_pickle=True).item()
                # ML_scores_new_updated is a new dictionary with same keys as ML_scores_new but empty lists
                ML_scores_new_updated = {
                    key: []
                    for key in ML_scores_new[level].keys()
                    if key not in keys_not_to_include
                }
                for task in TASKS:
                    if task not in TASKS_to_include:
                        continue
                    if task not in ML_scores_new[level]["task"]:
                        dFC_method = set(ML_scores_new[level]["dFC method"])
                        print(f"Task {task} not in ML_scores of {dFC_method}. Skipping.")
                        continue
                    for i in range(len(ML_scores_new[level]["task"])):
                        for key in ML_scores_new_updated.keys():
                            ML_scores_new_updated[key].append(
                                ML_scores_new[level][key][i]
                            )

                if ALL_ML_SCORES is None:
                    ALL_ML_SCORES = ML_scores_new_updated
                else:
                    for key in ML_scores_new_updated.keys():
                        if key in ALL_ML_SCORES:
                            ALL_ML_SCORES[key].extend(ML_scores_new_updated[key])
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue

    # check that the lists in all keys have the same length
    if ALL_ML_SCORES is not None:
        lengths = [len(v) for v in ALL_ML_SCORES.values()]
        if len(set(lengths)) != 1:
            print(
                f"Warning: Not all keys have the same length in ALL_ML_SCORES. key and length pairs: {dict(zip(ALL_ML_SCORES.keys(), lengths))}"
            )

    # save ALL_ML_SCORES
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    np.save(f"{output_root}/ALL_ML_SCORES_{simul_or_real}.npy", ALL_ML_SCORES)

    # ===== Plotting =====
    DOMAIN_ORDER, TASK2DOMAIN, DOMAIN_BASE = get_cog_domain_info(simul_or_real)
    # knobs
    GROUP = "test"
    TARGETS = [
        ("PCA", "Logistic regression balanced accuracy"),
        ("LE", "Logistic regression balanced accuracy"),
        ("PCA", "SVM balanced accuracy"),
        ("LE", "SVM balanced accuracy"),
        ("LE", "SI"),
        ("PCA", "SI"),
    ]
    # -------------------------------------------------------------------

    sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 1.2})
    sns.set_style("darkgrid")

    AGG_FUNC = np.median  # across-run aggregation

    for embedding, metric in TARGETS:
        # ---- slice ----
        df = pd.DataFrame.from_dict(ALL_ML_SCORES)
        df = df[df["task"].isin(TASKS_to_include)]
        df = df[(df["embedding"] == embedding) & (df["group"] == GROUP)]

        # methods in alphabetical order (your current rule)
        method_order = sorted(df["dFC method"].unique(), key=lambda s: s.lower())
        df["dFC method"] = pd.Categorical(
            df["dFC method"], categories=method_order, ordered=True
        )

        # --- domain tagging & task ordering/coloring (only for real data) ---
        if simul_or_real == "real":
            df["domain"] = df["task"].map(task_domain_real)
        elif simul_or_real == "simulated":
            df["domain"] = df["task"].map(task_domain_simul)
        # Use tasks present in THIS slice
        task_order, task_palette = build_task_order_and_palette(
            df["task"].unique(),
            simul_or_real=simul_or_real,
            similarity_L=0.05,
            similarity_S=0.04,
        )

        # ===== build BEST and ACROSS tables =====
        counts_task = df.groupby("task")["run"].nunique()
        multi_tasks = counts_task[counts_task > 1].index
        df_multi = df[
            df["task"].isin(multi_tasks)
        ]  # <- use this dataframe for ACROSS figures

        # ACROSS heatmap (aggregate then pivot):
        if not df_multi.empty:
            df_across = (
                df_multi.groupby(["task", "dFC method"], observed=True)[metric]
                .agg(score=AGG_FUNC)
                .reset_index()
            )

        # BEST: one row per (task, method) with the winning run kept
        df_best = (
            df.sort_values(["task", "dFC method", metric], ascending=[True, True, False])
            .drop_duplicates(subset=["task", "dFC method"], keep="first")
            .rename(columns={metric: "score"})
        )

        # ----------- POINTPLOT (BEST) -----------
        # 1) Make a 2-panel figure: left=plot, right=legend
        fig = plt.figure(figsize=(max(10, 0.6 * len(method_order)) + 5.0, 7.0))
        gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[1.0, 0.5], wspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[0, 1])  # empty panel for the legend

        # --- BACKGROUND: semi-transparent boxplot across tasks (per method) ---
        # one value per (task, method): use df_best['score']
        box_face = to_rgba("#DE9995", 0.18)  # neutral gray, ~18% opacity
        box_edge = "#730800"

        sns.boxplot(
            data=df_best,
            x="dFC method",
            y="score",
            order=method_order,
            whis=(
                5,
                95,
            ),  # <- 5th–95th percentile whiskers (change to "range", 1.5, etc. if you prefer)
            fliersize=0,  # hide outlier dots (keeps background clean)
            linewidth=1.0,
            width=0.2,  # narrower than default so points are visible
            color=box_face,  # face color (we’ll also set edge color below)
            ax=ax,
            zorder=1,
        )
        # ensure edges are visible but subtle; also enforce alpha on faces
        for artist in ax.artists:
            artist.set_edgecolor(box_edge)
            fc = artist.get_facecolor()
            artist.set_facecolor((fc[0], fc[1], fc[2], 0.12))  # set alpha explicitly
        for line in ax.lines:  # whiskers/medians/caps
            line.set_color(box_edge)
            line.set_alpha(0.5)
            line.set_zorder(1)

        # --- OVERLAY: method mean across tasks (black horizontal line) ---
        # (This is separate from the boxplot's median; gives an easy mean comparison)
        means = df_best.groupby("dFC method", observed=True)["score"].mean()
        xticks = ax.get_xticks()
        xlabs = [t.get_text() for t in ax.get_xticklabels()]
        xpos = {lab: xticks[i] for i, lab in enumerate(xlabs)}

        # bounds (SI vs BA)
        if metric == "SI":
            lower, upper = -1.0, 1.0
        else:
            lower, upper = 0.5, 1.0

        halfwidth = 0.1  # how wide the mean bar is around each tick
        for meth, m in means.items():
            if meth in xpos and pd.notna(m):
                m = min(upper, max(lower, m))  # clip to metric range
                x = xpos[meth]
                ax.hlines(
                    m, x - halfwidth, x + halfwidth, colors="#050505", lw=2.4, zorder=3
                )

        # --- FOREGROUND: your existing per-task pointplot (on top) ---
        sns.pointplot(
            data=df_best,
            x="dFC method",
            y="score",
            hue="task",
            order=method_order,
            hue_order=task_order,
            dodge=0.4,
            errorbar=None,
            linestyles="",
            markers="o",
            palette=task_palette,
            ax=ax,
            zorder=6,
        )

        # optional: crisp marker edges
        for line in ax.lines:
            try:
                line.set_markeredgecolor("#222222")
                line.set_markeredgewidth(0.8)
            except Exception:
                pass

        ax.set_xlabel("dFC method")
        ax.set_ylabel(metric)
        if metric == "SI":
            ax.set_ylim(top=1.02)
        else:
            ax.set_ylim(0.48, 1.02)
        ax.grid(True, axis="y", alpha=0.25)
        sns.despine(ax=ax, top=True, right=True)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

        # kill any in-axes legend and draw grouped legend in the right panel
        if ax.legend_:
            ax.legend_.remove()
        if simul_or_real == "real":
            domain_of = {t: task_domain_real(t) for t in task_order}
            draw_grouped_legend_panel(
                ax_leg,
                task_order,
                domain_of,
                task_palette,
                DOMAIN_ORDER,
                ncols=2,
                fontsize=8,
                markersize=5,
            )
            ax_leg.set_title("Task Paradigm", fontsize=9, pad=4, fontweight="bold")
        elif simul_or_real == "simulated":
            domain_of = {t: task_domain_simul(t) for t in task_order}
            draw_grouped_legend_panel(
                ax_leg,
                task_order,
                domain_of,
                task_palette,
                DOMAIN_ORDER,
                ncols=1,
                fontsize=8,
                markersize=5,
            )
            ax_leg.set_title("Task Paradigm", fontsize=9, pad=4, fontweight="bold")

        box = ax_leg.get_position()
        ax_leg.set_position(
            [box.x0, box.y0 - 0.03, box.width, box.height]
        )  # move down by ~3% fig height

        boldify_axes(ax, xlabel="dFC method", ylabel=metric)

        # IMPORTANT: don't call a plain tight_layout() now; the GridSpec already allocates space.
        # If you must, keep a small margin:
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])

        savefig_pub(
            f"{output_root}/ML_scores_{embedding}_{metric}_{level}_{simul_or_real}_best.png"
        )
        plt.close(fig)

        # ----------- HEATMAPS -----------
        # BEST heatmap: values from df_best
        mat_best = df_best.pivot(index="task", columns="dFC method", values="score")
        row_order = domain_sorted_rows(mat_best.index, TASKS_to_include, simul_or_real)
        col_order = [m for m in method_order if m in mat_best.columns]

        annot_best = df_best.assign(
            label=lambda x: x["score"].map(lambda v: f"{v:.2f}")
        ).pivot(index="task", columns="dFC method", values="label")

        if simul_or_real == "real":
            w = max(10, 0.65 * len(col_order))
            h = max(6.0, 0.30 * len(row_order))
        else:
            w = max(11, 11 / 7 * len(col_order))
            h = max(7.0, 0.35 * len(row_order))
        fig, ax = plt.subplots(figsize=(w, h))
        vmin, vmax, center = (
            (None, 1.0, 0.0) if metric == "SI" else (0.5 - 1e-6, 1.0, 0.5)
        )
        hm = sns.heatmap(
            mat_best.loc[row_order, col_order],
            vmin=vmin,
            vmax=vmax,
            center=center,
            cmap="coolwarm",
            annot=annot_best.loc[row_order, col_order],
            fmt="",
            annot_kws={"fontsize": 9, "fontweight": "bold", "linespacing": 1.15},
            cbar_kws={"shrink": 0.7, "pad": 0.02},
            ax=ax,
        )
        cbar = hm.collections[0].colorbar
        cbar.set_label(metric, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        boldify_axes(ax, xlabel="dFC method", ylabel="Task Paradigm", rotate_xticks=35)

        if simul_or_real == "real":
            task_to_domain = {
                t: task_domain_real(t) for t in row_order
            }  # your task_domain helper
            domain_x_frac = -0.8
            ylabel_pad_pts = 130
        elif simul_or_real == "simulated":
            task_to_domain = {
                t: task_domain_simul(t) for t in row_order
            }  # your task_domain helper
            domain_x_frac = -1.0
            ylabel_pad_pts = 110
        add_domains_between_ylabel_and_ticks(
            ax,
            row_order=row_order,
            task_to_domain=task_to_domain,
            label_rotation=0,  # try 0, 20, 30, or 45
            tick_pad_pts=0,  # pushes tick labels to the right
            ylabel_pad_pts=ylabel_pad_pts,  # moves y-axis label left
            domain_x_frac=domain_x_frac,  # where domain column sits (more negative = further left)
            left_extend_frac=0.01,  # extend the line a bit further left than the text
            label_x_offset_frac=0.010,  # nudge text right from the anchor
            label_align="left",  # <<< left-align labels
            label_kw=dict(
                fontsize=9, fontweight="bold", color="#222", ha="center", va="center"
            ),
            sep_kw=dict(color="#777", lw=1.0, alpha=0.9),
        )

        # Bold colorbar label (metric name) too:
        cbar.set_label(metric, fontsize=10, fontweight="bold")

        ax.set_xlabel("dFC method")
        ax.set_ylabel("Task Paradigm")
        # ax.set_title(f"Best across runs • {embedding} • {metric}", pad=8)
        ax.tick_params(axis="x", labelrotation=35, labelsize=9)
        plt.setp(ax.get_xticklabels(), fontweight="bold", rotation=35, ha="right")
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        sns.despine(ax=ax, top=True, right=True)
        plt.tight_layout()
        savefig_pub(
            f"{output_root}/ML_scores_heatmap_{embedding}_{metric}_{level}_{simul_or_real}_best.png"
        )
        plt.close(fig)

        # ACROSS heatmap: color = median; annotation = min–max & n (across runs)
        if df_multi.empty:
            print(
                f"[ACROSS-RUN] No tasks with ≥2 runs for {embedding} / {metric} — skipping across-run figures."
            )
        else:
            # aggregate across runs
            s = (
                df_multi.groupby(["task", "dFC method"], observed=True)[metric]
                .agg(n="count", med="median", vmin="min", vmax="max")
                .reset_index()
            )

            # heatmap scaling (avoid name clash with s['vmin'] / s['vmax'])
            if metric == "SI":
                cmin, cmax, ccenter = None, 1.0, 0.0  # SI in [-1,1], center at 0
            else:
                cmin, cmax, ccenter = (
                    0.5 - 1e-6,
                    1.0,
                    0.5,
                )  # accuracy in [0.5,1], center at chance

            # pivots
            mat_across = s.pivot(index="task", columns="dFC method", values="med")
            ann_text = s.assign(
                label=lambda d: d["vmin"].map(lambda v: f"{v:.2f}")
                + "\u2013"
                + d["vmax"].map(lambda v: f"{v:.2f}")
                + "\n"
                + d["n"].map(lambda n: f"n={n}")
            ).pivot(index="task", columns="dFC method", values="label")

            # order
            row_order = domain_sorted_rows(
                mat_across.index, TASKS_to_include, simul_or_real
            )
            col_order = [m for m in method_order if m in mat_across.columns]

            # plot
            w = max(9.0, 9 / 7 * len(col_order))
            h = max(7.0, 7 / 20 * len(row_order))
            fig, ax = plt.subplots(figsize=(w, h))
            hm = sns.heatmap(
                mat_across.loc[row_order, col_order],
                vmin=cmin,
                vmax=cmax,
                center=ccenter,
                cmap="coolwarm",
                annot=ann_text.loc[row_order, col_order],
                fmt="",
                annot_kws={"fontsize": 9, "fontweight": "bold", "linespacing": 1.15},
                cbar_kws={"shrink": 0.7, "pad": 0.02},
                ax=ax,
            )

            # domain sidebar & separators (your helper)
            if simul_or_real == "real":
                task_to_domain = {t: task_domain_real(t) for t in row_order}
                domain_x_frac = -0.8
                ylabel_pad_pts = 130
            else:  # "simulated"
                task_to_domain = {t: task_domain_simul(t) for t in row_order}
                domain_x_frac = -1.0
                ylabel_pad_pts = 110

            add_domains_between_ylabel_and_ticks(
                ax,
                row_order=row_order,
                task_to_domain=task_to_domain,
                label_rotation=0,
                tick_pad_pts=0,
                ylabel_pad_pts=ylabel_pad_pts,
                domain_x_frac=domain_x_frac,
                left_extend_frac=0.01,
                label_x_offset_frac=0.010,
                label_align="left",
                label_kw=dict(
                    fontsize=9, fontweight="bold", color="#222", ha="center", va="center"
                ),
                sep_kw=dict(color="#777", lw=1.0, alpha=0.9),
            )

            # cosmetics
            cbar = hm.collections[0].colorbar
            cbar.set_label(metric, fontsize=10, fontweight="bold")
            boldify_axes(
                ax, xlabel="dFC method", ylabel="Task Paradigm", rotate_xticks=35
            )
            plt.setp(ax.get_xticklabels(), fontweight="bold", rotation=35, ha="right")
            plt.setp(ax.get_yticklabels(), fontweight="bold")
            sns.despine(ax=ax, top=True, right=True)
            plt.tight_layout()
            savefig_pub(
                f"{output_root}/ML_scores_heatmap_{embedding}_{metric}_{level}_{simul_or_real}_across.png"
            )
            plt.close(fig)
