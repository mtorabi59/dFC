import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pydfc.data_loader import find_subj_list
from pydfc.ml_utils import load_task_data
from pydfc.task_utils import (
    calc_relative_task_on,
    calc_rest_duration,
    calc_task_duration,
    calc_transition_freq,
    extract_task_presence,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper_functions import (  # pyright: ignore[reportMissingImports]
    annotate_medians_by_geometry,
    annotate_medians_single_boxplot,
    as_long_df,
    order_by_median_dict,
    setup_pub_style,
)

fig_bbox_inches = "tight"
fig_pad = 0.1
show_title = False
save_fig_format = "png"  # pdf, png,

#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to analyze and visualize task timing statistics across multiple datasets.
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
    output_root = f"{multi_dataset_info['output_root']}/task_data_stats/{simul_or_real}"

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    task_ratio_all = {}
    transition_freq_all = {}
    rest_durations_all = {}
    task_durations_all = {}
    for dataset in DATASETS:

        print(f"Processing dataset: {dataset}")
        dataset_info_file = f"{main_root}/{dataset}/codes/dataset_info.json"
        roi_root = f"{main_root}/{dataset}/derivatives/ROI_timeseries"
        dFC_root = f"{main_root}/{dataset}/derivatives/dFC_assessed"

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

        for session in SESSIONS:
            for task_id, task in enumerate(TASKS):
                if not task in TASKS_to_include:
                    continue
                for run in RUNS[task]:
                    SUBJECTS = find_subj_list(roi_root)
                    # print(f"Number of subjects: {len(SUBJECTS)}")

                    for subj in SUBJECTS:

                        try:
                            task_data = load_task_data(
                                roi_root=roi_root,
                                subj=subj,
                                task=task,
                                run=run,
                                session=session,
                            )
                        except FileNotFoundError:
                            continue

                        task_presence, indices = extract_task_presence(
                            event_labels=task_data["event_labels"],
                            TR_task=1 / task_data["Fs_task"],
                            TR_mri=task_data["TR_mri"],
                            binary=True,
                            binarizing_method="GMM",
                            no_hrf=False,
                        )

                        relative_task_on = calc_relative_task_on(task_presence[indices])
                        num_of_transitions, relative_transition_freq = (
                            calc_transition_freq(task_presence[indices])
                        )
                        # calculate rest and task durations based original event labels
                        event_labels = np.multiply(task_data["event_labels"] != 0, 1)
                        rest_durations = calc_rest_duration(
                            event_labels, TR_mri=1 / task_data["Fs_task"]
                        )
                        task_durations = calc_task_duration(
                            event_labels, TR_mri=1 / task_data["Fs_task"]
                        )

                        if not task in task_ratio_all:
                            task_ratio_all[task] = []
                        if not task in transition_freq_all:
                            transition_freq_all[task] = []
                        if not task in rest_durations_all:
                            rest_durations_all[task] = []
                        if not task in task_durations_all:
                            task_durations_all[task] = []
                        task_ratio_all[task].append(relative_task_on)
                        transition_freq_all[task].append(relative_transition_freq)
                        # rest_durations and task_durations are lists
                        rest_durations_all[task].extend(rest_durations)
                        task_durations_all[task].extend(task_durations)

    DATA = {
        "task_ratio_all": task_ratio_all,
        "transition_freq_all": transition_freq_all,
        "rest_durations_all": rest_durations_all,
        "task_durations_all": task_durations_all,
    }
    # np.save(f"task_data_stats_{simul_or_real}.npy", DATA)

    # =========================
    # Paper-quality seaborn plots (patched)
    # =========================

    sns.set_theme(context="paper", style="darkgrid")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 500,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
        }
    )

    # ==============================
    # 1) Task ratio (sorted by median) — BOX PLOT + median labels
    # ==============================
    order_ratio, stats_ratio = order_by_median_dict(task_ratio_all, reverse=True)
    df_ratio = as_long_df(task_ratio_all, "task_ratio")
    df_ratio = df_ratio[df_ratio["task"].isin(order_ratio)]
    df_ratio["task"] = pd.Categorical(
        df_ratio["task"], categories=order_ratio, ordered=True
    )

    fig_w = max(15, 15 / 30 * len(order_ratio))
    plt.figure(figsize=(fig_w, 6))

    ax = sns.boxplot(
        data=df_ratio,
        x="task",
        y="task_ratio",
        order=order_ratio,
        width=0.6,
        linewidth=1,
        showfliers=False,
    )

    ax.set_xlabel("Task paradigm")
    ax.set_ylabel("Task ratio")
    ax.set_ylim(0, 1)  # keep ratios bounded

    # annotate medians (use integers if you prefer: fmt="{:.0f}")
    annotate_medians_single_boxplot(
        ax,
        df_ratio,
        x_col="task",
        y_col="task_ratio",
        order=order_ratio,
        fmt="{:.2f}",
        box_alpha=0.6,
    )

    for label in ax.get_xticklabels():
        label.set_rotation(65)
        label.set_horizontalalignment("right")
        label.set_fontweight("bold")
    if show_title:
        ax.set_title("Task ratio per task (box + samples, ordered by median)", pad=12)

    plt.tight_layout()
    plt.savefig(
        f"{output_root}/task_ratio_{simul_or_real}.{save_fig_format}",
        bbox_inches=fig_bbox_inches,
        pad_inches=fig_pad,
    )
    plt.close()

    # ======================================
    # 2) Transition frequency (sorted by median) — BOX PLOT + median labels
    # ======================================
    order_tf, stats_tf = order_by_median_dict(transition_freq_all, reverse=True)
    df_tf = as_long_df(transition_freq_all, "transition_freq")
    df_tf = df_tf[df_tf["task"].isin(order_tf)]
    df_tf["task"] = pd.Categorical(df_tf["task"], categories=order_tf, ordered=True)

    fig_w = max(15, 15 / 30 * len(order_tf))
    plt.figure(figsize=(fig_w, 6))

    ax = sns.boxplot(
        data=df_tf,
        x="task",
        y="transition_freq",
        order=order_tf,
        width=0.6,
        linewidth=1,
        showfliers=False,
    )

    ax.set_xlabel("Task paradigm")
    ax.set_ylabel("Relative transition frequency")

    # annotate medians
    annotate_medians_single_boxplot(
        ax,
        df_tf,
        x_col="task",
        y_col="transition_freq",
        order=order_tf,
        fmt="{:.2f}",
        box_alpha=0.6,
    )

    for label in ax.get_xticklabels():
        label.set_rotation(65)
        label.set_horizontalalignment("right")
        label.set_fontweight("bold")
    if show_title:
        ax.set_title(
            "Transition frequency per task (box + samples, ordered by median)", pad=12
        )

    plt.tight_layout()
    plt.savefig(
        f"{output_root}/transition_freq_{simul_or_real}.{save_fig_format}",
        bbox_inches=fig_bbox_inches,
        pad_inches=fig_pad,
    )
    plt.close()

    # =========================================================
    # 3) Rest vs Task durations: side-by-side per task paradigm (LOG SCALE)
    # =========================================================
    df_rest = as_long_df(rest_durations_all, "duration")
    df_rest["state"] = "Rest"
    df_task = as_long_df(task_durations_all, "duration")
    df_task["state"] = "Task"
    df_dur = pd.concat([df_rest, df_task], ignore_index=True)

    # Order tasks by mean Task duration (change to Rest if you prefer)
    order_dur, _ = order_by_median_dict(task_durations_all, reverse=True)
    df_dur = df_dur[df_dur["task"].isin(order_dur)]
    df_dur["task"] = pd.Categorical(df_dur["task"], categories=order_dur, ordered=True)

    # ---- LOG display handling (avoid -inf for zeros) ----
    # pick an adaptive epsilon based on the smallest positive value
    pos = df_dur.loc[df_dur["duration"] > 0, "duration"]
    if len(pos) == 0:
        EPS = 1e-3
    else:
        EPS = max(min(pos) / 10.0, 1e-3)  # small but data-driven
    df_dur["duration_plot"] = df_dur["duration"].clip(lower=EPS)

    fig_w = max(17, 17 / 30 * len(order_dur))
    plt.figure(figsize=(fig_w, 7))

    # Boxplot on log scale (no fliers; jitters will show samples, incl. singletons)
    ax = sns.boxplot(
        data=df_dur,
        x="task",
        y="duration_plot",
        hue="state",
        order=order_dur,
        hue_order=["Rest", "Task"],
        linewidth=1,
        dodge=True,
        showfliers=False,
        width=0.6,
    )

    # Put y-axis on log scale (preserves wide dynamic range)
    ax.set_yscale("log")

    # annotate medians on the median line (log-scale safe)
    annotate_medians_by_geometry(
        ax=ax,
        df_long=df_dur,  # the DF you plotted
        x_col="task",
        hue_col="state",
        y_col="duration_plot",  # the epsilon-clipped column you used for plotting
        x_order=order_dur,
        hue_order=["Rest", "Task"],
        fmt="{:.0f}",
        y_nudge_factor=1.08,  # bump if labels sit on the line in log-space
        bin_halfwidth=0.6,  # widen if categories are very tightly packed
        bbox_alpha=0.6,  # make label bg more opaque for legibility
    )

    # Clean up duplicated legends (boxplot + stripplot both add entries)
    handles, labels = ax.get_legend_handles_labels()
    # the first two unique handles correspond to Rest/Task once; keep those
    unique = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen:
            unique.append((h, l))
            seen.add(l)
    # Keep only Rest/Task (first two)
    handles_clean, labels_clean = (
        zip(*unique[:2]) if len(unique) >= 2 else (handles[:2], labels[:2])
    )
    ax.legend(handles_clean, labels_clean, title="", frameon=True, loc="upper right")

    ax.set_xlabel("Task paradigm")
    ax.set_ylabel("Duration (sec, log scale)")

    for label in ax.get_xticklabels():
        label.set_rotation(65)
        label.set_horizontalalignment("right")
        label.set_fontweight("bold")

    if show_title:
        ax.set_title("Rest vs Task durations per task (log scale; box + points)", pad=12)

    plt.tight_layout()
    plt.savefig(
        f"{output_root}/durations_rest_vs_task_{simul_or_real}.{save_fig_format}",
        bbox_inches=fig_bbox_inches,
        pad_inches=fig_pad,
    )
    plt.close()
    # =========================================================
