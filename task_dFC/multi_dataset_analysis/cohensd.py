import argparse
import json

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import datasets, plotting

from pydfc import data_loader
from pydfc.ml_utils import find_available_subjects, load_task_data
from pydfc.task_utils import cohen_d_bold, extract_task_presence

#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to compute and visualize Cohen's d effect sizes for task vs. rest BOLD signals across multiple datasets.
    """

    parser = argparse.ArgumentParser(description=HELPTEXT)

    parser.add_argument(
        "--multi_dataset_info", type=str, help="path to multi-dataset info file"
    )

    args = parser.parse_args()

    multi_dataset_info = args.multi_dataset_info

    # Read dataset info
    with open(multi_dataset_info, "r") as f:
        multi_dataset_info = json.load(f)

    main_root = multi_dataset_info["real_data"]["main_root"]
    DATASETS = multi_dataset_info["real_data"]["DATASETS"]
    TASKS_to_include = multi_dataset_info["real_data"]["TASKS_to_include"]
    output_root = f"{multi_dataset_info['output_root']}/CohensD"

    CohensD_across_task = {
        "task": [],
        "d_values": [],
        "dataset": [],
        "ROI": [],
    }
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

        for task in TASKS:
            if task not in TASKS_to_include:
                print(f"Skipping task {task} as it's not in the inclusion list.")
                continue
            d_values_all = []
            for session in SESSIONS:
                print(f"Processing task: {task}")
                SUBJECTS = find_available_subjects(
                    dFC_root=dFC_root,
                    task=task,
                    dFC_id=None,
                    session=session,
                )
                for subj in SUBJECTS:
                    for run in RUNS[task]:
                        try:
                            task_data = load_task_data(
                                roi_root=roi_root,
                                subj=subj,
                                task=task,
                                run=run,
                                session=session,
                            )
                        except:
                            continue

                        if run is None:
                            if session is None:
                                BOLD_file_name = "{subj_id}_{task}_time-series.npy"
                            else:
                                BOLD_file_name = (
                                    "{subj_id}_{session}_{task}_time-series.npy"
                                )
                        else:
                            if session is None:
                                BOLD_file_name = "{subj_id}_{task}_{run}_time-series.npy"
                            else:
                                BOLD_file_name = (
                                    "{subj_id}_{session}_{task}_{run}_time-series.npy"
                                )
                        try:
                            BOLD = data_loader.load_TS(
                                data_root=roi_root,
                                file_name=BOLD_file_name,
                                subj_id2load=subj,
                                task=task,
                                session=session,
                                run=run,
                            )
                        except Exception as e:
                            print(f"Error loading BOLD data: {e}")
                            continue
                        BOLD_data = BOLD.data  # np.ndarray (n_ROIs, n_TRs)

                        Fs_task = task_data["Fs_task"]
                        TR_task = 1 / Fs_task

                        TR_array = np.arange(0, BOLD_data.shape[1])
                        task_presence, indices = extract_task_presence(
                            event_labels=task_data["event_labels"],
                            TR_task=TR_task,
                            TR_mri=task_data["TR_mri"],
                            binary=True,
                            binarizing_method="GMM",
                            no_hrf=False,
                            TR_array=TR_array,
                        )

                        # if n_TRs do not match, align them
                        if BOLD_data.shape[1] != task_presence.shape[0]:
                            print(
                                f"Before alignment, shape of task_presence: {task_presence.shape}, shape of BOLD_data: {BOLD_data.shape}"
                            )
                            min_TRs = min(BOLD_data.shape[1], task_presence.shape[0])
                            task_presence = task_presence[:min_TRs]
                            BOLD_data = BOLD_data[:, :min_TRs]
                            print(
                                f"After alignment, shape of task_presence: {task_presence.shape}, shape of BOLD_data: {BOLD_data.shape}"
                            )
                            # also adjust indices
                            indices = [i for i in indices if i < min_TRs]
                        task_presence = task_presence[indices]  # (n_TRs,)
                        BOLD_data = BOLD_data[:, indices]  # (n_ROIs, n_TRs)

                        assert BOLD_data.shape[1] == task_presence.shape[0]

                        cohen_d = cohen_d_bold(X=BOLD_data.T, y=task_presence)
                        d_values_all.append(cohen_d)

            if len(d_values_all) == 0:
                print(f"No data found for task {task} in dataset {dataset}. Skipping.")
                continue
            d_values_all = np.array(d_values_all)  # (n_subjectsxrunsxsessions, n_ROIs)
            avg_d_values = np.nanmean(d_values_all, axis=0)  # (n_ROIs,)
            CohensD_across_task["d_values"].extend(avg_d_values)
            CohensD_across_task["task"].extend([task] * len(avg_d_values))
            CohensD_across_task["dataset"].extend([dataset] * len(avg_d_values))
            CohensD_across_task["ROI"].extend(BOLD.node_labels)

            # plot d values on a glass brain
            coords = BOLD.locs

            template_img = datasets.load_mni152_template()
            data = np.zeros(template_img.shape)
            affine = template_img.affine

            # Create a small sphere for each coordinate
            radius = 5  # in voxels
            for c, d in zip(coords, avg_d_values):
                ijk = np.round(nib.affines.apply_affine(np.linalg.inv(affine), c)).astype(
                    int
                )
                x, y, z = ijk
                for i in range(-radius, radius + 1):
                    for j in range(-radius, radius + 1):
                        for k in range(-radius, radius + 1):
                            if i**2 + j**2 + k**2 <= radius**2:
                                xi, yj, zk = x + i, y + j, z + k
                                if (
                                    (0 <= xi < data.shape[0])
                                    and (0 <= yj < data.shape[1])
                                    and (0 <= zk < data.shape[2])
                                ):
                                    data[xi, yj, zk] = d

            d_img = nib.Nifti1Image(data, affine)

            plotting.plot_glass_brain(
                d_img,
                display_mode="ortho",
                colorbar=True,
                plot_abs=False,
                cmap="coolwarm",
                vmax=np.max(avg_d_values),
            )

            plt.savefig(
                f"{output_root}/cohensd_region_{task}.png",
                dpi=120,
                bbox_inches="tight",
                pad_inches=0.1,
                format="png",
            )

            plt.close()

            # Load Schaefer atlas (100 parcels)
            schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=100)

            # atlas_img is the path to the NIfTI file; load it
            atlas_img = nib.load(schaefer["maps"])
            labels = schaefer["labels"]  # list of labels
            labels = [label.decode() for label in labels]
            # check that the labels match BOLD.node_labels
            assert all(
                i == j for i, j in zip(labels, BOLD.node_labels)
            ), "Labels do not match!"

            atlas_data = atlas_img.get_fdata()
            cohen_img_data = np.zeros(atlas_data.shape)

            for i, d in enumerate(avg_d_values):
                cohen_img_data[atlas_data == (i + 1)] = d  # labels start from 1

            cohen_img = nib.Nifti1Image(cohen_img_data, affine=atlas_img.affine)

            plotting.plot_glass_brain(
                cohen_img,
                display_mode="ortho",
                colorbar=True,
                cmap="coolwarm",
                plot_abs=False,
                vmax=np.max(avg_d_values),
            )

            plt.savefig(
                f"{output_root}/cohensd_voxel_{task}.png",
                dpi=120,
                bbox_inches="tight",
                pad_inches=0.1,
                format="png",
            )

            plt.close()

    # --- Across-task correlation with ML performance (ABSOLUTE Cohen's d) ---
    # Load ALL_ML_SCORES
    ALL_ML_SCORES = np.load(
        f"{multi_dataset_info['output_root']}/ML_results/ALL_ML_SCORES_real.npy",
        allow_pickle=True,
    ).item()

    embedding = "LE"
    metric = "SVM balanced accuracy"
    GROUP = "test"

    # Build dataframe if not already done
    DF = pd.DataFrame.from_dict(CohensD_across_task)

    # Use absolute Cohen's d
    DF["abs_d"] = DF["d_values"].abs()

    # Choose an order (sort tasks by their MAX |d| to align with Fig. 2)
    max_abs_per_task = (
        DF.groupby("task")["abs_d"]
        .max()
        .sort_values(ascending=False)
        .reset_index(name="abs_max")
    )

    df = pd.DataFrame.from_dict(ALL_ML_SCORES)
    df = df[df["task"].isin(TASKS_to_include)]
    df = df[(df["embedding"] == embedding) & (df["group"] == GROUP)]

    # alphabetical method order
    method_order = sorted(df["dFC method"].unique(), key=lambda s: s.lower())
    df["dFC method"] = pd.Categorical(
        df["dFC method"], categories=method_order, ordered=True
    )

    # ===== build BEST and ACROSS tables =====
    counts_task = df.groupby("task")["run"].nunique()
    multi_tasks = counts_task[counts_task > 1].index
    df_multi = df[
        df["task"].isin(multi_tasks)
    ]  # <- use this dataframe for ACROSS figures

    # BEST: one row per (task, method) with the winning run kept
    df_best = (
        df.sort_values(["task", "dFC method", metric], ascending=[True, True, False])
        .drop_duplicates(subset=["task", "dFC method"], keep="first")
        .rename(columns={metric: "score"})
    )

    # keep only the task and score columns
    df_best = df_best[["task", "score"]]

    # average over dFC methods and make a new dataframe
    df_best = df_best.groupby("task").agg({"score": "mean"}).reset_index()
    # find the correlation between max_abs_per_task["abs_max"] and df_best['score']
    merged = pd.merge(max_abs_per_task, df_best, on="task")

    # task="task-ppalocalizer" is an outlier, show it as a different color and exclude it from the correlation calculation
    outlier = merged[merged["task"] == "task-ppalocalizer"]
    merged = merged[merged["task"] != "task-ppalocalizer"]
    plt.style.use("seaborn-v0_8-paper")
    sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 1.2})
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="abs_max", y="score", data=merged, s=60, edgecolor="k", label="Task Paradigms"
    )
    sns.scatterplot(
        x="abs_max",
        y="score",
        data=outlier,
        color="orange",
        s=80,
        edgecolor="k",
        label="Outlier: task-ppalocalizer",
    )

    # fit and plot regression line
    sns.regplot(
        x="abs_max",
        y="score",
        data=merged,
        scatter=False,
        color="red",
        line_kws={"label": "Best fit"},
    )

    plt.xlabel("Max |Cohen's d| per Task", fontweight="bold", fontsize=14)
    plt.ylabel("SVM Balanced Accuracy", fontweight="bold", fontsize=14)
    plt.legend(fontsize=12)
    correlation = merged["abs_max"].corr(merged["score"])
    plt.text(
        0.05,
        0.95,
        f"correlation  r = {correlation:.2f}",
        transform=plt.gca().transAxes,
        fontsize=17,
        fontweight="bold",
        verticalalignment="top",
    )

    plt.xticks(fontweight="bold", fontsize=12)
    plt.yticks(fontweight="bold", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f"{output_root}/CohensdCorr.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
        format="png",
    )
    plt.close()

    # --- Across-task visualizations (ABSOLUTE Cohen's d) ---
    sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 1.2})
    sns.set_style("darkgrid")

    # Build dataframe if not already done
    DF = pd.DataFrame.from_dict(CohensD_across_task)

    # Use absolute Cohen's d
    DF["abs_d"] = DF["d_values"].abs()

    # Choose an order (sort tasks by their MAX |d| to align with Fig. 2)
    max_abs_per_task = (
        DF.groupby("task")["abs_d"]
        .max()
        .sort_values(ascending=False)
        .reset_index(name="abs_max")
    )
    task_order = max_abs_per_task["task"].tolist()

    # Dynamic width so labels don't collide (0.6 inch per task, min 14 inches)
    fig_width = max(14, 0.6 * len(task_order))

    # -------- Figure 1: Boxplot of |Cohen's d| per task with individual samples --------
    plt.figure(figsize=(fig_width, 7))

    # Boxplot (hide outliers to avoid double-plotting with the samples)
    ax = sns.boxplot(
        data=DF, x="task", y="abs_d", order=task_order, showfliers=False, width=0.6
    )

    # Overlay individual samples (one point per ROI sample)
    sns.stripplot(
        data=DF,
        x="task",
        y="abs_d",
        order=task_order,
        dodge=False,
        jitter=0.25,
        size=2,
        alpha=0.45,
        ax=ax,
    )

    ax.set_xlabel("Task")
    ax.set_ylabel("|Cohen's d|")
    ax.set_ylim(bottom=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        f"{output_root}/CohensD_abs_boxplot_with_samples_per_task.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
        format="png",
    )
    plt.close()

    # -------- Figure 2: Max |Cohen's d| across ROIs per task --------
    plt.figure(figsize=(fig_width, 6))

    ax = sns.barplot(data=max_abs_per_task, x="task", y="abs_max", order=task_order)

    # Optional: annotate bars with values (trim to 2 decimals)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 2),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Task")
    ax.set_ylabel("Max |Cohen's d|")
    ax.set_ylim(bottom=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        f"{output_root}/CohensD_abs_max_per_task.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
        format="png",
    )
    plt.close()
