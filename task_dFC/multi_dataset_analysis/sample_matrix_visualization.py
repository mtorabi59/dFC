import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import ttest_ind

from pydfc.ml_utils import (
    dFC_feature_extraction,
    embed_dFC_features,
    find_available_subjects,
    process_SB_features,
)

use_raw_features = False  # if True, use raw dFC features instead of embedded features
normalize_dFC = True
FCS_proba_for_SB = True
train_test_ratio = 0.8
embedding = "LE"

if use_raw_features:
    raw_or_embedded = "_raw"
else:
    raw_or_embedded = ""

#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to visualize the feature-sample matrix for each dataset, task, and dFC measure.
    """

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

    output_root = f"{multi_dataset_info['output_root']}/feature-sample/{simul_or_real}"
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for dataset in DATASETS:
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

        for dFC_id in range(0, 7):
            DATA = {}
            for session in SESSIONS[:1]:  # Only process the first session
                for task_id, task in enumerate(TASKS):
                    if task not in TASKS_to_include:
                        print(f"Skipping task: {task} as it is not in TASKS_to_include.")
                        continue
                    for run in RUNS[task][:1]:  # Only process the first run
                        print(
                            f"Processing dataset: {dataset}, task: {task}, run: {run}, session: {session}, dFC_id: {dFC_id}"
                        )

                        SUBJECTS = find_available_subjects(
                            dFC_root=dFC_root,
                            task=task,
                            run=run,
                            session=session,
                            dFC_id=dFC_id,
                        )

                        # randomly select train_test_ratio of the subjects for training
                        # and rest for testing using numpy.random.choice
                        train_subjects = np.random.choice(
                            SUBJECTS, int(train_test_ratio * len(SUBJECTS)), replace=False
                        )
                        test_subjects = np.setdiff1d(SUBJECTS, train_subjects)
                        print(
                            f"Number of train subjects: {len(train_subjects)} and test subjects: {len(test_subjects)}"
                        )

                        (
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            subj_label_train,
                            subj_label_test,
                            measure_name,
                        ) = dFC_feature_extraction(
                            task=task,
                            train_subjects=train_subjects,
                            test_subjects=test_subjects,
                            dFC_id=dFC_id,
                            roi_root=roi_root,
                            dFC_root=dFC_root,
                            run=run,
                            session=session,
                            dynamic_pred="no",
                            normalize_dFC=normalize_dFC,
                            FCS_proba_for_SB=FCS_proba_for_SB,  # for state-based dFC features, we use FCS_proba
                        )

                        if measure_name is None:
                            print(
                                f"Skipping dataset: {dataset}, task: {task}, run: {run}, session: {session}, dFC_id: {dFC_id} due to no measure_name."
                            )
                            continue

                        measure_is_state_based = None
                        if measure_name in ["SlidingWindow", "Time-Freq"]:
                            measure_is_state_based = False
                        elif measure_name in [
                            "CAP",
                            "Clustering",
                            "ContinuousHMM",
                            "DiscreteHMM",
                            "Windowless",
                        ]:
                            measure_is_state_based = True
                        else:
                            # raise error
                            raise ValueError(f"Unknown measure name: {measure_name}")

                        if measure_is_state_based:
                            X_train_embedded = process_SB_features(
                                X=X_train, measure_name=measure_name
                            )
                            X_test_embedded = process_SB_features(
                                X=X_test, measure_name=measure_name
                            )
                        else:
                            # embed dFC features
                            try:
                                X_train_embedded, X_test_embedded = embed_dFC_features(
                                    train_subjects=train_subjects,
                                    test_subjects=test_subjects,
                                    X_train=X_train,
                                    X_test=X_test,
                                    y_train=y_train,
                                    y_test=y_test,
                                    subj_label_train=subj_label_train,
                                    subj_label_test=subj_label_test,
                                    embedding=embedding,
                                    n_components="auto",
                                    n_neighbors_LE=125,
                                    LE_embedding_method="embed+procrustes",
                                    measure_is_state_based=measure_is_state_based,
                                )
                                assert (
                                    X_train_embedded.shape[0] == y_train.shape[0]
                                ), "Number of samples do not match."
                                assert (
                                    X_test_embedded.shape[0] == y_test.shape[0]
                                ), "Number of samples do not match."
                            except Exception as e:
                                print(
                                    f"Error in embedding dFC features with {embedding}: {e}"
                                )
                                X_train_embedded = None
                                X_test_embedded = None

                        assert (
                            task not in DATA
                        ), f"Task {task} already exists in DATA. Overwriting."
                        DATA[task] = {
                            "X_train": X_train,
                            "X_test": X_test,
                            "X_train_embedded": X_train_embedded,
                            "X_test_embedded": X_test_embedded,
                            "y_train": y_train,
                            "y_test": y_test,
                            "subj_label_train": subj_label_train,
                            "subj_label_test": subj_label_test,
                            "measure_name": measure_name,
                        }
            # save the data
            # save each task in a separate file and name the file as the task name, measure name, and dataset name
            for task in DATA.keys():
                if use_raw_features:
                    X_train = DATA[task]["X_train"]
                    X_test = DATA[task]["X_test"]
                else:
                    X_train = DATA[task]["X_train_embedded"]
                    X_test = DATA[task]["X_test_embedded"]
                y_train = DATA[task]["y_train"]
                y_test = DATA[task]["y_test"]
                subj_label_train = DATA[task]["subj_label_train"]
                subj_label_test = DATA[task]["subj_label_test"]
                measure_name = DATA[task]["measure_name"]

                if X_train is None or X_test is None:
                    print(f"Skipping task {task} due to embedding error.")
                    continue

                # np.save(f"{output_root}/processed_data/{dataset}_{task}_{measure_name}.npy", DATA[task])

                SORT_FEATURES = True
                ZSCORE = True
                V_RANGE = 2.0  # heatmap color range after z-scoring

                for group, X, y in zip(
                    ["train", "test"], [X_train, X_test], [y_train, y_test]
                ):

                    # X: (n_samples, n_features) = LE-transformed dFC features
                    # y: (n_samples,) binary (0=rest, 1=task)
                    # Optional: z-score features so the imshow uses comparable scales
                    Xz = X.copy().astype(float)
                    if ZSCORE:
                        Xz = (Xz - Xz.mean(0)) / (Xz.std(0) + 1e-8)

                    # --- supervised feature order ---
                    t, p = ttest_ind(Xz[y == 1], Xz[y == 0], axis=0, equal_var=False)
                    if SORT_FEATURES:
                        if group == "train":
                            # if test, use train's t-stat order
                            col_order = np.argsort(
                                -np.abs(t)
                            )  # strongest class contrast first
                    else:
                        col_order = np.arange(Xz.shape[1])  # original order

                    # --- row order: cluster within each class (cosine is nice for patterns) ---
                    def order_rows(A):
                        if len(A) <= 2:
                            return np.arange(len(A))
                        return leaves_list(linkage(A, method="average", metric="cosine"))

                    rest_idx = np.where(y == 0)[0]
                    task_idx = np.where(y == 1)[0]
                    rest_order = rest_idx[order_rows(Xz[rest_idx])]
                    task_order = task_idx[order_rows(Xz[task_idx])]
                    row_order = np.r_[rest_order, task_order]
                    split = len(rest_order)

                    # --- plot: main heatmap + class strip + top contrast bar ---
                    fig = plt.figure(figsize=(20, 10))
                    gs = fig.add_gridspec(
                        nrows=2,
                        ncols=2,
                        height_ratios=[0.07, 1],
                        width_ratios=[1, 0.03],
                        hspace=0.05,
                        wspace=0.05,
                    )
                    ax_top = fig.add_subplot(gs[0, 0])
                    ax_main = fig.add_subplot(gs[1, 0])
                    ax_lab = fig.add_subplot(gs[1, 1])

                    # top bar: signed t-stat per feature (same column order)
                    t_ord = t[col_order]
                    m = np.nanmax(np.abs(t_ord))
                    im_top = ax_top.imshow(
                        t_ord[None, :], aspect="auto", cmap="coolwarm", vmin=-m, vmax=m
                    )
                    ax_top.set_xticks([])
                    ax_top.set_yticks([])
                    ax_top.set_title(
                        "Feature contrast (t-stat): task − rest", fontsize=10
                    )

                    # main heatmap
                    im = ax_main.imshow(
                        Xz[row_order][:, col_order],
                        aspect="auto",
                        cmap="coolwarm",
                        vmin=-V_RANGE,
                        vmax=V_RANGE,
                    )
                    ax_main.axhline(
                        split - 0.5, color="k", lw=1
                    )  # separator between rest/task
                    ax_main.set_ylabel("samples")
                    ax_main.set_xticks([])

                    # class strip (right)
                    cmap_lbl = ListedColormap(
                        [[0.85, 0.85, 0.85], [0.25, 0.5, 0.9]]
                    )  # gray=rest, blue=task
                    ax_lab.imshow(
                        y[row_order][:, None],
                        aspect="auto",
                        cmap=cmap_lbl,
                        vmin=0,
                        vmax=1,
                    )
                    ax_lab.set_xticks([])
                    ax_lab.set_yticks([])
                    ax_lab.set_title("class")

                    # --- vertical centers in the main heatmap (data coords) ---
                    n_rows = Xz.shape[0]
                    y_center_rest = (split - 1) / 2.0 if split > 0 else -0.5
                    y_center_task = (
                        (split + (n_rows - 1)) / 2.0 if n_rows > split else split - 0.5
                    )

                    # centers (already computed)
                    # y_center_rest, y_center_task
                    # x positions on the strip:
                    x_right_ax = 1.0 - 0.02

                    # blended: x in ax_lab axes coords, y in ax_main data coords
                    blended = mtransforms.blended_transform_factory(
                        ax_lab.transAxes, ax_main.transData
                    )

                    # left label: place at strip edge, nudge  +6 pts right
                    ax_main.annotate(
                        "rest (0)",
                        xy=(x_right_ax, y_center_rest),
                        xycoords=blended,
                        xytext=(6, 0),
                        textcoords="offset points",  # -> to the right of the strip
                        ha="left",
                        va="center",
                        fontsize=9,
                        zorder=7,
                    )

                    # right label: place at strip edge, nudge  +6 pts further right
                    ax_main.annotate(
                        "task (1)",
                        xy=(x_right_ax, y_center_task),
                        xycoords=blended,
                        xytext=(6, 0),
                        textcoords="offset points",  # -> outside the strip on the right
                        ha="left",
                        va="center",
                        fontsize=9,
                        zorder=7,
                    )

                    # colorbar (small)
                    cax = fig.add_axes(
                        [0.12, 0.06, 0.3, 0.02]
                    )  # x, y, w, h in figure coords
                    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
                    cb.set_label("z-scored feature value", fontsize=9)
                    cb.ax.tick_params(labelsize=8)

                    # if the folder does not exist, create it
                    if not os.path.exists(f"feature-sample/{measure_name}"):
                        os.makedirs(f"feature-sample/{measure_name}")

                    plt.savefig(
                        f"{output_root}/{measure_name}/feature-sample_{simul_or_real}_sorted-samples_{task}_{group}{raw_or_embedded}.png",
                        dpi=150,
                        bbox_inches="tight",
                        pad_inches=0.2,
                        format="png",
                    )
                    plt.close()

                    # plot unsorted version as well
                    print(f"Embedded shape: {X.shape}")

                    # plot X_embedded and y in an imshow as subplots
                    if group == "train":
                        w = 30
                        h = 10
                    elif group == "test":
                        w = 10
                        h = 10
                    # fig, axs = plt.subplots(2, 1, figsize=(w, h))
                    # make bottom subplot skinny like a color strip
                    fig, axs = plt.subplots(
                        2,
                        1,
                        figsize=(w, h),
                        sharex=True,
                        gridspec_kw={"height_ratios": [15, 1], "hspace": 0.1},
                    )

                    split = np.sum(y == 0)  # number of rest samples
                    axs[0].imshow(X.T, aspect="auto", origin="lower", cmap="seismic")
                    axs[0].set_title("LE Embedded Features")
                    axs[0].set_xlabel("Sample")
                    axs[0].set_ylabel("Feature")

                    axs[1].imshow(
                        y[np.newaxis, :], aspect=20, origin="lower", cmap="seismic"
                    )
                    axs[1].set_title("Target")
                    axs[1].set_xticks([])
                    axs[1].set_yticks([])
                    plt.savefig(
                        f"{output_root}/{measure_name}/feature-sample_{simul_or_real}_unsorted_{task}_{group}{raw_or_embedded}.png",
                        dpi=150,
                        bbox_inches="tight",
                        pad_inches=0.2,
                        format="png",
                    )
                    plt.close()

                    # sort the data such that y=1 samples come first
                    sort_indices = np.argsort(y, kind="stable")
                    X = X[sort_indices]
                    y = y[sort_indices]

                    # plot X_embedded and y in an imshow as subplots
                    # fig, axs = plt.subplots(2, 1, figsize=(w, h))
                    # make bottom subplot skinny like a color strip
                    fig, axs = plt.subplots(
                        2,
                        1,
                        figsize=(w, h),
                        sharex=True,
                        gridspec_kw={"height_ratios": [15, 1], "hspace": 0.1},
                    )

                    axs[0].imshow(X.T, aspect="auto", origin="lower", cmap="seismic")
                    axs[0].axvline(
                        split - 0.5, color="k", lw=3
                    )  # separator between rest/task
                    axs[0].set_title("LE Embedded Features")
                    axs[0].set_xlabel("Sample")
                    axs[0].set_ylabel("Feature")

                    axs[1].imshow(
                        y[np.newaxis, :], aspect=20, origin="lower", cmap="seismic"
                    )
                    axs[1].set_title("Target")
                    axs[1].set_xticks([])
                    axs[1].set_yticks([])
                    plt.savefig(
                        f"{output_root}/{measure_name}/feature-sample_{simul_or_real}_sorted-label_{task}_{group}{raw_or_embedded}.png",
                        dpi=150,
                        bbox_inches="tight",
                        pad_inches=0.2,
                        format="png",
                    )
                    plt.close()
