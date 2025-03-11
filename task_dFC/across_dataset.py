import argparse
import json
import os
import traceback

import numpy as np

from pydfc.dfc_utils import dFC_mat2vec

#######################################################################################


def get_dataset_info(main_root, dataset):
    # get the dataset_info.json
    dataset_info_path = os.path.join(main_root, dataset, "codes", "dataset_info.json")
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)

    TASKS = dataset_info["TASKS"]
    if "RUNS" in dataset_info:
        RUNS = dataset_info["RUNS"]
    else:
        RUNS = None
    if RUNS is None:
        RUNS = {task: [None] for task in TASKS}

    if "SESSIONS" in dataset_info:
        SESSIONS = dataset_info["SESSIONS"]
    else:
        SESSIONS = None
    if SESSIONS is None:
        SESSIONS = [None]

    if "{dataset}" in dataset_info["main_root"]:
        dataset_main_root = dataset_info["main_root"].replace("{dataset}", dataset)
    else:
        dataset_main_root = dataset_info["main_root"]

    if "{main_root}" in dataset_info["ML_root"]:
        ML_root = dataset_info["ML_root"].replace("{main_root}", dataset_main_root)
    else:
        ML_root = dataset_info["ML_root"]

    return TASKS, RUNS, SESSIONS, ML_root


def plot_affinity_matrix(centroids_mat, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.neighbors import kneighbors_graph

    fig_dpi = 120
    fig_bbox_inches = "tight"
    fig_pad = 0.1
    save_fig_format = "png"  # pdf, png,

    X = np.array(centroids_mat)  # shape: (n_centroids, n_regions*(n_regions-1)/2)

    affinity_matrix = kneighbors_graph(
        X,
        n_neighbors=125,
        mode="connectivity",
        include_self=False,
        metric="correlation",
    )

    # plot a heatmap of the affinity matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(affinity_matrix.toarray())
    if save_path is not None:
        plt.savefig(
            save_path,
            format=save_fig_format,
            bbox_inches=fig_bbox_inches,
            dpi=fig_dpi,
            pad_inches=fig_pad,
        )
    plt.close()


def run_across_dataset_analysis(main_root, DATASETS):
    """_summary_

    Parameters
    ----------
    main_root : str
        the main root of the datasets
    DATASETS : list
        the list of datasets
    """
    RESULTS = {
        "centroids_mat": [],
        "task": [],
        "run": [],
        "session": [],
        "measure_name": [],
        "dataset": [],
    }
    for dataset in DATASETS:

        TASKS, RUNS, SESSIONS, ML_root = get_dataset_info(main_root, dataset)

        # Load data
        # look for all centroids files
        # dataset_root/ML_root/centroids/session/centroids_{session}_{task}_{run}_{measure_name}.npy
        for session in SESSIONS:
            if session is None:
                input_path = os.path.join(ML_root, "centroids")
            else:
                input_path = os.path.join(ML_root, "centroids", session)
            ALL_CENTROIDS_FILES = os.listdir(input_path)
            ALL_CENTROIDS_FILES = [f for f in ALL_CENTROIDS_FILES if "centroids_" in f]
            for task in TASKS:
                for run in RUNS[task]:
                    centroids_files = [f for f in ALL_CENTROIDS_FILES if f"_{task}_" in f]
                    if run is not None:
                        centroids_files = [f for f in centroids_files if f"_{run}_" in f]
                    if session is not None:
                        centroids_files = [
                            f for f in centroids_files if f"_{session}_" in f
                        ]
                    for centroids_file in centroids_files:
                        measure_name = centroids_file.split("_")[-1].replace(".npy", "")
                        centroids = np.load(os.path.join(input_path, centroids_file))
                        centroids_mat = centroids[
                            "centroids_mat"
                        ]  # shape: (n_clusters, n_regions, n_regions)
                        centroids_mat = dFC_mat2vec(
                            centroids_mat
                        )  # shape: (n_clusters, n_regions*(n_regions-1)/2)
                        for i in range(centroids_mat.shape[0]):
                            RESULTS["centroids_mat"].append(centroids_mat[i])
                            RESULTS["task"].append(task)
                            RESULTS["run"].append(run)
                            RESULTS["session"].append(session)
                            RESULTS["measure_name"].append(measure_name)
                            RESULTS["dataset"].append(dataset)

    # give statistics
    print(f"Number of centroids: {len(RESULTS['centroids_mat'])}")
    print(f"Number of tasks: {len(set(RESULTS['task']))}")
    print(f"Number of measure_names: {len(set(RESULTS['measure_name']))}")
    print(f"Number of datasets: {len(set(RESULTS['dataset']))}")

    # plot the affinity matrix
    plot_affinity_matrix(RESULTS["centroids_mat"], save_path="affinity_matrix.png")


#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to run across-dataset analysis on dFC results.
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

    print("Multi-Dataset Analysis started ...")

    main_root = multi_dataset_info["main_root"]
    DATASETS = multi_dataset_info["DATASETS"]

    try:
        run_across_dataset_analysis()
    except Exception as e:
        print(f"Error in run_across_dataset_analysis: {e}")
        traceback.print_exc()
    print("run_across_dataset_analysis finished.")

    print("Multi-Dataset Analysis finished.")

#######################################################################################
