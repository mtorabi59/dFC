import argparse
import json
import os
import traceback

import numpy as np

from pydfc.ml_utils import (
    cluster_for_visual,
    extract_task_features,
    task_presence_classification,
    task_presence_clustering,
)

#######################################################################################


def run_task_features_extraction(
    TASKS,
    RUNS,
    SESSIONS,
    roi_root,
    dFC_root,
    output_root,
):
    for session in SESSIONS:

        # Extract task features without HRF effect
        task_features = extract_task_features(
            TASKS=TASKS,
            RUNS=RUNS,
            session=session,
            roi_root=roi_root,
            dFC_root=dFC_root,
            no_hrf=True,
        )

        # Extract task features with HRF effect
        task_features_hrf = extract_task_features(
            TASKS=TASKS,
            RUNS=RUNS,
            session=session,
            roi_root=roi_root,
            dFC_root=dFC_root,
            no_hrf=False,
        )

        if session is None:
            folder = f"{output_root}/task_features"
        else:
            folder = f"{output_root}/task_features/{session}"
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
        except OSError as err:
            print(err)
        try:
            if not os.path.exists(f"{folder}/task_features.npy"):
                np.save(f"{folder}/task_features.npy", task_features)
            if not os.path.exists(f"{folder}/task_features_hrf.npy"):
                np.save(f"{folder}/task_features_hrf.npy", task_features_hrf)
        except OSError as err:
            print(err)


def run_classification(
    dFC_id,
    TASKS,
    RUNS,
    SESSIONS,
    roi_root,
    dFC_root,
    output_root,
    dynamic_pred="no",
    normalize_dFC=True,
):
    for session in SESSIONS:
        if not session is None:
            print(f"=================== {session} ===================")

        ML_scores = {}
        for task_id, task in enumerate(TASKS):
            for run in RUNS[task]:
                try:
                    ML_scores_new = task_presence_classification(
                        task=task,
                        dFC_id=dFC_id,
                        roi_root=roi_root,
                        dFC_root=dFC_root,
                        run=run,
                        session=session,
                        dynamic_pred=dynamic_pred,
                        normalize_dFC=normalize_dFC,
                    )
                    for key in ML_scores_new:
                        if key not in ML_scores:
                            ML_scores[key] = list()
                        ML_scores[key].extend(ML_scores_new[key])
                except Exception as e:
                    print(
                        f"Error in task presence classification for {session} {task} {run}: {e}"
                    )
                    traceback.print_exc()

        if session is None:
            folder = f"{output_root}/classification"
        else:
            folder = f"{output_root}/classification/{session}"
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
        except OSError as err:
            print(err)

        np.save(f"{folder}/ML_scores_classify_{dFC_id}.npy", ML_scores)


def run_clustering(
    dFC_id,
    TASKS,
    RUNS,
    SESSIONS,
    roi_root,
    dFC_root,
    output_root,
    normalize_dFC=True,
):
    for session in SESSIONS:
        if not session is None:
            print(f"=================== {session} ===================")
        clustering_scores = {
            "subj_id": list(),
            "task": list(),
            "run": list(),
            "dFC method": list(),
            "Kmeans ARI": list(),
            "SI": list(),
            "embedding": list(),
        }

        clustering_RESULTS = {}
        for task_id, task in enumerate(TASKS):
            clustering_RESULTS[task] = {}
            for run in RUNS[task]:
                try:
                    clustering_RESULTS_new, clustering_scores_new = (
                        task_presence_clustering(
                            task=task,
                            dFC_id=dFC_id,
                            roi_root=roi_root,
                            dFC_root=dFC_root,
                            run=run,
                            session=session,
                            normalize_dFC=normalize_dFC,
                        )
                    )
                    if run is None:
                        clustering_RESULTS[task] = clustering_RESULTS_new
                    else:
                        clustering_RESULTS[task][run] = clustering_RESULTS_new
                    for key in clustering_scores:
                        clustering_scores[key].extend(clustering_scores_new[key])
                except Exception as e:
                    print(
                        f"Error in task presence clustering for {session} {task} {run}: {e}"
                    )
                    traceback.print_exc()

        if session is None:
            folder = f"{output_root}/clustering"
        else:
            folder = f"{output_root}/clustering/{session}"
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
        except OSError as err:
            print(err)
        np.save(f"{folder}/clustering_RESULTS_{dFC_id}.npy", clustering_RESULTS)

        np.save(f"{folder}/clustering_scores_{dFC_id}.npy", clustering_scores)


def run_clustering_for_visual(
    dFC_id,
    TASKS,
    RUNS,
    SESSIONS,
    roi_root,
    dFC_root,
    output_root,
    normalize_dFC=True,
):
    for session in SESSIONS:
        if not session is None:
            print(f"=================== {session} ===================")

        for task_id, task in enumerate(TASKS):
            for run in RUNS[task]:
                try:
                    (
                        centroids_mat,
                        measure_name,
                        co_occurrence_matrix,
                        cluster_label_percentage,
                        task_label_percentage,
                    ) = cluster_for_visual(
                        task=task,
                        dFC_id=dFC_id,
                        roi_root=roi_root,
                        dFC_root=dFC_root,
                        run=run,
                        session=session,
                        normalize_dFC=normalize_dFC,
                    )

                    centroids = {
                        "centroids_mat": centroids_mat,
                        "co_occurrence_matrix": co_occurrence_matrix,
                        "cluster_label_percentage": cluster_label_percentage,
                        "task_label_percentage": task_label_percentage,
                    }

                    # save the centroids
                    suffix = "centroids"
                    if session is not None:
                        suffix = f"{suffix}_{session}"
                    suffix = f"{suffix}_{task}"
                    if run is not None:
                        suffix = f"{suffix}_{run}"
                    suffix = f"{suffix}_{measure_name}"

                    if session is None:
                        folder = f"{output_root}/centroids"
                    else:
                        folder = f"{output_root}/centroids/{session}"
                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    np.save(
                        f"{folder}/{suffix}.npy",
                        centroids,
                    )

                except Exception as e:
                    print(
                        f"Error in clustering for visualization for {session} {task} {run}: {e}"
                    )
                    traceback.print_exc()


#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to apply Machine Learning on dFC results to predict task presence.
    """

    parser = argparse.ArgumentParser(description=HELPTEXT)

    parser.add_argument("--dataset_info", type=str, help="path to dataset info file")

    args = parser.parse_args()

    dataset_info_file = args.dataset_info

    # Read dataset info
    with open(dataset_info_file, "r") as f:
        dataset_info = json.load(f)

    print("Task presence prediction started ...")

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
        main_root = dataset_info["main_root"].replace(
            "{dataset}", dataset_info["dataset"]
        )
    else:
        main_root = dataset_info["main_root"]

    if "{main_root}" in dataset_info["roi_root"]:
        roi_root = dataset_info["roi_root"].replace("{main_root}", main_root)
    else:
        roi_root = dataset_info["roi_root"]

    if "{main_root}" in dataset_info["dFC_root"]:
        dFC_root = dataset_info["dFC_root"].replace("{main_root}", main_root)
    else:
        dFC_root = dataset_info["dFC_root"]

    if "{main_root}" in dataset_info["ML_root"]:
        ML_root = dataset_info["ML_root"].replace("{main_root}", main_root)
    else:
        ML_root = dataset_info["ML_root"]

    # The task feature extraction will be executed multiple times in parallel redundantly
    try:
        run_task_features_extraction(
            TASKS=TASKS,
            RUNS=RUNS,
            SESSIONS=SESSIONS,
            roi_root=roi_root,
            dFC_root=dFC_root,
            output_root=ML_root,
        )
    except Exception as e:
        print(f"Error in task features extraction: {e}")
        traceback.print_exc()
    print("Task features extraction finished.")

    job_id = os.getenv("SGE_TASK_ID")  # for SGE
    if job_id is None:
        job_id = os.getenv("SLURM_ARRAY_TASK_ID")  # for SLURM
    job_id = int(job_id)
    dFC_id = job_id - 1  # TASK_ID starts from 1 not 0

    print(f"Task presence classification started for dFC ID {dFC_id}...")
    try:
        run_classification(
            dFC_id=dFC_id,
            TASKS=TASKS,
            RUNS=RUNS,
            SESSIONS=SESSIONS,
            roi_root=roi_root,
            dFC_root=dFC_root,
            output_root=ML_root,
            dynamic_pred="no",
            normalize_dFC=True,
        )
    except Exception as e:
        print(f"Error in classification for dFC ID {dFC_id}: {e}")
        traceback.print_exc()
    print(f"Task presence classification finished for dFC ID {dFC_id}.")
    # print(f"Task presence clustering started for dFC ID {dFC_id} ...")
    # try:
    #     run_clustering(
    #         dFC_id=dFC_id,
    #         TASKS=TASKS,
    #         RUNS=RUNS,
    #         SESSIONS=SESSIONS,
    #         roi_root=roi_root,
    #         dFC_root=dFC_root,
    #         output_root=ML_root,
    #         normalize_dFC=True,
    #     )
    # except Exception as e:
    #     print(f"Error in clustering for dFC ID {dFC_id}: {e}")
    #     traceback.print_exc()

    # print(f"Task presence clustering finished for dFC ID {dFC_id}.")

    print(f"Clustering for visualization started for dFC ID {dFC_id} ...")
    try:
        run_clustering_for_visual(
            dFC_id=dFC_id,
            TASKS=TASKS,
            RUNS=RUNS,
            SESSIONS=SESSIONS,
            roi_root=roi_root,
            dFC_root=dFC_root,
            output_root=ML_root,
            normalize_dFC=True,
        )
    except Exception as e:
        print(f"Error in clustering for visualization for dFC ID {dFC_id}: {e}")
        traceback.print_exc()

    print(f"Clustering for visualization finished for dFC ID {dFC_id}.")

    print(f"Task presence prediction finished for dFC ID {dFC_id}.")

#######################################################################################
