import argparse
import json
import os
import traceback

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from pydfc.ml_utils import (
    dFC_feature_extraction,
    embed_dFC_features,
    extract_task_features,
    find_available_subjects,
    task_presence_classification,
    task_presence_clustering,
)

#######################################################################################


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
        ML_scores = {
            "subj_id": list(),
            "group": list(),
            "task": list(),
            "run": list(),
            "dFC method": list(),
            "Logistic regression accuracy": list(),
            "KNN accuracy": list(),
            # "Random Forest accuracy": list(),
            # "Gradient Boosting accuracy": list(),
        }

        ML_RESULT = {}
        for task_id, task in enumerate(TASKS):
            ML_RESULT[task] = {}
            for run in RUNS[task]:
                ML_RESULT_new, ML_scores_new = task_presence_classification(
                    task=task,
                    dFC_id=dFC_id,
                    roi_root=roi_root,
                    dFC_root=dFC_root,
                    run=run,
                    session=session,
                    dynamic_pred=dynamic_pred,
                    normalize_dFC=normalize_dFC,
                )
                if run is None:
                    ML_RESULT[task] = ML_RESULT_new
                else:
                    ML_RESULT[task][run] = ML_RESULT_new
                for key in ML_scores:
                    ML_scores[key].extend(ML_scores_new[key])

        if session is None:
            folder = f"{output_root}"
        else:
            folder = f"{output_root}/{session}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(f"{folder}/ML_RESULT_{dFC_id}.npy", ML_RESULT)

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
        }

        clustering_RESULTS = {}
        for task_id, task in enumerate(TASKS):
            clustering_RESULTS[task] = {}
            for run in RUNS[task]:
                clustering_RESULTS_new, clustering_scores_new = task_presence_clustering(
                    task=task,
                    dFC_id=dFC_id,
                    roi_root=roi_root,
                    dFC_root=dFC_root,
                    run=run,
                    session=session,
                    normalize_dFC=normalize_dFC,
                )
                if run is None:
                    clustering_RESULTS[task] = clustering_RESULTS_new
                else:
                    clustering_RESULTS[task][run] = clustering_RESULTS_new
                for key in clustering_scores:
                    clustering_scores[key].extend(clustering_scores_new[key])

        if session is None:
            folder = f"{output_root}"
        else:
            folder = f"{output_root}/{session}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(f"{folder}/clustering_RESULTS_{dFC_id}.npy", clustering_RESULTS)

        np.save(f"{folder}/clustering_scores_{dFC_id}.npy", clustering_scores)


def run_task_paradigm_clustering(
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
        # find SUBJECTS common to all tasks
        for task_id, task in enumerate(TASKS):
            if task_id == 0:
                SUBJECTS = find_available_subjects(
                    dFC_root=dFC_root, task=task, dFC_id=dFC_id
                )
            else:
                SUBJECTS = np.intersect1d(
                    SUBJECTS,
                    find_available_subjects(dFC_root=dFC_root, task=task, dFC_id=dFC_id),
                )
        print(f"Number of subjects: {len(SUBJECTS)}")

        X = None
        y = None
        subj_label = None
        measure_name = None
        for task_id, task in enumerate(TASKS):
            for run in RUNS[task]:
                X_new, _, _, _, subj_label_new, _, measure_name_new = (
                    dFC_feature_extraction(
                        task=task,
                        train_subjects=SUBJECTS,
                        test_subjects=[],
                        dFC_id=dFC_id,
                        roi_root=roi_root,
                        dFC_root=dFC_root,
                        run=run,
                        session=session,
                        dynamic_pred="no",
                        normalize_dFC=normalize_dFC,
                    )
                )

                if measure_name is not None:
                    assert (
                        measure_name == measure_name_new
                    ), "dFC measure is not consistent."
                else:
                    measure_name = measure_name_new

                y_new = np.ones(X_new.shape[0]) * task_id
                if X is None and y is None:
                    X = X_new
                    y = y_new
                    subj_label = subj_label_new
                else:
                    X = np.concatenate((X, X_new), axis=0)
                    y = np.concatenate((y, y_new), axis=0)
                    subj_label = np.concatenate((subj_label, subj_label_new), axis=0)

        assert X.shape[0] == y.shape[0], "Number of samples do not match."
        assert X.shape[0] == subj_label.shape[0], "Number of samples do not match."

        # rearrange the order of the samples so that the samples of the same subject are together
        idx = np.argsort(subj_label)
        X = X[idx, :]
        y = y[idx]
        subj_label = subj_label[idx]

        # embed dFC features
        X, _ = embed_dFC_features(
            train_subjects=SUBJECTS,
            test_subjects=[],
            X_train=X,
            X_test=None,
            y_train=y,
            y_test=None,
            subj_label_train=subj_label,
            subj_label_test=None,
            embedding="LE",
            n_components=30,
            n_neighbors_LE=125,
            LE_embedding_method="embed+procrustes",
        )

        # clustering
        # apply kmeans clustering to dFC features

        n_clusters = len(TASKS)  # corresponding to task paradigms

        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=5)
        labels_pred = kmeans.fit_predict(X_normalized)

        # ARI score
        print(f"ARI score: {adjusted_rand_score(y, labels_pred)}")

        # # visualize clustering centroids
        # centroids = kmeans.cluster_centers_
        # centroids = pca.inverse_transform(centroids)
        # centroids = scaler.inverse_transform(centroids)
        # n_regions = int((1 + np.sqrt(1 + 8 * centroids.shape[1])) / 2)
        # centroids_mat = dFC_vec2mat(centroids, n_regions)

        task_paradigm_clstr_RESULTS = {
            "dFC_method": measure_name,
            "StandardScaler": scaler,
            "kmeans": kmeans,
            "ARI": adjusted_rand_score(y, labels_pred),
            "SI": silhouette_score(X_normalized, y),
            # "centroids": centroids_mat,
            "task_paradigms": TASKS,
        }

        if session is None:
            folder = f"{output_root}"
        else:
            folder = f"{output_root}/{session}"
        if not os.path.exists(folder):
            os.makedirs(folder)

        np.save(
            f"{folder}/task_paradigm_clstr_RESULTS_{dFC_id}.npy",
            task_paradigm_clstr_RESULTS,
        )


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

    extract_task_features(
        TASKS=TASKS,
        RUNS=RUNS,
        SESSIONS=SESSIONS,
        roi_root=roi_root,
        dFC_root=dFC_root,
        output_root=ML_root,
    )
    print("Task features extraction finished.")

    job_id = int(os.getenv("SGE_TASK_ID"))
    dFC_id = job_id - 1  # SGE_TASK_ID starts from 1 not 0

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
    print(f"Task presence clustering started for dFC ID {dFC_id} ...")
    try:
        run_clustering(
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
        print(f"Error in clustering for dFC ID {dFC_id}: {e}")
        traceback.print_exc()

    print(f"Task presence clustering finished for dFC ID {dFC_id}.")

    print(f"Task paradigm clustering started for dFC ID {dFC_id} ...")
    try:
        run_task_paradigm_clustering(
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
        print(f"Error in task paradigm clustering for dFC ID {dFC_id}: {e}")
        traceback.print_exc()

    print(f"Task paradigm clustering finished for dFC ID {dFC_id}.")
    print(f"Task presence prediction finished for dFC ID {dFC_id}.")

#######################################################################################
