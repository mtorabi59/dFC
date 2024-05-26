import argparse
import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pydfc import DFC, data_loader, task_utils
from pydfc.dfc_utils import dFC_mat2vec, rank_norm

#######################################################################################


def find_available_subjects(dFC_root, task, dFC_id=None):
    """
    Find the subjects that have dFC results for the given task and dFC_id (method).
    """
    SUBJECTS = list()
    ALL_SUBJ_FOLDERS = os.listdir(f"{dFC_root}/")
    ALL_SUBJ_FOLDERS = [folder for folder in ALL_SUBJ_FOLDERS if "sub-" in folder]
    ALL_SUBJ_FOLDERS.sort()
    for subj_folder in ALL_SUBJ_FOLDERS:
        ALL_DFC_FILES = os.listdir(f"{dFC_root}/{subj_folder}/")
        ALL_DFC_FILES = [dFC_file for dFC_file in ALL_DFC_FILES if task in dFC_file]
        if dFC_id is not None:
            ALL_DFC_FILES = [
                dFC_file for dFC_file in ALL_DFC_FILES if f"_{dFC_id}.npy" in dFC_file
            ]
        ALL_DFC_FILES.sort()
        if len(ALL_DFC_FILES) > 0:
            SUBJECTS.append(subj_folder)
    return SUBJECTS


def extract_task_features(TASKS, roi_root, output_root):
    """
    Extract task features from the event data."""
    task_features = {
        "task": list(),
        "relative_task_on": list(),
        "avg_task_duration": list(),
        "var_task_duration": list(),
        "avg_rest_duration": list(),
        "var_rest_duration": list(),
        "num_of_transitions": list(),
        "relative_transition_freq": list(),
    }
    for task_id, task in enumerate(TASKS):

        if task == "task-restingstate":
            continue

        SUBJECTS = find_available_subjects(dFC_root=dFC_root, task=task)

        for subj in SUBJECTS:
            # event data
            task_data = np.load(
                f"{roi_root}/{subj}/{subj}_{task}_task-data.npy", allow_pickle="TRUE"
            ).item()
            Fs_task = task_data["Fs_task"]
            TR_task = 1 / Fs_task

            task_presence = task_utils.extract_task_presence(
                event_labels=task_data["event_labels"],
                TR_task=TR_task,
                TR_mri=task_data["TR_mri"],
                binary=True,
            )

            relative_task_on = task_utils.relative_task_on(task_presence)
            # task duration
            avg_task_duration, var_task_duration = task_utils.task_duration(
                task_presence, task_data["TR_mri"]
            )
            # rest duration
            avg_rest_duration, var_rest_duration = task_utils.rest_duration(
                task_presence, task_data["TR_mri"]
            )
            # freq of transitions
            num_of_transitions, relative_transition_freq = task_utils.transition_freq(
                task_presence
            )

            task_features["task"].append(task)
            task_features["relative_task_on"].append(relative_task_on)
            task_features["avg_task_duration"].append(avg_task_duration)
            task_features["var_task_duration"].append(var_task_duration)
            task_features["avg_rest_duration"].append(avg_rest_duration)
            task_features["var_rest_duration"].append(var_rest_duration)
            task_features["num_of_transitions"].append(num_of_transitions)
            task_features["relative_transition_freq"].append(relative_transition_freq)

    folder = f"{output_root}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(f"{folder}/task_features.npy", task_features)


def dFC_feature_extraction_subj_lvl(
    dFC,
    task_data,
    dynamic_pred="no",
    normalize_dFC=True,
):
    """
    Extract features and target for task presence classification
    for a single subject.
    dynamic_pred: "no", "past", "past_and_future"
    """
    # dFC features
    dFC_mat = dFC.get_dFC_mat()
    TR_array = dFC.TR_array
    if normalize_dFC:
        dFC_mat = rank_norm(dFC_mat)
    dFC_vecs = dFC_mat2vec(dFC_mat)

    # event data
    task_presence = task_utils.extract_task_presence(
        event_labels=task_data["event_labels"],
        TR_task=1 / task_data["Fs_task"],
        TR_mri=task_data["TR_mri"],
        TR_array=TR_array,
        binary=True,
    )

    features = dFC_vecs
    target = task_presence.ravel()

    if dynamic_pred == "past":
        # concat current TR and two TR before of features to predict the current TR of target
        # ignore the edge case of the first two TRs
        features = np.concatenate(
            (features, np.roll(features, 1, axis=0), np.roll(features, 2, axis=0)), axis=1
        )
        features = features[2:, :]
        target = target[2:]
    elif dynamic_pred == "past_and_future":
        # concat current TR and two TR before and after of features to predict the current TR of target
        # ignore the edge case of the first and last two TRs
        features = np.concatenate(
            (
                features,
                np.roll(features, 1, axis=0),
                np.roll(features, 2, axis=0),
                np.roll(features, -1, axis=0),
                np.roll(features, -2, axis=0),
            ),
            axis=1,
        )
        features = features[2:-2, :]
        target = target[2:-2]

    return features, target


def dFC_feature_extraction(
    task,
    train_subjects,
    test_subjects,
    dFC_id,
    roi_root,
    dFC_root,
    dynamic_pred="no",
    normalize_dFC=True,
):
    """
    Extract features and target for task presence classification
    for all subjects.
    """
    X_train = None
    y_train = None
    subj_label_train = list()
    for subj in train_subjects:
        dFC = np.load(
            f"{dFC_root}/{subj}/dFC_{task}_{dFC_id}.npy", allow_pickle="TRUE"
        ).item()

        task_data = np.load(
            f"{roi_root}/{subj}/{subj}_{task}_task-data.npy", allow_pickle="TRUE"
        ).item()

        X_subj, y_subj = dFC_feature_extraction_subj_lvl(
            dFC=dFC,
            task_data=task_data,
            dynamic_pred=dynamic_pred,
            normalize_dFC=normalize_dFC,
        )

        subj_label_train.extend([subj for i in range(X_subj.shape[0])])
        if X_train is None and y_train is None:
            X_train = X_subj
            y_train = y_subj
        else:
            X_train = np.concatenate((X_train, X_subj), axis=0)
            y_train = np.concatenate((y_train, y_subj), axis=0)

    X_test = None
    y_test = None
    subj_label_test = list()
    for subj in test_subjects:
        dFC = np.load(
            f"{dFC_root}/{subj}/dFC_{task}_{dFC_id}.npy", allow_pickle="TRUE"
        ).item()

        task_data = np.load(
            f"{roi_root}/{subj}/{subj}_{task}_task-data.npy", allow_pickle="TRUE"
        ).item()

        X_subj, y_subj = dFC_feature_extraction_subj_lvl(
            dFC=dFC,
            task_data=task_data,
            dynamic_pred=dynamic_pred,
            normalize_dFC=normalize_dFC,
        )

        subj_label_test.extend([subj for i in range(X_subj.shape[0])])
        if X_test is None and y_test is None:
            X_test = X_subj
            y_test = y_subj
        else:
            X_test = np.concatenate((X_test, X_subj), axis=0)
            y_test = np.concatenate((y_test, y_subj), axis=0)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    subj_label_train = np.array(subj_label_train)
    subj_label_test = np.array(subj_label_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        subj_label_train,
        subj_label_test,
        dFC.measure.measure_name,
    )


def task_presence_classification(
    task,
    dFC_id,
    roi_root,
    dFC_root,
    dynamic_pred="no",
    normalize_dFC=True,
    train_test_ratio=0.8,
    explained_var_threshold=0.95,
):
    print(f"=============== {task} ===============")

    if task == "task-restingstate":
        return

    SUBJECTS = find_available_subjects(dFC_root=dFC_root, task=task, dFC_id=dFC_id)

    # randomly select train_test_ratio of the subjects for training
    # and rest for testing using numpy.random.choice
    train_subjects = np.random.choice(
        SUBJECTS, int(train_test_ratio * len(SUBJECTS)), replace=False
    )
    test_subjects = np.setdiff1d(SUBJECTS, train_subjects)
    print(
        f"Number of train subjects: {len(train_subjects)} and test subjects: {len(test_subjects)}"
    )

    X_train, X_test, y_train, y_test, subj_label_train, subj_label_test, measure_name = (
        dFC_feature_extraction(
            task=task,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
            dFC_id=dFC_id,
            roi_root=roi_root,
            dFC_root=dFC_root,
            dynamic_pred=dynamic_pred,
            normalize_dFC=normalize_dFC,
        )
    )

    # task presence classification

    print("task presence classification ...")

    # find num_PCs
    pca = PCA(svd_solver="full", whiten=False)
    pca.fit(X_train)
    num_PCs = (
        np.where(np.cumsum(pca.explained_variance_ratio_) > explained_var_threshold)[0][0]
        + 1
    )

    # create a pipeline with a knn model to find the best n_neighbors
    knn = make_pipeline(
        StandardScaler(),
        PCA(n_components=num_PCs),
        KNeighborsClassifier(),
    )
    # create a dictionary of all values we want to test for n_neighbors
    param_grid = {"kneighborsclassifier__n_neighbors": np.arange(1, 30)}
    # use gridsearch to test all values for n_neighbors
    knn_gscv = GridSearchCV(knn, param_grid, cv=5)
    # fit model to data
    knn_gscv.fit(X_train, y_train)

    n_neighbors = knn_gscv.best_params_["kneighborsclassifier__n_neighbors"]

    neigh = make_pipeline(
        StandardScaler(),
        PCA(n_components=num_PCs),
        KNeighborsClassifier(n_neighbors=n_neighbors),
    ).fit(X_train, y_train)

    ML_RESULT = {
        "pca": pca,
        "num_PCs": num_PCs,
        "cv_results": knn_gscv.cv_results_,
        "KNN": neigh,
        "KNN train score": neigh.score(X_train, y_train),
        "KNN test score": neigh.score(X_test, y_test),
    }

    print(f"KNN train score {measure_name} {task}: {neigh.score(X_train, y_train)}")
    print(f"KNN test score {measure_name} {task}: {neigh.score(X_test, y_test)}")

    # measure pred score on each subj

    ML_scores = {
        "subj_id": list(),
        "group": list(),
        "task": list(),
        "dFC method": list(),
        "KNN accuracy": list(),
    }
    for subj in SUBJECTS:
        ML_scores["subj_id"].append(subj)
        if subj in train_subjects:
            ML_scores["group"].append("train")
            features = X_train[subj_label_train == subj, :]
            target = y_train[subj_label_train == subj]
        elif subj in test_subjects:
            ML_scores["group"].append("test")
            features = X_test[subj_label_test == subj, :]
            target = y_test[subj_label_test == subj]

        pred = neigh.predict(features)

        ML_scores["KNN accuracy"].append(balanced_accuracy_score(target, pred))

        ML_scores["task"].append(task)
        ML_scores["dFC method"].append(measure_name)

    return ML_RESULT, ML_scores


def run_classification(
    TASKS,
    roi_root,
    dFC_root,
    output_root,
    dynamic_pred="no",
    normalize_dFC=True,
):
    ML_scores = {
        "subj_id": list(),
        "group": list(),
        "task": list(),
        "dFC method": list(),
        "KNN accuracy": list(),
    }
    for dFC_id in range(0, 7):
        print(f"=================== dFC {dFC_id} ===================")

        ML_RESULT = {}
        for task_id, task in enumerate(TASKS):
            ML_RESULT_new, ML_scores_new = task_presence_classification(
                task=task,
                dFC_id=dFC_id,
                roi_root=roi_root,
                dFC_root=dFC_root,
                dynamic_pred=dynamic_pred,
                normalize_dFC=normalize_dFC,
            )
            ML_RESULT[task] = ML_RESULT_new
            for key in ML_scores:
                ML_scores[key].extend(ML_scores_new[key])

        folder = f"{output_root}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(f"{folder}/ML_RESULT_{dFC_id}.npy", ML_RESULT)

    np.save(f"{folder}/ML_scores_classify.npy", ML_scores)


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

    # Read global configs
    with open(dataset_info_file, "r") as f:
        dataset_info = json.load(f)

    print("Task presence prediction started ...")

    TASKS = dataset_info["TASKS"]

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
        roi_root=roi_root,
        output_root=ML_root,
    )
    run_classification(
        TASKS=TASKS,
        roi_root=roi_root,
        dFC_root=dFC_root,
        output_root=ML_root,
        dynamic_pred="no",
        normalize_dFC=True,
    )

    print("Task presence prediction CODE finished running.")

#######################################################################################
