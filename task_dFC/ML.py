import argparse
import json
import os
import traceback

import numpy as np
from scipy.spatial import procrustes
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, silhouette_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pydfc import DFC, data_loader, task_utils
from pydfc.dfc_utils import dFC_mat2vec, dFC_vec2mat, rank_norm

#######################################################################################


def find_available_subjects(dFC_root, task, run=None, session=None, dFC_id=None):
    """
    Find the subjects that have dFC results for the given task and dFC_id (method).
    """
    SUBJECTS = list()
    ALL_SUBJ_FOLDERS = os.listdir(f"{dFC_root}/")
    ALL_SUBJ_FOLDERS = [folder for folder in ALL_SUBJ_FOLDERS if "sub-" in folder]
    ALL_SUBJ_FOLDERS.sort()
    for subj_folder in ALL_SUBJ_FOLDERS:
        if session is None:
            ALL_DFC_FILES = os.listdir(f"{dFC_root}/{subj_folder}/")
        else:
            ALL_DFC_FILES = os.listdir(f"{dFC_root}/{subj_folder}/{session}/")
        ALL_DFC_FILES = [
            dFC_file for dFC_file in ALL_DFC_FILES if f"_{task}_" in dFC_file
        ]
        if dFC_id is not None:
            ALL_DFC_FILES = [
                dFC_file for dFC_file in ALL_DFC_FILES if f"_{dFC_id}.npy" in dFC_file
            ]
        if run is not None:
            ALL_DFC_FILES = [
                dFC_file for dFC_file in ALL_DFC_FILES if f"_{run}_" in dFC_file
            ]
        if session is not None:
            ALL_DFC_FILES = [
                dFC_file for dFC_file in ALL_DFC_FILES if f"_{session}_" in dFC_file
            ]
        ALL_DFC_FILES.sort()
        if len(ALL_DFC_FILES) > 0:
            SUBJECTS.append(subj_folder)
    return SUBJECTS


def extract_task_features(TASKS, RUNS, SESSIONS, roi_root, output_root):
    """
    Extract task features from the event data."""
    for session in SESSIONS:
        task_features = {
            "task": list(),
            "run": list(),
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

            for run in RUNS[task]:

                SUBJECTS = find_available_subjects(
                    dFC_root=dFC_root, task=task, run=run, session=session
                )

                for subj in SUBJECTS:
                    # event data
                    task_data = load_task_data(
                        roi_root=roi_root, subj=subj, task=task, run=run, session=session
                    )
                    Fs_task = task_data["Fs_task"]
                    TR_task = 1 / Fs_task

                    task_presence = task_utils.extract_task_presence(
                        event_labels=task_data["event_labels"],
                        TR_task=TR_task,
                        TR_mri=task_data["TR_mri"],
                        binary=True,
                        binarizing_method="mean",
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
                    num_of_transitions, relative_transition_freq = (
                        task_utils.transition_freq(task_presence)
                    )

                    task_features["task"].append(task)
                    task_features["run"].append(run)
                    task_features["relative_task_on"].append(relative_task_on)
                    task_features["avg_task_duration"].append(avg_task_duration)
                    task_features["var_task_duration"].append(var_task_duration)
                    task_features["avg_rest_duration"].append(avg_rest_duration)
                    task_features["var_rest_duration"].append(var_rest_duration)
                    task_features["num_of_transitions"].append(num_of_transitions)
                    task_features["relative_transition_freq"].append(
                        relative_transition_freq
                    )

        if session is None:
            folder = f"{output_root}"
        else:
            folder = f"{output_root}/{session}"
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
        binarizing_method="mean",
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


def load_dFC(dFC_root, subj, task, dFC_id, run=None, session=None):
    """
    Load the dFC results for a given subject, task, dFC_id, run and session.
    """
    if session is None:
        if run is None:
            dFC = np.load(
                f"{dFC_root}/{subj}/dFC_{task}_{dFC_id}.npy", allow_pickle="TRUE"
            ).item()
        else:
            dFC = np.load(
                f"{dFC_root}/{subj}/dFC_{task}_{run}_{dFC_id}.npy", allow_pickle="TRUE"
            ).item()
    else:
        if run is None:
            dFC = np.load(
                f"{dFC_root}/{subj}/{session}/dFC_{session}_{task}_{dFC_id}.npy",
                allow_pickle="TRUE",
            ).item()
        else:
            dFC = np.load(
                f"{dFC_root}/{subj}/{session}/dFC_{session}_{task}_{run}_{dFC_id}.npy",
                allow_pickle="TRUE",
            ).item()

    return dFC


def load_task_data(roi_root, subj, task, run=None, session=None):
    """
    Load the task data for a given subject, task and run.
    """
    if session is None:
        if run is None:
            task_data = np.load(
                f"{roi_root}/{subj}/{subj}_{task}_task-data.npy", allow_pickle="TRUE"
            ).item()
        else:
            task_data = np.load(
                f"{roi_root}/{subj}/{subj}_{task}_{run}_task-data.npy",
                allow_pickle="TRUE",
            ).item()
    else:
        if run is None:
            task_data = np.load(
                f"{roi_root}/{subj}/{session}/{subj}_{session}_{task}_task-data.npy",
                allow_pickle="TRUE",
            ).item()
        else:
            task_data = np.load(
                f"{roi_root}/{subj}/{session}/{subj}_{session}_{task}_{run}_task-data.npy",
                allow_pickle="TRUE",
            ).item()

    return task_data


def precheck_for_procruste(X_best, X_subj):
    """
    Check if the two matrices have the same number of rows. if not, make them the same.
    """
    # for the procrustes transformation, the number of samples should be the same
    if X_subj.shape[0] > X_best.shape[0]:
        # add zero rows to the embedding of the best subject
        X_best_new = np.concatenate(
            (
                X_best,
                np.zeros(
                    (
                        X_subj.shape[0] - X_best.shape[0],
                        X_best.shape[1],
                    )
                ),
            ),
            axis=0,
        )
    elif X_subj.shape[0] < X_best.shape[0]:
        # remove extra rows from the embedding of the best subject
        X_best_new = X_best[: X_subj.shape[0], :]
    else:
        X_best_new = X_best

    X_best_new = X_best_new.copy()

    return X_best_new


def embed_dFC_features(
    train_subjects,
    test_subjects,
    X_train,
    X_test,
    y_train,
    y_test,
    subj_label_train,
    subj_label_test,
    embedding="PCA",
    n_components=30,
    n_neighbors_LE=125,
    LE_embedding_method="concat+embed",
):
    """
    Embed the dFC features into a lower dimensional space using PCA or LE. For LE, it assumes that the samples of the same subject are contiguous.

    for LE, first the LE is applied on each subj separately and then the procrustes transformation is applied to align the embeddings of different subjects.
    All the subjects are transformed into the space of the subject with the highest silhouette score.

    LE_embedding_method: "concat+embed" or "embed+procrustes"
    """
    if embedding == "PCA":
        pca = PCA(n_components=n_components, svd_solver="full", whiten=False)
        pca.fit(X_train)
        X_train_embed = pca.transform(X_train)
        if X_test is not None:
            X_test_embed = pca.transform(X_test)
        else:
            X_test_embed = None
    elif embedding == "LE":
        if LE_embedding_method == "embed+procrustes":
            # first embed the dFC features of each subject into a lower dimensional space using LE separately
            embed_dict = {}
            for subject in train_subjects:
                # assert the samples of the same subject are contiguous
                assert np.all(
                    np.diff(np.where(subj_label_train == subject)[0]) == 1
                ), f"Indices of {subject} are not consecutive"
                X_subj = X_train[subj_label_train == subject, :]
                y_subj = y_train[subj_label_train == subject]
                LE = SpectralEmbedding(
                    n_components=n_components,
                    n_neighbors=min(n_neighbors_LE, X_subj.shape[0]),
                )
                X_subj_embed = LE.fit_transform(X_subj)
                SI = silhouette_score(X_subj_embed, y_subj)
                embed_dict[subject] = {"X_subj_embed": X_subj_embed, "SI": SI}

            # find the best transformation based on the SI score
            best_SI = -1
            best_subject = None
            for subject in embed_dict:
                if embed_dict[subject]["SI"] > best_SI:
                    best_SI = embed_dict[subject]["SI"]
                    best_subject = subject

            # apply procrustes transformation to align the embeddings of different subjects
            # use the embeddings of the subject with the highest SI score as the reference
            X_train_embed = None
            for subject in train_subjects:
                X_subj_embed = embed_dict[subject]["X_subj_embed"]
                # procrustes transformation
                if subject == best_subject:
                    X_subj_embed_transformed = X_subj_embed
                else:
                    # for the procrustes transformation, the number of samples should be the same
                    X_best_subj_embed = precheck_for_procruste(
                        embed_dict[best_subject]["X_subj_embed"], X_subj_embed
                    )
                    _, X_subj_embed_transformed, _ = procrustes(
                        X_best_subj_embed, X_subj_embed
                    )
                if X_train_embed is None:
                    X_train_embed = X_subj_embed_transformed
                else:
                    X_train_embed = np.concatenate(
                        (X_train_embed, X_subj_embed_transformed), axis=0
                    )

            # apply the same transformation to the test set
            X_test_embed = None
            for subject in test_subjects:
                # assert the samples of the same subject are contiguous
                assert np.all(
                    np.diff(np.where(subj_label_test == subject)[0]) == 1
                ), f"Indices of {subject} are not consecutive"
                X_subj = X_test[subj_label_test == subject, :]
                LE = SpectralEmbedding(
                    n_components=n_components,
                    n_neighbors=min(n_neighbors_LE, X_subj.shape[0]),
                )
                X_subj_embed = LE.fit_transform(X_subj)
                # procrustes transformation
                # for the procrustes transformation, the number of samples should be the same
                X_best_subj_embed = precheck_for_procruste(
                    embed_dict[best_subject]["X_subj_embed"], X_subj_embed
                )
                _, X_subj_embed_transformed, _ = procrustes(
                    X_best_subj_embed, X_subj_embed
                )
                if X_test_embed is None:
                    X_test_embed = X_subj_embed_transformed
                else:
                    X_test_embed = np.concatenate(
                        (X_test_embed, X_subj_embed_transformed), axis=0
                    )
        elif LE_embedding_method == "concat+embed":
            LE = SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors_LE)
            X_train_embed = LE.fit_transform(X_train)
            if X_test is not None:
                X_test_embed = LE.transform(X_test)
            else:
                X_test_embed = None

    return X_train_embed, X_test_embed


def dFC_feature_extraction(
    task,
    train_subjects,
    test_subjects,
    dFC_id,
    roi_root,
    dFC_root,
    run=None,
    session=None,
    dynamic_pred="no",
    normalize_dFC=True,
):
    """
    Extract features and target for task presence classification
    for all subjects.
    if run is specified, dFC results for that run will be used.
    """
    dFC_measure_name = None
    X_train = None
    y_train = None
    subj_label_train = list()
    for subj in train_subjects:

        dFC = load_dFC(
            dFC_root=dFC_root,
            subj=subj,
            task=task,
            dFC_id=dFC_id,
            run=run,
            session=session,
        )
        task_data = load_task_data(
            roi_root=roi_root, subj=subj, task=task, run=run, session=session
        )

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

        if dFC_measure_name is None:
            dFC_measure_name = dFC.measure.measure_name
        else:
            assert (
                dFC_measure_name == dFC.measure.measure_name
            ), "dFC measure is not consistent."

    X_test = None
    y_test = None
    subj_label_test = list()
    for subj in test_subjects:
        dFC = load_dFC(
            dFC_root=dFC_root,
            subj=subj,
            task=task,
            dFC_id=dFC_id,
            run=run,
            session=session,
        )
        task_data = load_task_data(
            roi_root=roi_root, subj=subj, task=task, run=run, session=session
        )

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

        if dFC_measure_name is None:
            dFC_measure_name = dFC.measure.measure_name
        else:
            assert (
                dFC_measure_name == dFC.measure.measure_name
            ), "dFC measure is not consistent."

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    subj_label_train = np.array(subj_label_train)
    subj_label_test = np.array(subj_label_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        subj_label_train,
        subj_label_test,
        dFC_measure_name,
    )


def logistic_regression_classify(X_train, y_train, X_test, y_test):
    """
    Logistic regression classification
    """
    # create a pipeline with a logistic regression model to find the best C
    logistic_reg = make_pipeline(
        StandardScaler(), LogisticRegression(penalty="l1", solver="saga")
    )
    # create a dictionary of all values we want to test for C
    param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    # use gridsearch to test all values for C
    lr_gscv = GridSearchCV(logistic_reg, param_grid, cv=5)
    # fit model to data
    lr_gscv.fit(X_train, y_train)

    C = lr_gscv.best_params_["logisticregression__C"]

    log_reg = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l1", C=C, solver="saga"),
    ).fit(X_train, y_train)

    RESULT = {
        "log_reg_model": log_reg,
        "log_reg_C": C,
        "log_reg_train_score": log_reg.score(X_train, y_train),
        "log_reg_test_score": log_reg.score(X_test, y_test),
    }

    return RESULT


def KNN_classify(X_train, y_train, X_test, y_test):
    """
    KNN classification
    """
    # create a pipeline with a knn model to find the best n_neighbors
    knn = make_pipeline(
        StandardScaler(),
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
        KNeighborsClassifier(n_neighbors=n_neighbors),
    ).fit(X_train, y_train)

    RESULT = {
        "KNN_cv_results": knn_gscv.cv_results_,
        "KNN_model": neigh,
        "KNN_train_score": neigh.score(X_train, y_train),
        "KNN_test_score": neigh.score(X_test, y_test),
    }

    return RESULT


def random_forest_classify(X_train, y_train, X_test, y_test):
    """
    Random Forest classification
    """
    # create a pipeline with a random forest model to find the best n_estimators
    rf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(),
    )
    # create a dictionary of all values we want to test for n_estimators
    param_grid = {
        "randomforestclassifier__n_estimators": [10, 50, 100, 200],
        "randomforestclassifier__max_depth": [None, 5, 10, 20, 30],
    }
    # use gridsearch to test all values for n_estimators
    rf_gscv = GridSearchCV(rf, param_grid, cv=5)
    # fit model to data
    rf_gscv.fit(X_train, y_train)

    n_estimators = rf_gscv.best_params_["randomforestclassifier__n_estimators"]
    max_depth = rf_gscv.best_params_["randomforestclassifier__max_depth"]

    rf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth),
    ).fit(X_train, y_train)

    RESULT = {
        "RF_cv_results": rf_gscv.cv_results_,
        "RF_model": rf,
        "RF_train_score": rf.score(X_train, y_train),
        "RF_test_score": rf.score(X_test, y_test),
    }

    return RESULT


def gradient_boosting_classify(X_train, y_train, X_test, y_test):
    """
    Gradient Boosting classification
    """
    # create a pipeline with a gradient boosting model to find the best n_estimators
    gb = make_pipeline(
        StandardScaler(),
        GradientBoostingClassifier(),
    )
    # create a dictionary of all values we want to test for n_estimators
    param_grid = {
        "gradientboostingclassifier__n_estimators": [10, 50, 100, 200],
        "gradientboostingclassifier__learning_rate": [0.01, 0.1, 0.2],
        "gradientboostingclassifier__max_depth": [3, 5, 10],
    }
    # use gridsearch to test all values for n_estimators
    gb_gscv = GridSearchCV(gb, param_grid, cv=5)
    # fit model to data
    gb_gscv.fit(X_train, y_train)

    n_estimators = gb_gscv.best_params_["gradientboostingclassifier__n_estimators"]
    learning_rate = gb_gscv.best_params_["gradientboostingclassifier__learning_rate"]
    max_depth = gb_gscv.best_params_["gradientboostingclassifier__max_depth"]

    gb = make_pipeline(
        StandardScaler(),
        GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate
        ),
    ).fit(X_train, y_train)

    RESULT = {
        "GB_cv_results": gb_gscv.cv_results_,
        "GB_model": gb,
        "GB_train_score": gb.score(X_train, y_train),
        "GB_test_score": gb.score(X_test, y_test),
    }

    return RESULT


def task_presence_classification(
    task,
    dFC_id,
    roi_root,
    dFC_root,
    run=None,
    session=None,
    dynamic_pred="no",
    normalize_dFC=True,
    train_test_ratio=0.8,
):
    """
    perform task presence classification using logistic regression, KNN, Random Forest, Gradient Boosting
    for a given task and dFC method and run.
    """
    if run is None:
        print(f"=============== {task} ===============")
    else:
        print(f"=============== {task} {run} ===============")

    if task == "task-restingstate":
        return

    SUBJECTS = find_available_subjects(
        dFC_root=dFC_root, task=task, run=run, session=session, dFC_id=dFC_id
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

    X_train, X_test, y_train, y_test, subj_label_train, subj_label_test, measure_name = (
        dFC_feature_extraction(
            task=task,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
            dFC_id=dFC_id,
            roi_root=roi_root,
            dFC_root=dFC_root,
            run=run,
            session=session,
            dynamic_pred=dynamic_pred,
            normalize_dFC=normalize_dFC,
        )
    )

    # embed dFC features
    X_train, X_test = embed_dFC_features(
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        subj_label_train=subj_label_train,
        subj_label_test=subj_label_test,
        embedding="LE",
        n_components=30,
        n_neighbors_LE=125,
        LE_embedding_method="concat+embed",
    )

    # task presence classification

    print("task presence classification ...")

    # logistic regression
    log_reg_RESULT = logistic_regression_classify(X_train, y_train, X_test, y_test)

    # KNN
    KNN_RESULT = KNN_classify(X_train, y_train, X_test, y_test)

    # # Random Forest
    # RF_RESULT = random_forest_classify(
    #     X_train, y_train, X_test, y_test
    # )

    # # Gradient Boosting
    # GBT_RESULT = gradient_boosting_classify(
    #     X_train, y_train, X_test, y_test
    # )

    ML_RESULT = {}
    for key in log_reg_RESULT:
        ML_RESULT[key] = log_reg_RESULT[key]
    for key in KNN_RESULT:
        ML_RESULT[key] = KNN_RESULT[key]
    # for key in RF_RESULT:
    #     ML_RESULT[key] = RF_RESULT[key]
    # for key in GBT_RESULT:
    #     ML_RESULT[key] = GBT_RESULT[key]

    # measure pred score on each subj

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
    log_reg = log_reg_RESULT["log_reg_model"]
    KNN = KNN_RESULT["KNN_model"]
    # RF = RF_RESULT["RF_model"]
    # GBT = GBT_RESULT["GB_model"]

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

        pred_lr = log_reg.predict(features)
        pred_KNN = KNN.predict(features)
        # pred_RF = RF.predict(features)
        # pred_GBT = GBT.predict(features)

        ML_scores["Logistic regression accuracy"].append(
            balanced_accuracy_score(target, pred_lr)
        )
        ML_scores["KNN accuracy"].append(balanced_accuracy_score(target, pred_KNN))
        # ML_scores["Random Forest accuracy"].append(
        #     balanced_accuracy_score(target, pred_RF)
        # )
        # ML_scores["Gradient Boosting accuracy"].append(
        #     balanced_accuracy_score(target, pred_GBT)
        # )

        ML_scores["task"].append(task)
        ML_scores["run"].append(run)
        ML_scores["dFC method"].append(measure_name)

    return ML_RESULT, ML_scores


def task_presence_clustering(
    task,
    dFC_id,
    roi_root,
    dFC_root,
    run=None,
    session=None,
    normalize_dFC=True,
):
    if run is None:
        print(f"=============== {task} ===============")
    else:
        print(f"=============== {task} {run} ===============")

    if task == "task-restingstate":
        return

    SUBJECTS = find_available_subjects(
        dFC_root=dFC_root, task=task, run=run, session=session, dFC_id=dFC_id
    )

    print(f"Number of subjects: {len(SUBJECTS)}")

    X, _, y, _, subj_label, _, measure_name = dFC_feature_extraction(
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
        LE_embedding_method="concat+embed",
    )

    # clustering
    # apply kmeans clustering to dFC features

    n_clusters = 2  # corresponding to task and rest

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

    clustering_RESULTS = {
        "StandardScaler": scaler,
        "kmeans": kmeans,
        "ARI": adjusted_rand_score(y, labels_pred),
        # "centroids": centroids_mat,
    }

    clustering_scores = {
        "subj_id": list(),
        "task": list(),
        "run": list(),
        "dFC method": list(),
        "Kmeans ARI": list(),
        "SI": list(),
    }
    for subj in SUBJECTS:
        clustering_scores["subj_id"].append(subj)
        features = X[subj_label == subj, :]
        target = y[subj_label == subj]

        features_normalized = scaler.transform(features)
        pred_kmeans = kmeans.predict(features_normalized)

        clustering_scores["Kmeans ARI"].append(adjusted_rand_score(target, pred_kmeans))

        # silhouette score in terms of separability of original labels, not the clustering labels
        clustering_scores["SI"].append(silhouette_score(features, target))

        clustering_scores["task"].append(task)
        clustering_scores["run"].append(run)
        clustering_scores["dFC method"].append(measure_name)

    return clustering_RESULTS, clustering_scores


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


def task_paradigm_clustering(
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
            LE_embedding_method="concat+embed",
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
        task_paradigm_clustering(
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
