# -*- coding: utf-8 -*-
"""
Functions to facilitate applying ML algorithms to dFC.

Created on Aug 8 2024
@author: Mohammad Torabi
"""
import os
import warnings

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, kneighbors_graph
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dfc_utils import dFC_mat2vec, dFC_vec2mat, rank_norm
from .task_utils import (
    calc_relative_task_on,
    calc_rest_duration,
    calc_task_duration,
    calc_transition_freq,
    extract_task_presence,
)

################################# Feature Loading Functions ####################################


def find_available_subjects(dFC_root, task, run=None, session=None, dFC_id=None):
    """
    Find the subjects that have dFC results for the given task and dFC_id (method).

    If run and session are specified, the dFC results for that run and session will be used.
    Otherwise, the subjects that have dFC results at least for one run and session will returned.
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


################################# Feature Extraction Functions ####################################


def extract_task_features(TASKS, RUNS, session, roi_root, dFC_root, no_hrf=False):
    """
    Extract task features from the event data.

    if no_hrf is True, the task presence will be binarized without convolving with HRF.
    Therefore the task features will be extracted based on the event labels and
    without the effect of HRF.
    """
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

                task_presence = extract_task_presence(
                    event_labels=task_data["event_labels"],
                    TR_task=TR_task,
                    TR_mri=task_data["TR_mri"],
                    binary=True,
                    binarizing_method="shift",
                    no_hrf=no_hrf,
                )

                relative_task_on = calc_relative_task_on(task_presence)
                # task duration
                avg_task_duration, var_task_duration = calc_task_duration(
                    task_presence, task_data["TR_mri"]
                )
                # rest duration
                avg_rest_duration, var_rest_duration = calc_rest_duration(
                    task_presence, task_data["TR_mri"]
                )
                # freq of transitions
                num_of_transitions, relative_transition_freq = calc_transition_freq(
                    task_presence
                )

                task_features["task"].append(task)
                task_features["run"].append(run)
                task_features["relative_task_on"].append(relative_task_on)
                task_features["avg_task_duration"].append(avg_task_duration)
                task_features["var_task_duration"].append(var_task_duration)
                task_features["avg_rest_duration"].append(avg_rest_duration)
                task_features["var_rest_duration"].append(var_rest_duration)
                task_features["num_of_transitions"].append(num_of_transitions)
                task_features["relative_transition_freq"].append(relative_transition_freq)

    return task_features


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
    task_presence = extract_task_presence(
        event_labels=task_data["event_labels"],
        TR_task=1 / task_data["Fs_task"],
        TR_mri=task_data["TR_mri"],
        TR_array=TR_array,
        binary=True,
        binarizing_method="shift",
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


################################# Feature Embedding Functions ####################################


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


def generalized_procrustes(X_embed_dict):
    """
    Generalized Procrustes Analysis

    X_embed_dict: dict
        dict of scans and their embeddings

    returns the mean X across scans to be used as the reference for procrustes transformation
    """
    # initial step
    # not all scans have the same number of samples
    # find the max number of samples among all scans
    max_samples = 0
    for scan in X_embed_dict:
        if X_embed_dict[scan].shape[0] > max_samples:
            max_samples = X_embed_dict[scan].shape[0]

    # find the mean embedding of all scan to use as the reference for procrustes transformation
    X_list = []
    for scan in X_embed_dict:
        X_scan_embed = X_embed_dict[scan]
        # add zero rows to the embedding of the scan with less samples
        if X_scan_embed.shape[0] < max_samples:
            X_scan_embed_new = np.concatenate(
                (
                    X_scan_embed,
                    np.zeros(
                        (
                            max_samples - X_scan_embed.shape[0],
                            X_scan_embed.shape[1],
                        )
                    ),
                ),
                axis=0,
            )
        else:
            X_scan_embed_new = X_scan_embed
        X_list.append(X_scan_embed_new)

    # now iteratively find the mean X for transform
    for iter_num in range(100):

        try:
            # initialize Procrustes distance
            current_distance = 0

            num_X = len(X_list)

            # initialize a mean X by randomly selecting
            # one of the Xs using np.random.choice
            mean_X = X_list[np.random.choice(num_X)]

            # create array for new Xs, add
            new_Xs = np.zeros(np.array(X_list).shape)

            counter = 0
            flag = False
            while True:
                counter += 1
                if counter > 1e6:
                    # if the algorithm does not converge, break the cycle
                    # to avoid infinite loop
                    flag = True
                    break

                # add the mean X as first element of array
                new_Xs[0] = mean_X

                # superimpose all shapes to current mean
                for i in range(1, num_X):
                    _, new_X, _ = procrustes(mean_X, X_list[i])
                    new_Xs[i] = new_X

                # calculate new mean
                new_mean = np.mean(new_Xs, axis=0)

                _, _, new_distance = procrustes(new_mean, mean_X)

                # if the distance did not change, break the cycle
                if np.abs(new_distance - current_distance) < 1e-6:
                    break

                # align the new_mean to old mean
                _, new_mean, _ = procrustes(mean_X, new_mean)

                # update mean and distance
                mean_X = new_mean
                current_distance = new_distance

            if not flag:
                return mean_X
        except:
            continue

    raise RuntimeError("Generalized Procrustes Analysis did not converge.")


def twonn(X, discard_ratio=0.1):
    """
    Calculates intrinsic dimension of the provided data points with the TWO-NN algorithm.

    -----------
    Parameters:

    X : 2d array-like
        (n_samples, n_features)
    discard_fraction : float between 0 and 1
        Fraction of largest distances to discard (heuristic from the paper)

    Returns:

    d : float
        Intrinsic dimension of the dataset according to TWO-NN.
    """

    num_samples = X.shape[0]

    NN = NearestNeighbors(n_neighbors=30)
    NN.fit(X)
    distances, _ = NN.kneighbors(return_distance=True)

    mu = np.zeros((num_samples))
    for i in range(num_samples):
        # find the two nearest neighbors that have
        # different distances and the distance is not 0
        r1, r2 = None, None
        for j in range(distances.shape[1]):
            if distances[i, j] != 0:
                if r1 is None:
                    r1 = distances[i, j]
                elif distances[i, j] != r1:
                    r2 = distances[i, j]
                    break
        if r1 is not None and r2 is not None:
            mu[i] = r2 / r1
        else:
            mu[i] = np.nan

    # discard NaN values
    mu = mu[~np.isnan(mu)]
    # large distances will cause the estimation to be biased, discard them
    mu = mu[np.argsort(mu)[: int((1 - discard_ratio) * num_samples)]]

    # CDF
    CDF = np.arange(1, 1 + len(mu)) / num_samples
    # Fit the formula: log(1 - CDF) = d * log(mu)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(mu).reshape(-1, 1), -np.log(1 - CDF).reshape(-1, 1))
    d = lr.coef_[0][0]

    return d


def SI_ID(X, y, search_range=range(2, 50, 5), n_neighbors_LE=125):
    """
    Find the intrinsic dimension of the data based on the silhouette score.
    """

    SI_score = {}
    for n_components in search_range:
        try:
            X_train_embed, _ = embed_dFC_features(
                train_subjects=["subj"],
                test_subjects=[],
                X_train=X,
                X_test=None,
                y_train=y,
                y_test=None,
                subj_label_train=np.array(["subj"] * len(y)),
                subj_label_test=None,
                embedding="LE",
                n_components=n_components,
                n_neighbors_LE=n_neighbors_LE,
                LE_embedding_method="embed+procrustes",
            )
        except:
            continue

        SI_score[n_components] = silhouette_score(X_train_embed, y)

    # find the intrinsic dimension based on the silhouette score
    intrinsic_dim = max(SI_score, key=SI_score.get)

    return intrinsic_dim


def find_intrinsic_dim(
    X,
    y,
    subj_label,
    subjects,
    method="SI",
    n_neighbors_LE=125,
    search_range_SI=range(2, 50, 5),
):
    """
    Find the number of components to use for embedding the data using LE.
    Find the average intrinsic dimension across all subjects.

    method: "SI" or "twonn"

    Returns:
    intrinsic_dim: number of components to use for embedding
    """
    if method == "SI":
        intrinsic_dim_all = list()
        for subject in subjects:
            X_subj = X[subj_label == subject, :]
            y_subj = y[subj_label == subject]
            intrinsic_dim_all.append(
                SI_ID(
                    X_subj,
                    y_subj,
                    search_range=search_range_SI,
                    n_neighbors_LE=n_neighbors_LE,
                )
            )
        intrinsic_dim = int(np.mean(intrinsic_dim_all))
    elif method == "twonn":
        intrinsic_dim_all = list()
        for subject in subjects:
            X_subj = X[subj_label == subject, :]
            intrinsic_dim_all.append(twonn(X_subj, discard_ratio=0.1))
        intrinsic_dim = int(np.mean(intrinsic_dim_all))
    return intrinsic_dim


def LE_transform(X, n_components, n_neighbors, distance_metric="euclidean"):
    """
    Apply Laplacian Eigenmaps (LE) to transform data into a lower dimensional space.

    if n_neighbors >= n_samples, n_neighbors will be changed to the lower limit n_neighbors
    """
    min_n_neighbors = 70

    if n_neighbors >= X.shape[0]:
        n_neighbors_to_be_used = min_n_neighbors
        # raise a warning
        warnings.warn(
            "n_neighbors is larger than the number of samples. n_neighbors is set to the minimum value of 70."
        )
    else:
        n_neighbors_to_be_used = n_neighbors

    affinity_matrix = kneighbors_graph(
        X,
        n_neighbors=n_neighbors_to_be_used,
        mode="connectivity",
        include_self=False,
        metric=distance_metric,
    )
    affinity_matrix = affinity_matrix.toarray()
    affinity_matrix = np.divide(affinity_matrix + affinity_matrix.T, 2)
    LE = SpectralEmbedding(
        n_components=n_components,
        affinity="precomputed",
        n_neighbors=n_neighbors_to_be_used,
        eigen_solver="lobpcg",
    )
    X_embed = LE.fit_transform(X=affinity_matrix)
    return X_embed


def LE_embed_procustes(
    X_train,
    X_test,
    y_train,
    y_test,
    subj_label_train,
    subj_label_test,
    train_subjects,
    test_subjects,
    n_components=30,
    n_neighbors_LE=125,
    procruste_method="best_SI",
):
    if procruste_method == "best_SI":
        # first embed the dFC features of each subject into a lower dimensional space using LE separately
        embed_dict = {}
        for subject in train_subjects:
            # assert the samples of the same subject are contiguous
            assert np.all(
                np.diff(np.where(subj_label_train == subject)[0]) == 1
            ), f"Indices of {subject} are not consecutive"
            X_subj = X_train[subj_label_train == subject, :]
            y_subj = y_train[subj_label_train == subject]
            X_subj_embed = LE_transform(
                X=X_subj,
                n_components=n_components,
                n_neighbors=n_neighbors_LE,
                distance_metric="correlation",
            )
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
            X_subj_embed = LE_transform(
                X=X_subj,
                n_components=n_components,
                n_neighbors=n_neighbors_LE,
                distance_metric="correlation",
            )
            # procrustes transformation
            # for the procrustes transformation, the number of samples should be the same
            X_best_subj_embed = precheck_for_procruste(
                embed_dict[best_subject]["X_subj_embed"], X_subj_embed
            )
            _, X_subj_embed_transformed, _ = procrustes(X_best_subj_embed, X_subj_embed)
            if X_test_embed is None:
                X_test_embed = X_subj_embed_transformed
            else:
                X_test_embed = np.concatenate(
                    (X_test_embed, X_subj_embed_transformed), axis=0
                )

    elif procruste_method == "generalized":
        # in this method we use generalized procrustes analysis to align the embeddings of different subjects
        # first embed the dFC features of each subject into a lower dimensional space using LE separately
        embed_dict = {}
        for subject in train_subjects:
            # assert the samples of the same subject are contiguous
            assert np.all(
                np.diff(np.where(subj_label_train == subject)[0]) == 1
            ), f"Indices of {subject} are not consecutive"
            X_subj = X_train[subj_label_train == subject, :]
            X_subj_embed = LE_transform(
                X=X_subj,
                n_components=n_components,
                n_neighbors=n_neighbors_LE,
                distance_metric="correlation",
            )
            embed_dict[subject] = X_subj_embed

        mean_X_train = generalized_procrustes(embed_dict)

        X_train_embed = None
        for subject in train_subjects:
            X_subj_embed = embed_dict[subject]
            mean_X_train_new_size = precheck_for_procruste(mean_X_train, X_subj_embed)
            _, X_subj_embed_transformed, _ = procrustes(
                mean_X_train_new_size, X_subj_embed
            )
            if X_train_embed is None:
                X_train_embed = X_subj_embed_transformed
            else:
                X_train_embed = np.concatenate(
                    (X_train_embed, X_subj_embed_transformed), axis=0
                )

        X_test_embed = None
        for subject in test_subjects:
            X_subj = X_test[subj_label_test == subject, :]
            X_subj_embed = LE_transform(
                X=X_subj,
                n_components=n_components,
                n_neighbors=n_neighbors_LE,
                distance_metric="correlation",
            )
            mean_X_train_new_size = precheck_for_procruste(mean_X_train, X_subj_embed)
            _, X_subj_embed_transformed, _ = procrustes(
                mean_X_train_new_size, X_subj_embed
            )
            if X_test_embed is None:
                X_test_embed = X_subj_embed_transformed
            else:
                X_test_embed = np.concatenate(
                    (X_test_embed, X_subj_embed_transformed), axis=0
                )

    return X_train_embed, X_test_embed


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
    n_components="auto",
    n_neighbors_LE=125,
    LE_embedding_method="embed+procrustes",
):
    """
    Embed the dFC features into a lower dimensional space using PCA or LE. For LE, it assumes that the samples of the same subject are contiguous.

    for LE, first the LE is applied on each subj separately and then the procrustes transformation is applied to align the embeddings of different subjects.
    All the subjects are transformed into the space of the subject with the highest silhouette score.

    LE_embedding_method: "concat+embed" or "embed+procrustes"
    """
    # make a copy of the data
    X_train = X_train.copy()
    if X_test is not None:
        X_test = X_test.copy()

    if embedding == "PCA":
        # if n_components is not specified, use 95% of the variance
        if n_components == "auto":
            pca = PCA(n_components=0.95, svd_solver="full", whiten=False)
        else:
            pca = PCA(n_components=n_components, svd_solver="full", whiten=False)
        pca.fit(X_train)
        X_train_embed = pca.transform(X_train)
        if X_test is not None:
            X_test_embed = pca.transform(X_test)
        else:
            X_test_embed = None
    elif embedding == "LE":
        # if n_components is not specified, find the intrinsic dimension of the data using training set and based on the silhouette score
        if n_components == "auto":
            n_components = find_intrinsic_dim(
                X=X_train,
                y=y_train,
                subj_label=subj_label_train,
                subjects=train_subjects,
                method="SI",
                n_neighbors_LE=n_neighbors_LE,
                search_range_SI=range(2, 50, 5),
            )

        if LE_embedding_method == "embed+procrustes":
            X_train_embed, X_test_embed = LE_embed_procustes(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                subj_label_train=subj_label_train,
                subj_label_test=subj_label_test,
                train_subjects=train_subjects,
                test_subjects=test_subjects,
                n_components=n_components,
                n_neighbors_LE=n_neighbors_LE,
                procruste_method="generalized",
            )
        elif LE_embedding_method == "concat+embed":
            # since SpectralEmbedding does not have transform method, we need to fit the LE on the whole data
            if X_test is not None:
                X_concat = np.concatenate((X_train, X_test), axis=0)
            else:
                X_concat = X_train
            X_concat_embed = LE_transform(
                X=X_concat,
                n_components=n_components,
                n_neighbors=n_neighbors_LE,
                distance_metric="correlation",
            )
            X_train_embed = X_concat_embed[: X_train.shape[0], :]
            if X_test is not None:
                X_test_embed = X_concat_embed[X_train.shape[0] :, :]
            else:
                X_test_embed = None

    return X_train_embed, X_test_embed


################################# Classification Framework Functions ####################################


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

    ML_RESULT = {"PCA": {}, "LE": {}}
    ML_scores = {
        "subj_id": list(),
        "group": list(),
        "task": list(),
        "run": list(),
        "dFC method": list(),
        "embedding": list(),
    }
    for embedding in ["PCA", "LE"]:
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
            )
        except:
            continue

        # task presence classification

        print("task presence classification ...")

        # logistic regression
        log_reg_RESULT = logistic_regression_classify(
            X_train_embedded, y_train, X_test_embedded, y_test
        )

        # KNN
        KNN_RESULT = KNN_classify(X_train_embedded, y_train, X_test_embedded, y_test)

        # # Random Forest
        # RF_RESULT = random_forest_classify(
        #     X_train_embedded, y_train, X_test_embedded, y_test
        # )

        # # Gradient Boosting
        # GBT_RESULT = gradient_boosting_classify(
        #     X_train_embedded, y_train, X_test_embedded, y_test
        # )

        for key in log_reg_RESULT:
            ML_RESULT[embedding][key] = log_reg_RESULT[key]
        for key in KNN_RESULT:
            ML_RESULT[embedding][key] = KNN_RESULT[key]
        # for key in RF_RESULT:
        #     ML_RESULT[embedding][key] = RF_RESULT[key]
        # for key in GBT_RESULT:
        #     ML_RESULT[embedding][key] = GBT_RESULT[key]

        # measure pred score on each subj
        log_reg = log_reg_RESULT["log_reg_model"]
        KNN = KNN_RESULT["KNN_model"]
        # RF = RF_RESULT["RF_model"]
        # GBT = GBT_RESULT["GB_model"]

        ML_models = {"Logistic regression": log_reg, "KNN": KNN}

        for subj in SUBJECTS:
            ML_scores["subj_id"].append(subj)
            if subj in train_subjects:
                ML_scores["group"].append("train")
                features = X_train_embedded[subj_label_train == subj, :]
                target = y_train[subj_label_train == subj]
            elif subj in test_subjects:
                ML_scores["group"].append("test")
                features = X_test_embedded[subj_label_test == subj, :]
                target = y_test[subj_label_test == subj]

            # measure pred score using different metrics on each subj
            for model_name, model in ML_models.items():
                pred = model.predict(features)
                # accuracy score
                if not f"{model_name} accuracy" in ML_scores:
                    ML_scores[f"{model_name} accuracy"] = list()
                ML_scores[f"{model_name} accuracy"].append(accuracy_score(target, pred))
                # balanced accuracy score
                if not f"{model_name} balanced accuracy" in ML_scores:
                    ML_scores[f"{model_name} balanced accuracy"] = list()
                ML_scores[f"{model_name} balanced accuracy"].append(
                    balanced_accuracy_score(target, pred)
                )
                # precision score
                if not f"{model_name} precision" in ML_scores:
                    ML_scores[f"{model_name} precision"] = list()
                ML_scores[f"{model_name} precision"].append(precision_score(target, pred))
                # recall score
                if not f"{model_name} recall" in ML_scores:
                    ML_scores[f"{model_name} recall"] = list()
                ML_scores[f"{model_name} recall"].append(recall_score(target, pred))
                # f1 score
                if not f"{model_name} f1" in ML_scores:
                    ML_scores[f"{model_name} f1"] = list()
                ML_scores[f"{model_name} f1"].append(f1_score(target, pred))
                # confusion matrix
                tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
                # false positive rate
                if not f"{model_name} fp" in ML_scores:
                    ML_scores[f"{model_name} fp"] = list()
                ML_scores[f"{model_name} fp"].append(fp)
                # false negative rate
                if not f"{model_name} fn" in ML_scores:
                    ML_scores[f"{model_name} fn"] = list()
                ML_scores[f"{model_name} fn"].append(fn)
                # true positive rate
                if not f"{model_name} tp" in ML_scores:
                    ML_scores[f"{model_name} tp"] = list()
                ML_scores[f"{model_name} tp"].append(tp)
                # true negative rate
                if not f"{model_name} tn" in ML_scores:
                    ML_scores[f"{model_name} tn"] = list()
                ML_scores[f"{model_name} tn"].append(tn)
                # average precision score
                if not f"{model_name} average precision" in ML_scores:
                    ML_scores[f"{model_name} average precision"] = list()
                ML_scores[f"{model_name} average precision"].append(
                    average_precision_score(target, pred)
                )

            ML_scores["task"].append(task)
            ML_scores["run"].append(run)
            ML_scores["dFC method"].append(measure_name)
            ML_scores["embedding"].append(embedding)

    return ML_RESULT, ML_scores


################################# Clustering Framework Functions ####################################


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

    clustering_RESULTS = {"PCA": {}, "LE": {}}
    clustering_scores = {
        "subj_id": list(),
        "task": list(),
        "run": list(),
        "dFC method": list(),
        "Kmeans ARI": list(),
        "SI": list(),
        "embedding": list(),
    }
    for embedding in ["PCA", "LE"]:
        # embed dFC features
        try:
            X_embedded, _ = embed_dFC_features(
                train_subjects=SUBJECTS,
                test_subjects=[],
                X_train=X,
                X_test=None,
                y_train=y,
                y_test=None,
                subj_label_train=subj_label,
                subj_label_test=None,
                embedding=embedding,
                n_components="auto",
                n_neighbors_LE=125,
                LE_embedding_method="embed+procrustes",
            )
        except:
            continue

        # clustering
        # apply kmeans clustering to dFC features

        n_clusters = 2  # corresponding to task and rest

        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_embedded)
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

        clustering_RESULTS[embedding] = {
            "StandardScaler": scaler,
            "kmeans": kmeans,
            "ARI": adjusted_rand_score(y, labels_pred),
            # "centroids": centroids_mat,
        }

        for subj in SUBJECTS:
            clustering_scores["subj_id"].append(subj)
            features = X_embedded[subj_label == subj, :]
            target = y[subj_label == subj]

            features_normalized = scaler.transform(features)
            pred_kmeans = kmeans.predict(features_normalized)

            clustering_scores["Kmeans ARI"].append(
                adjusted_rand_score(target, pred_kmeans)
            )

            # silhouette score in terms of separability of original labels, not the clustering labels
            clustering_scores["SI"].append(silhouette_score(features, target))

            clustering_scores["task"].append(task)
            clustering_scores["run"].append(run)
            clustering_scores["dFC method"].append(measure_name)
            clustering_scores["embedding"].append(embedding)

    return clustering_RESULTS, clustering_scores


def task_paradigm_clustering(
    dFC_id,
    TASKS,
    RUNS,
    session,
    roi_root,
    dFC_root,
    normalize_dFC=True,
):
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
            X_new, _, _, _, subj_label_new, _, measure_name_new = dFC_feature_extraction(
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

            # normalize the features
            X_new = zscore(X_new, axis=0)

            if measure_name is not None:
                assert measure_name == measure_name_new, "dFC measure is not consistent."
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

    task_paradigm_clstr_RESULTS = {"PCA": {}, "LE": {}}
    for embedding in ["PCA", "LE"]:
        # embed dFC features
        try:
            X_embed, _ = embed_dFC_features(
                train_subjects=SUBJECTS,
                test_subjects=[],
                X_train=X,
                X_test=None,
                y_train=y,
                y_test=None,
                subj_label_train=subj_label,
                subj_label_test=None,
                embedding=embedding,
                n_components="auto",
                n_neighbors_LE=125,
                LE_embedding_method="embed+procrustes",
            )
        except:
            continue

        # clustering
        # apply kmeans clustering to dFC features

        n_clusters = len(TASKS)  # corresponding to task paradigms

        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_embed)
        kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=5)
        labels_pred = kmeans.fit_predict(X_normalized)

        # # visualize clustering centroids
        # centroids = kmeans.cluster_centers_
        # centroids = pca.inverse_transform(centroids)
        # centroids = scaler.inverse_transform(centroids)
        # n_regions = int((1 + np.sqrt(1 + 8 * centroids.shape[1])) / 2)
        # centroids_mat = dFC_vec2mat(centroids, n_regions)

        task_paradigm_clstr_RESULTS[embedding] = {
            "dFC_method": measure_name,
            "StandardScaler": scaler,
            "kmeans": kmeans,
            "ARI": adjusted_rand_score(y, labels_pred),
            "SI": silhouette_score(X_normalized, y),
            # "centroids": centroids_mat,
            "task_paradigms": TASKS,
        }

    return task_paradigm_clstr_RESULTS


def co_occurrence(task_labels, clstr_labels):
    """
    Calculate the co-occurrence between task labels and clustering labels.
    """
    co_occurrence_matrix = np.zeros(
        (len(np.unique(task_labels)), len(np.unique(clstr_labels)))
    )
    for i, task_label in enumerate(np.unique(task_labels)):
        for j, clstr_label in enumerate(np.unique(clstr_labels)):
            co_occurrence_matrix[i, j] = np.sum(
                (task_labels == task_label) & (clstr_labels == clstr_label)
            )

    # now find the percentage of time each cluster label was present in each task label
    cluster_label_percentage = (
        co_occurrence_matrix / np.sum(co_occurrence_matrix, axis=1)[:, None]
    )
    # make sure that the sum of each row is 1
    assert np.allclose(
        np.sum(cluster_label_percentage, axis=1), 1
    ), "Sum of each row is not 1."

    # now find the percentage of time each task label occupied each cluster label
    task_label_percentage = (
        co_occurrence_matrix / np.sum(co_occurrence_matrix, axis=0)[None, :]
    )
    # make sure that the sum of each column is 1
    assert np.allclose(
        np.sum(task_label_percentage, axis=0), 1
    ), "Sum of each column is not 1."

    return co_occurrence_matrix, cluster_label_percentage, task_label_percentage


def cluster_for_visual(
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

    SUBJECTS = find_available_subjects(
        dFC_root=dFC_root, task=task, run=run, session=session, dFC_id=dFC_id
    )

    print(f"Number of subjects: {len(SUBJECTS)}")

    X, _, y, _, _, _, measure_name = dFC_feature_extraction(
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

    # clustering
    # apply kmeans clustering to dFC features
    n_clusters = 5

    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=5)
    clstr_labels = kmeans.fit_predict(X)  # clstr_labels = (n_samples,)

    # calculate the co-occurrence matrix
    co_occurrence_matrix, cluster_label_percentage, task_label_percentage = co_occurrence(
        y, clstr_labels
    )

    # get centroids
    centroids = kmeans.cluster_centers_
    n_regions = int((1 + np.sqrt(1 + 8 * centroids.shape[1])) / 2)
    centroids_mat = dFC_vec2mat(
        centroids, n_regions
    )  # shape: n_clusters x n_regions x n_regions

    return (
        centroids_mat,
        measure_name,
        co_occurrence_matrix,
        cluster_label_percentage,
        task_label_percentage,
    )
