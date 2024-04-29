import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pydfc import DFC, data_loader, task_utils
from pydfc.dfc_utils import dFC_mat2vec, rank_norm

# Data parameters
dataset = "ds000001"

# main_root = f"./DATA/{dataset}" # for local
main_root = f"../../DATA/task-based/simulated/{dataset}"  # for server
roi_root = f"{main_root}/derivatives/ROI_timeseries"
dFC_root = f"{main_root}/derivatives/dFC_assessed"
output_root = "./ML_RESULTS_KNN_classify"

TASKS = ["task-pulse"]

dynamic_pred = "no"  # 'past' or 'past_and_future' or 'no' (only current TR)
normalize_dFC = True

SUBJECTS = data_loader.find_subj_list(data_root=roi_root, sessions=TASKS)

# randomly select 80% of the subjects for training and 20% for testing using numpy.random.choice
train_subjects = np.random.choice(SUBJECTS, int(0.8 * len(SUBJECTS)), replace=False)
test_subjects = np.setdiff1d(SUBJECTS, train_subjects)

print(
    f"number of train_subjects: {len(train_subjects)} and test_subjects: {len(test_subjects)}"
)


################## TASK FEATURES ##################

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

    for subj in SUBJECTS:
        # event data
        task_data = np.load(
            f"{roi_root}/{subj}_{task}/task_data.npy", allow_pickle="TRUE"
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


################## TASK PRESENCE CLASSIFICATION ##################
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
        print(f"=============== {task} ===============")

        if task == "task-restingstate":
            continue

        X_train = None
        X_test = None
        y_condition_train = None
        y_condition_test = None
        subj_label_train = list()
        subj_label_test = list()

        for subj in SUBJECTS:

            dFC = np.load(
                f"{dFC_root}/{task}/{subj}/dFC_{dFC_id}.npy", allow_pickle="TRUE"
            ).item()

            dFC_mat = dFC.get_dFC_mat()
            TR_array = dFC.TR_array
            if normalize_dFC:
                dFC_mat = rank_norm(dFC_mat)

            dFC_vecs = dFC_mat2vec(dFC_mat)

            # event data
            task_data = np.load(
                f"{roi_root}/{subj}_{task}/task_data.npy", allow_pickle="TRUE"
            ).item()
            Fs_task = task_data["Fs_task"]
            TR_task = 1 / Fs_task

            task_presence = task_utils.extract_task_presence(
                event_labels=task_data["event_labels"],
                TR_task=TR_task,
                TR_mri=task_data["TR_mri"],
                TR_array=TR_array,
                binary=True,
            )

            X_new = dFC_vecs
            y_new = task_presence.ravel()

            if dynamic_pred == "past":
                # concat current TR and two TR before of X_new to predict the current TR of y_new
                # ignore the edge case of the first two TRs
                X_new = np.concatenate(
                    (X_new, np.roll(X_new, 1, axis=0), np.roll(X_new, 2, axis=0)), axis=1
                )
                X_new = X_new[2:, :]
                y_new = y_new[2:]

            elif dynamic_pred == "past_and_future":
                # concat current TR and two TR before and after of X_new to predict the current TR of y_new
                # ignore the edge case of the first and last two TRs
                X_new = np.concatenate(
                    (
                        X_new,
                        np.roll(X_new, 1, axis=0),
                        np.roll(X_new, 2, axis=0),
                        np.roll(X_new, -1, axis=0),
                        np.roll(X_new, -2, axis=0),
                    ),
                    axis=1,
                )
                X_new = X_new[2:-2, :]
                y_new = y_new[2:-2]

            if subj in train_subjects:
                subj_label_train.extend([subj for i in range(X_new.shape[0])])
                if X_train is None and y_condition_train is None:
                    X_train = X_new
                    y_condition_train = y_new
                else:
                    X_train = np.concatenate((X_train, X_new), axis=0)
                    y_condition_train = np.concatenate((y_condition_train, y_new), axis=0)
            elif subj in test_subjects:
                subj_label_test.extend([subj for i in range(X_new.shape[0])])
                if X_test is None and y_condition_test is None:
                    X_test = X_new
                    y_condition_test = y_new
                else:
                    X_test = np.concatenate((X_test, X_new), axis=0)
                    y_condition_test = np.concatenate((y_condition_test, y_new), axis=0)

        print(
            X_train.shape, X_test.shape, y_condition_train.shape, y_condition_test.shape
        )
        subj_label_train = np.array(subj_label_train)
        subj_label_test = np.array(subj_label_test)
        print(subj_label_train.shape, subj_label_test.shape)

        # task presence classification

        print("task presence classification ...")

        # find num_PCs
        pca = PCA(svd_solver="full", whiten=False)
        pca.fit(X_train)
        num_PCs = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0][0] + 1

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
        knn_gscv.fit(X_train, y_condition_train)

        n_neighbors = knn_gscv.best_params_["kneighborsclassifier__n_neighbors"]

        neigh = make_pipeline(
            StandardScaler(),
            PCA(n_components=num_PCs),
            KNeighborsClassifier(n_neighbors=n_neighbors),
        ).fit(X_train, y_condition_train)

        ML_RESULT[task] = {
            "pca": pca,
            "num_PCs": num_PCs,
            "cv_results": knn_gscv.cv_results_,
            "KNN": neigh,
            "KNN train score": neigh.score(X_train, y_condition_train),
            "KNN test score": neigh.score(X_test, y_condition_test),
        }

        print(
            f"KNN train score {dFC.measure.measure_name} {task}: {neigh.score(X_train, y_condition_train)}"
        )
        print(
            f"KNN test score {dFC.measure.measure_name} {task}: {neigh.score(X_test, y_condition_test)}"
        )

        # measure pred score on each subj

        for subj in SUBJECTS:
            ML_scores["subj_id"].append(subj)
            if subj in train_subjects:
                ML_scores["group"].append("train")
                features = X_train[subj_label_train == subj, :]
                target = y_condition_train[subj_label_train == subj]
            elif subj in test_subjects:
                ML_scores["group"].append("test")
                features = X_test[subj_label_test == subj, :]
                target = y_condition_test[subj_label_test == subj]

            pred = neigh.predict(features)

            ML_scores["KNN accuracy"].append(balanced_accuracy_score(target, pred))

            ML_scores["task"].append(task)
            ML_scores["dFC method"].append(dFC.measure.measure_name)

    folder = f"{output_root}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(f"{folder}/ML_RESULT_{dFC.measure.measure_name}.npy", ML_RESULT)

np.save(f"{folder}/task_features_KNN_classify.npy", task_features)
np.save(f"{folder}/ML_scores_KNN_classify.npy", ML_scores)
