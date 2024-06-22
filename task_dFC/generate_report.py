import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import image, plotting
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pydfc import DFC, data_loader, task_utils
from pydfc.dfc_utils import TR_intersection, dFC_mat2vec, dFC_vec2mat, rank_norm

################################# Parameters ####################################

fig_dpi = 120
fig_bbox_inches = "tight"
fig_pad = 0.1
show_title = True
save_fig_format = "png"  # pdf, png,

#######################################################################################


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


# def plot_anatomical(
#     fmriprep_root,
#     subj,
#     anat_suffix,
#     session=None,
# ):
#     anat_suffix = '_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
#     anat_file = f"{fmriprep_root}/{subj}/anat/{subj}{anat_suffix}"
#     display = plotting.plot_anat(anat_file, title="plot_anat")


# def plot_functional(
#     fmriprep_root,
#     subj,
#     bold_suffix,
#     task,
#     session=None,
#     run=None,
# ):
#     if session is None:
#         if run is None:
#             task_file = f"{subj}_{task}{bold_suffix}"
#         else:
#             task_file = f"{subj}_{task}_{run}{bold_suffix}"
#         func_file = f"{fmriprep_root}/{subj}/func/{task_file}"
#     else:
#         if run is None:
#             task_file = f"{subj}_{session}_{task}{bold_suffix}"
#         else:
#             task_file = f"{subj}_{session}_{task}_{run}{bold_suffix}"
#         func_file = f"{fmriprep_root}/{subj}/{session}/func/{task_file}"

#     # Compute voxel-wise mean functional image across time dimension. Now we have
#     # functional image in 3D assigned in mean_func_img
#     mean_func_img = image.mean_img(func_file)
#     display = plotting.plot_anat(mean_func_img, title="plot_func")


def plot_roi_signals(
    roi_root,
    subj,
    task,
    start_time,
    end_time,
    output_root,
    nodes_list=range(0, 10),
    session=None,
    run=None,
):
    if session is None:
        if run is None:
            file_name = "{subj_id}_{task}_time-series.npy"
        else:
            file_name = "{subj_id}_{task}_{run}_time-series.npy"
    else:
        if run is None:
            file_name = "{subj_id}_{session}_{task}_time-series.npy"
        else:
            file_name = "{subj_id}_{session}_{task}_{run}_time-series.npy"

    task_data = load_task_data(roi_root, subj, task, run, session)
    TR_mri = task_data["TR_mri"]

    BOLD = data_loader.load_TS(
        data_root=roi_root,
        file_name=file_name,
        subj_id2load=subj,
        task=task,
        run=run,
        session=session,
    )

    time = np.arange(0, BOLD.data.shape[1]) * TR_mri
    start_TR = int(start_time / TR_mri)
    end_TR = int(end_time / TR_mri)
    # keep the figure width proportional to the number of time points
    fig_width = int(2.5 * (end_time - start_time) / TR_mri)
    plt.figure(figsize=(fig_width, 5))
    for i in nodes_list:
        plt.plot(time[start_TR:end_TR], BOLD.data[i, start_TR:end_TR], linewidth=4)
    # put vertical lines at the start of each TR
    for TR in range(start_TR, end_TR):
        plt.axvline(x=TR * TR_mri, color="r", linestyle="--")
    # show TR labels on the red lines with a small font and at the top
    for TR in range(start_TR, end_TR):
        plt.text(TR * TR_mri, 1.2, f"TR {TR}", fontsize=8, color="black", ha="center")
    if show_title:
        plt.title("ROI signals")
    plt.xlabel("Time (s)")

    # save the figure
    output_dir = f"{output_root}/subject_results/{subj}/ROI_signals"
    if session is not None:
        output_dir = f"{output_dir}/{session}"
    output_dir = f"{output_dir}/{task}"
    if run is not None:
        output_dir = f"{output_dir}/{run}"
    output_dir = f"{output_dir}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(
        f"{output_dir}/ROI_signals.{save_fig_format}",
        dpi=fig_dpi,
        bbox_inches=fig_bbox_inches,
        pad_inches=fig_pad,
        format=save_fig_format,
    )

    plt.close()


def plot_event_labels(
    roi_root,
    subj,
    task,
    start_time,
    end_time,
    output_root,
    run=None,
    session=None,
):
    task_data = load_task_data(roi_root, subj, task, run, session)
    Fs_task = task_data["Fs_task"]
    TR_task = 1 / Fs_task
    TR_mri = task_data["TR_mri"]

    time = np.arange(0, task_data["event_labels"].shape[0]) / Fs_task
    start_timepoint = int(start_time / TR_task)
    end_timepoint = int(end_time / TR_task)
    # keep the figure width proportional to the number of time points
    fig_width = int(2.5 * (end_time - start_time) / TR_mri)
    plt.figure(figsize=(fig_width, 5))
    plt.plot(
        time[start_timepoint:end_timepoint],
        task_data["event_labels"][start_timepoint:end_timepoint],
        linewidth=4,
    )
    plt.title("Event labels")
    plt.xlabel("Time (s)")

    # save the figure
    output_dir = f"{output_root}/subject_results/{subj}/event_labels"
    if session is not None:
        output_dir = f"{output_dir}/{session}"
    output_dir = f"{output_dir}/{task}"
    if run is not None:
        output_dir = f"{output_dir}/{run}"
    output_dir = f"{output_dir}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(
        f"{output_dir}/event_labels.{save_fig_format}",
        dpi=fig_dpi,
        bbox_inches=fig_bbox_inches,
        pad_inches=fig_pad,
        format=save_fig_format,
    )

    plt.close()


def plot_task_presence(
    roi_root,
    subj,
    task,
    start_time,
    end_time,
    output_root,
    run=None,
    session=None,
):
    task_data = load_task_data(roi_root, subj, task, run, session)
    Fs_task = task_data["Fs_task"]
    TR_task = 1 / Fs_task
    TR_mri = task_data["TR_mri"]
    Fs_mri = 1 / TR_mri

    task_presence_non_binarized = task_utils.extract_task_presence(
        event_labels=task_data["event_labels"],
        TR_task=TR_task,
        TR_mri=task_data["TR_mri"],
        binary=False,
    )

    task_presence = task_utils.extract_task_presence(
        event_labels=task_data["event_labels"],
        TR_task=TR_task,
        TR_mri=task_data["TR_mri"],
        binary=True,
        binarizing_method="mean",
    )

    time = np.arange(0, task_presence.shape[0]) / Fs_mri
    start_TR = int(start_time / TR_mri)
    end_TR = int(end_time / TR_mri)
    # keep the figure width proportional to the number of time points in data
    fig_width = int(2.5 * (end_time - start_time) / TR_mri)
    plt.figure(figsize=(fig_width, 5))
    plt.plot(
        time[start_TR:end_TR], task_presence_non_binarized[start_TR:end_TR], linewidth=4
    )
    plt.plot(time[start_TR:end_TR], task_presence[start_TR:end_TR], linewidth=4)
    # plot mean of task presence_non_binarized as a line
    plt.plot(
        time[start_TR:end_TR],
        np.mean(task_presence_non_binarized) * np.ones_like(time[start_TR:end_TR]),
        linewidth=4,
    )
    # put vertical lines at the start of each TR
    for TR in range(start_TR, end_TR):
        plt.axvline(x=TR * TR_mri, color="r", linestyle="--")
    # show TR labels on the red lines with a small font and at the top
    for TR in range(start_TR, end_TR):
        plt.text(TR * TR_mri, 1.2, f"TR {TR}", fontsize=8, color="black", ha="center")
    plt.title("Task presence")
    plt.xlabel("Time (s)")

    # save the figure
    output_dir = f"{output_root}/subject_results/{subj}/task_presence"
    if session is not None:
        output_dir = f"{output_dir}/{session}"
    output_dir = f"{output_dir}/{task}"
    if run is not None:
        output_dir = f"{output_dir}/{run}"
    output_dir = f"{output_dir}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(
        f"{output_dir}/task_presence.{save_fig_format}",
        dpi=fig_dpi,
        bbox_inches=fig_bbox_inches,
        pad_inches=fig_pad,
        format=save_fig_format,
    )

    plt.close()


def calculate_subj_lvl_task_presence_characteristics(
    roi_root,
    subj,
    task,
    run=None,
    session=None,
):
    task_data = load_task_data(roi_root, subj, task, run, session)
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
    num_of_transitions, relative_transition_freq = task_utils.transition_freq(
        task_presence
    )

    print(f"Relative task on: {relative_task_on}")
    print(f"Average task duration: {avg_task_duration} seconds")
    print(f"Average rest duration: {avg_rest_duration} seconds")
    print(f"Number of transitions: {num_of_transitions}")
    print(f"Relative transition frequency: {relative_transition_freq}")


# def plot_FCS():
#     visualize_FCS(
#         measure,
#         normalize=True,
#         fix_lim=False,
#         save_image=save_image,
#         output_root=output_root + "FCS/",
#     )


def plot_dFC_matrices(
    dFC_root,
    subj,
    task,
    start_time,
    end_time,
    output_root,
    run=None,
    session=None,
):
    """
    plot dFC matrices for a given subject, task, run, session, start_time and end_time
    parameters:
    ----------
        dFC_root: str, path to dFC results
        subj: str, subject id
        task: str, task name
        start_time: float, start time in seconds
        end_time: float, end time in seconds
    """
    task_data = load_task_data(roi_root, subj, task, run, session)
    TR_mri = task_data["TR_mri"]

    dFC_lst = list()
    for dFC_id in range(0, 20):  # change this to the number of dFCs you have
        try:
            dFC = load_dFC(dFC_root, subj, task, dFC_id, run, session)
            dFC_lst.append(dFC)
        except Exception:
            pass

    TRs = TR_intersection(dFC_lst)
    start_TR = int(start_time / TR_mri)
    end_TR = int(end_time / TR_mri)
    start_TR_idx = np.where(np.array(TRs) >= start_TR)[0][0]
    end_TR_idx = np.where(np.array(TRs) <= end_TR)[0][-1]
    chosen_TRs = TRs[start_TR_idx:end_TR_idx]

    output_dir = f"{output_root}/subject_results/{subj}/dFC_matrices"
    if session is not None:
        output_dir = f"{output_dir}/{session}"
    output_dir = f"{output_dir}/{task}"
    if run is not None:
        output_dir = f"{output_dir}/{run}"
    output_dir = f"{output_dir}/"

    for dFC in dFC_lst:
        dFC.visualize_dFC(
            TRs=chosen_TRs,
            normalize=False,
            rank_norm=True,
            fix_lim=False,
            save_image=True,
            output_root=output_dir,
        )


def plot_ML_results(
    ML_root, output_root, task, run=None, session=None, ML_algorithm="KNN"
):
    """
    Plot the ML results for a given task, run and session.
    parameters:
    ----------
        ML_root: str, path to ML results
        output_root: str, path to save the figures
        task: str, task name
        run: int, run number
        session: str, session name
        ML_algorithm: str, ML algorithm name (default: KNN, other options: Logistic regression)
    """
    if session is None:
        ML_scores = np.load(
            f"{ML_root}/ML_scores_classify.npy", allow_pickle="TRUE"
        ).item()
    else:
        ML_scores = np.load(
            f"{ML_root}/{session}/ML_scores_classify.npy", allow_pickle="TRUE"
        ).item()

    sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 1.0})

    sns.set_style("darkgrid")

    dataframe = pd.DataFrame(ML_scores)
    if run is not None:
        dataframe = dataframe[dataframe["run"] == run]

    plt.figure(figsize=(10, 5))
    g = sns.pointplot(
        data=dataframe[dataframe["task"] == task],
        x="dFC method",
        y=f"{ML_algorithm} accuracy",
        hue="group",
        errorbar="sd",
        linestyle="none",
        dodge=True,
        capsize=0.1,
    )
    g.axhline(0.5, color="r", linestyle="--")
    if show_title:
        g.set_title(task, fontdict={"fontsize": 10, "fontweight": "bold"})

    # save the figure
    if session is None:
        output_dir = f"{output_root}/group_results/classification"
    else:
        output_dir = f"{output_root}/group_results/classification/{session}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if ML_algorithm == "Logistic regression":
        ML_algorithm_name = "LogReg"
    elif ML_algorithm == "KNN":
        ML_algorithm_name = "KNN"

    if run is None:
        plt.savefig(
            f"{output_dir}/ML_results_classify_{ML_algorithm_name}_{task}.{save_fig_format}",
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
    else:
        plt.savefig(
            f"{output_dir}/ML_results_classify_{ML_algorithm_name}_{task}_{run}.{save_fig_format}",
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )

    plt.close()


def plot_task_presence_characteristics():
    pass


def plot_clustering_results():
    pass


# def plot_dFC_clustering(
#     dFC_root,
#     subj,
#     task,
#     start_time,
#     end_time,
#     run=None,
#     session=None,
#     normalize_dFC=True,
# ):
#     task_data = load_task_data(roi_root, subj, task, run, session)
#     TR_mri = task_data['TR_mri']

#     dFC_lst = list()
#     for dFC_id in range(0, 20): # change this to the number of dFCs you have
#         try:
#             dFC = load_dFC(dFC_root, subj, task, dFC_id, run, session)
#             dFC_lst.append(dFC)
#         except Exception:
#             pass

#     for dFC in dFC_lst:
#         dFC_mat = dFC.get_dFC_mat()
#         TR_array = dFC.TR_array
#         if normalize_dFC:
#             dFC_mat = rank_norm(dFC_mat)
#         dFC_vecs = dFC_mat2vec(dFC_mat)

#         # apply kmeans clustering with PCA to dFC vectors
#         n_clusters = 2

#         scaler = StandardScaler()
#         dFC_vecs = scaler.fit_transform(dFC_vecs)
#         # PCA
#         # find number of components that explain 95% of variance
#         pca = PCA()
#         pca.fit(dFC_vecs)
#         n_components = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0][0] + 1
#         # print(f"Number of components: {n_components}")
#         pca = PCA(n_components=n_components)
#         pca.fit(dFC_vecs)


#         dFC_vecs_pca = pca.transform(dFC_vecs)
#         kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=20)
#         labels_pred = kmeans.fit_predict(dFC_vecs_pca)

#         start_TR = int(start_time/TR_mri)
#         end_TR = int(end_time/TR_mri)
#         start_TR_idx = np.where(np.array(TR_array) >= start_TR)[0][0]
#         end_TR_idx = np.where(np.array(TR_array) <= end_TR)[0][-1]

#         # plot labels_pred
#         plt.figure(figsize=(35, 2))
#         plt.plot(time[start_TR:end_TR], labels_pred[start_TR_idx:end_TR_idx], linewidth=4)
#         # put vertical lines at the start of each TR
#         for TR in chosen_TRs:
#             plt.axvline(x=TR*TR_mri, color='r', linestyle='--')
#             # plt.text(TR*TR_mri, 0.5, f"TR {TR}", fontsize=8, color='black', ha='center')
#         plt.title(f"Cluster labels of {dFC.measure.measure_name}")
#         plt.xlabel('Time (s)')
#         plt.show()


def create_html_report(
    subj,
    SESSIONS,
    TASKS,
    RUNS,
    reports_root,
):
    """
    This function creates an html report for the subject results
    using the generated figures.
    """

    # create html report
    subj_dir = f"{reports_root}/subject_results/{subj}"
    file = open(f"{subj_dir}/report.html", "w")
    file.write("<html>\n")
    file.write("<head>\n")
    file.write(f"<title>Subject {subj} Results</title>\n")
    file.write("</head>\n")
    file.write("<body>\n")
    file.write(f"<h1>Subject {subj} Results</h1>\n")
    for session in SESSIONS:
        if session is not None:
            file.write(f"<h1> {session} </h1>\n")
        for task in TASKS:
            file.write(f"<h1> {task} </h1>\n")
            for run in RUNS[task]:
                if run is not None:
                    file.write(f"<h2> {run} </h2>\n")
                if session is not None:
                    session_task_run_dir = f"{session}/{task}"
                else:
                    session_task_run_dir = f"{task}"
                if run is not None:
                    session_task_run_dir = f"{session_task_run_dir}/{run}"

                img_height = 100

                # display ROI signals
                ROI_signals_img = (
                    f"{subj_dir}/ROI_signals/{session_task_run_dir}/ROI_signals.png"
                )
                img = plt.imread(ROI_signals_img)
                height, width, _ = img.shape
                # change the width so that height equals img_height
                width = int(width * img_height / height)
                # replace the path to the image with a relative path
                ROI_signals_img = ROI_signals_img.replace(subj_dir, ".")
                file.write(
                    f"<img src='{ROI_signals_img}' alt='ROI signals' width='{width}' height='{img_height}'>\n"
                )
                file.write("<br>\n")

                # display event labels
                event_labels_img = (
                    f"{subj_dir}/event_labels/{session_task_run_dir}/event_labels.png"
                )
                img = plt.imread(event_labels_img)
                height, width, _ = img.shape
                # change the width so that height equals img_height
                width = int(width * img_height / height)
                # replace the path to the image with a relative path
                event_labels_img = event_labels_img.replace(subj_dir, ".")
                file.write(
                    f"<img src='{event_labels_img}' alt='Event labels' width='{width}' height='{img_height}'>\n"
                )
                file.write("<br>\n")

                # display task presence
                task_presence_img = (
                    f"{subj_dir}/task_presence/{session_task_run_dir}/task_presence.png"
                )
                img = plt.imread(task_presence_img)
                height, width, _ = img.shape
                # change the width so that height equals img_height
                width = int(width * img_height / height)
                # replace the path to the image with a relative path
                task_presence_img = task_presence_img.replace(subj_dir, ".")
                file.write(
                    f"<img src='{task_presence_img}' alt='Task presence' width='{width}' height='{img_height}'>\n"
                )
                file.write("<br>\n")

                # display dFC matrices
                img_height = 45
                # for dFC matrices find all png files in the directory
                dFC_matrices_dir = f"{subj_dir}/dFC_matrices/{session_task_run_dir}"
                if os.path.exists(dFC_matrices_dir):
                    for file_name in os.listdir(dFC_matrices_dir):
                        if file_name.endswith(".png"):
                            file.write(f"<h3>{file_name[:file_name.find('_dFC')]}</h3>\n")
                            dFC_matrices_img = f"{dFC_matrices_dir}/{file_name}"
                            # get the original size of the image
                            img = plt.imread(dFC_matrices_img)
                            height, width, _ = img.shape
                            # change the width so that height equals img_height
                            width = int(width * img_height / height)
                            # replace the path to the image with a relative path
                            dFC_matrices_img = dFC_matrices_img.replace(subj_dir, ".")
                            file.write(
                                f"<img src='{dFC_matrices_img}' alt='{file_name}' width='{width}' height='{img_height}'>\n"
                            )
                            file.write("<br>\n")
    file.write("</body>\n")
    file.write("</html>\n")
    file.close()


#######################################################################################
if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to generate a report of subject results.
    """

    parser = argparse.ArgumentParser(description=HELPTEXT)

    parser.add_argument("--dataset_info", type=str, help="path to dataset info file")
    parser.add_argument("--subj_list", type=str, help="path to subject list file")

    args = parser.parse_args()

    dataset_info_file = args.dataset_info
    subj_list_file = args.subj_list

    # Read dataset info
    with open(dataset_info_file, "r") as f:
        dataset_info = json.load(f)

    # Read subject list file, a txt file with one subject id per line
    with open(subj_list_file, "r") as f:
        SUBJECTS = f.read().splitlines()

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

    if "{main_root}" in dataset_info["reports_root"]:
        reports_root = dataset_info["reports_root"].replace("{main_root}", main_root)
    else:
        reports_root = dataset_info["reports_root"]

    print("Generating report...")

    # Generate report only 5 random subjects
    # SUBJECTS = np.random.choice(SUBJECTS, 5)
    SUBJECTS = SUBJECTS[:1]

    start_time = 0
    end_time = 200

    for subj in SUBJECTS:
        for session in SESSIONS:
            for task in TASKS:
                for run in RUNS[task]:

                    try:
                        plot_dFC_matrices(
                            dFC_root=dFC_root,
                            subj=subj,
                            task=task,
                            start_time=start_time,
                            end_time=end_time,
                            output_root=reports_root,
                            run=run,
                            session=session,
                        )
                    except Exception as e:
                        print(f"Error in plotting dFC matrices: {e}")

                    try:
                        plot_roi_signals(
                            roi_root=roi_root,
                            subj=subj,
                            task=task,
                            start_time=start_time,
                            end_time=end_time,
                            nodes_list=range(0, 10),
                            output_root=reports_root,
                            run=run,
                            session=session,
                        )
                    except Exception as e:
                        print(f"Error in plotting ROI signals: {e}")

                    try:
                        plot_event_labels(
                            roi_root=roi_root,
                            subj=subj,
                            task=task,
                            start_time=start_time,
                            end_time=end_time,
                            output_root=reports_root,
                            run=run,
                            session=session,
                        )
                    except Exception as e:
                        print(f"Error in plotting event labels: {e}")

                    try:
                        plot_task_presence(
                            roi_root=roi_root,
                            subj=subj,
                            task=task,
                            start_time=start_time,
                            end_time=end_time,
                            output_root=reports_root,
                            run=run,
                            session=session,
                        )
                    except Exception as e:
                        print(f"Error in plotting task presence: {e}")
        # create html report
        try:
            create_html_report(
                subj=subj,
                SESSIONS=SESSIONS,
                TASKS=TASKS,
                RUNS=RUNS,
                reports_root=reports_root,
            )
        except Exception as e:
            print(f"Error in creating html report: {e}")

    for session in SESSIONS:
        for task in TASKS:
            for run in RUNS[task]:
                plot_ML_results(
                    ML_root=ML_root,
                    output_root=reports_root,
                    task=task,
                    run=run,
                    session=session,
                    ML_algorithm="KNN",
                )
                # plot_ML_results(
                #     ML_root=ML_root,
                #     output_root=reports_root,
                #     task=task,
                #     run=run,
                #     session=session,
                #     ML_algorithm="Logistic regression",
                # )

    print("Report generated successfully!")

#######################################################################################
