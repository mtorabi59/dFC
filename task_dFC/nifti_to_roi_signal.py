import argparse
import json
import os
import warnings

import numpy as np

from pydfc import data_loader, task_utils

# warnings.simplefilter("ignore")


################################# FUNCTIONS #################################
def run_roi_signal_extraction(
    subj, task, main_root, fmriprep_root, bold_suffix, output_root
):
    """
    Extract ROI signals and task labels for a given subject and task
    """
    # find the func file for this subject and task
    try:
        ALL_TASK_FILES = os.listdir(f"{fmriprep_root}/{subj}/func/")
    except FileNotFoundError:
        print(f"Subject {subj} not found in {fmriprep_root}")
        return

    ALL_TASK_FILES = [
        file_i
        for file_i in ALL_TASK_FILES
        if (bold_suffix in file_i) and (task in file_i)
    ]  # only keep the denoised files? or use the original files?

    if not len(ALL_TASK_FILES) >= 1:
        # if the func file is not found, exclude the subject
        print(f"Func file not found for {subj} {task}")
        return

    # there might be multiple runs for the same task
    # check if "_run" exists in all the task file names
    if all(["_run" in task_file for task_file in ALL_TASK_FILES]):
        multi_run_flag = True
        # find all the runs
        RUNS = [
            task_file[
                task_file.find("_run")
                + 1 : task_file.find("_run")
                + 1
                + task_file[task_file.find("_run") + 1 :].find("_")
            ]
            for task_file in ALL_TASK_FILES
        ]
        # sort
        RUNS.sort()
        print(f"Found multiple runs for {subj} {task}: {RUNS}")
    else:
        multi_run_flag = False
        RUNS = [""]

    for run in RUNS:
        task_file = [file_i for file_i in ALL_TASK_FILES if run in file_i][0]
        nifti_file = f"{fmriprep_root}/{subj}/func/{task_file}"
        info_file = (
            f"{main_root}/bids/{subj}/func/{task_file.replace(bold_suffix, '_bold.json')}"
        )

        ################################# LOAD JSON INFO #########################
        # Opening JSON file as a dictionary
        f = open(info_file)
        acquisition_data = json.load(f)
        f.close()
        TR_mri = acquisition_data["RepetitionTime"]
        ################################# EXTRACT TIME SERIES #########################
        # extract ROI signals and convert to TIME_SERIES object
        time_series = data_loader.nifti2timeseries(
            nifti_file=nifti_file,
            n_rois=100,
            Fs=1 / TR_mri,
            subj_id=subj,
            confound_strategy="no_motion",
            standardize="zscore",
            TS_name="BOLD",
            session=task,
        )
        num_time_mri = time_series.n_time
        ################################# EXTRACT TASK LABELS #########################
        oversampling = 50  # more samples per TR than the func data to have a better event_labels time resolution
        if task == "task-restingstate":
            events = []
            event_types = ["rest"]
            event_labels = np.zeros((int(num_time_mri * oversampling), 1))
            task_labels = np.zeros((int(num_time_mri * oversampling), 1))
            Fs_task = float(1 / TR_mri) * oversampling
        else:
            task_events_root = f"{main_root}/bids/{subj}/func"
            ALL_EVENTS_FILES = os.listdir(task_events_root)
            ALL_EVENTS_FILES = [
                file_i
                for file_i in ALL_EVENTS_FILES
                if (subj in file_i)
                and (task in file_i)
                and (run in file_i)
                and ("events.tsv" in file_i)
            ]
            if not len(ALL_EVENTS_FILES) == 1:
                # if the events file is not found, exclude the subject
                print(f"Events file not found for {subj} {task} {run}")
                return
            # load the tsv events file
            events_file = f"{task_events_root}/{ALL_EVENTS_FILES[0]}"
            events = np.genfromtxt(events_file, delimiter="\t", dtype=str)
            # get the event labels
            event_labels, Fs_task, event_types = task_utils.events_time_to_labels(
                events=events,
                TR_mri=TR_mri,
                num_time_mri=num_time_mri,
                event_types=None,
                oversampling=oversampling,
                return_0_1=False,
            )
            # fill task labels with task's index
            task_labels = np.ones((int(num_time_mri * oversampling), 1)) * TASKS.index(
                task
            )
        ################################# SAVE #################################
        # save the ROI time series and task data
        task_data = {
            "task": task,
            "task_labels": task_labels,
            "task_types": TASKS,
            "event_labels": event_labels,
            "event_types": event_types,
            "events": events,
            "Fs_task": Fs_task,
            "TR_mri": TR_mri,
            "num_time_mri": num_time_mri,
        }
        if multi_run_flag:
            output_file_prefix = f"{subj}_{task}_{run}"
        else:
            output_file_prefix = f"{subj}_{task}"
        if not os.path.exists(f"{output_root}/{subj}/"):
            os.makedirs(f"{output_root}/{subj}/")
        np.save(f"{output_root}/{subj}/{output_file_prefix}_time-series.npy", time_series)
        np.save(f"{output_root}/{subj}/{output_file_prefix}_task-data.npy", task_data)


########################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to convert nifti files to ROI signals for a given participant.
    """

    parser = argparse.ArgumentParser(description=HELPTEXT)

    parser.add_argument("--dataset_info", type=str, help="path to dataset info file")
    parser.add_argument("--participant_id", type=str, help="participant id")

    args = parser.parse_args()

    dataset_info_file = args.dataset_info
    participant_id = args.participant_id

    # Read global configs
    with open(dataset_info_file, "r") as f:
        dataset_info = json.load(f)

    print(
        f"subject-level ROI signal extraction CODE started running ... for subject: {participant_id} ..."
    )

    TASKS = dataset_info["TASKS"]

    if "{dataset}" in dataset_info["main_root"]:
        main_root = dataset_info["main_root"].replace(
            "{dataset}", dataset_info["dataset"]
        )
    else:
        main_root = dataset_info["main_root"]

    if "{main_root}" in dataset_info["fmriprep_root"]:
        fmriprep_root = dataset_info["fmriprep_root"].replace("{main_root}", main_root)
    else:
        fmriprep_root = dataset_info["fmriprep_root"]

    if "{main_root}" in dataset_info["roi_root"]:
        output_root = dataset_info["roi_root"].replace("{main_root}", main_root)
    else:
        output_root = dataset_info["roi_root"]

    for task in TASKS:
        run_roi_signal_extraction(
            subj=participant_id,
            task=task,
            main_root=main_root,
            fmriprep_root=fmriprep_root,
            bold_suffix=dataset_info["bold_suffix"],
            output_root=output_root,
        )

    print(
        f"subject-level ROI signal extraction CODE finished running ... for subject: {participant_id} ..."
    )

####################################################################
