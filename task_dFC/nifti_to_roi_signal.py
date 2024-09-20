import argparse
import json
import os
import warnings

import numpy as np

from pydfc import data_loader, task_utils

# warnings.simplefilter("ignore")


################################# FUNCTIONS #################################
def run_roi_signal_extraction(
    subj,
    task,
    main_root,
    fmriprep_root,
    bold_suffix,
    output_root,
    session=None,
    RUNS=[None],
    trial_type_label="trial_type",
    rest_labels=[],
):
    """
    Extract ROI signals and task labels for a given subject and task
    and optionally session.
    """
    if session is None:
        session_str = ""
    else:
        session_str = session
    # find the func file for this subject and task
    try:
        if session is None:
            ALL_TASK_FILES = os.listdir(f"{fmriprep_root}/{subj}/func/")
        else:
            ALL_TASK_FILES = os.listdir(f"{fmriprep_root}/{subj}/{session}/func/")
    except FileNotFoundError:
        print(f"Subject {subj} {session_str} not found in {fmriprep_root}")
        return

    ALL_TASK_FILES = [
        file_i
        for file_i in ALL_TASK_FILES
        if (bold_suffix in file_i) and (f"_{task}_" in file_i)
    ]  # only keep the denoised files? or use the original files?

    if not len(ALL_TASK_FILES) >= 1:
        # if the func file is not found, exclude the subject
        print(f"Func file not found for {subj} {session_str} {task}")
        return

    for run in RUNS:
        if run is None:
            task_file = ALL_TASK_FILES[0]
        else:
            task_file = [file_i for file_i in ALL_TASK_FILES if f"_{run}_" in file_i][0]
        if session is None:
            nifti_file = f"{fmriprep_root}/{subj}/func/{task_file}"
            task_events_root = f"{main_root}/bids/{subj}/func"
        else:
            nifti_file = f"{fmriprep_root}/{subj}/{session}/func/{task_file}"
            task_events_root = f"{main_root}/bids/{subj}/{session}/func"
        info_file = f"{task_events_root}/{task_file.replace(bold_suffix, '_bold.json')}"

        # in some cases the info file is common for all subjects and can be found in f"{main_root}/bids"
        if not os.path.exists(info_file):
            ALL_COMMON_FILES = os.listdir(f"{main_root}/bids/")
            ALL_COMMON_FILES = [
                file_i
                for file_i in ALL_COMMON_FILES
                if (f"{task}_" in file_i) and ("_bold.json" in file_i)
            ]
            if len(ALL_COMMON_FILES) == 1:
                info_file = f"{main_root}/bids/{ALL_COMMON_FILES[0]}"
        if not os.path.exists(info_file):
            # if the info file is not found, exclude the subject
            if run is None:
                print(f"bold.json info file not found for {subj} {session_str} {task}")
            else:
                print(
                    f"bold.json info file not found for {subj} {session_str} {task} {run}"
                )
            return
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

        ALL_EVENTS_FILES = os.listdir(task_events_root)
        ALL_EVENTS_FILES = [
            file_i
            for file_i in ALL_EVENTS_FILES
            if (f"{subj}_" in file_i)
            and (f"_{task}_" in file_i)
            and ("events.tsv" in file_i)
        ]
        if not run is None:
            ALL_EVENTS_FILES = [
                file_i for file_i in ALL_EVENTS_FILES if f"_{run}_" in file_i
            ]
        if not session is None:
            ALL_EVENTS_FILES = [
                file_i for file_i in ALL_EVENTS_FILES if f"_{session}_" in file_i
            ]

        if not len(ALL_EVENTS_FILES) == 1:
            # in some cases the event file is common for all subjects and can be found in f"{main_root}/bids"
            ALL_EVENTS_FILES_COMMON = os.listdir(f"{main_root}/bids/")
            ALL_EVENTS_FILES_COMMON = [
                file_i
                for file_i in ALL_EVENTS_FILES_COMMON
                if (f"{task}_" in file_i) and ("events.tsv" in file_i)
            ]
            if len(ALL_EVENTS_FILES_COMMON) == 1:
                events_file = f"{main_root}/bids/{ALL_EVENTS_FILES_COMMON[0]}"
            else:
                # if the events file is not found, exclude the subject
                if run is None:
                    print(f"Events file not found for {subj} {session_str} {task}")
                else:
                    print(f"Events file not found for {subj} {session_str} {task} {run}")
                return
        else:
            events_file = f"{task_events_root}/{ALL_EVENTS_FILES[0]}"

        # load the tsv events file
        events = np.genfromtxt(events_file, delimiter="\t", dtype=str)
        # get the event labels
        event_labels, Fs_task, event_types = task_utils.events_time_to_labels(
            events=events,
            TR_mri=TR_mri,
            num_time_mri=num_time_mri,
            event_types=None,
            oversampling=oversampling,
            trial_type_label=trial_type_label,
            rest_labels=rest_labels,
            return_0_1=False,
        )
        # fill task labels with task's index
        task_labels = np.ones((int(num_time_mri * oversampling), 1)) * TASKS.index(task)
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

        if session is None:
            subj_session_prefix = f"{subj}"
            output_dir = f"{output_root}/{subj}"
        else:
            subj_session_prefix = f"{subj}_{session}"
            output_dir = f"{output_root}/{subj}/{session}"

        if run is None:
            output_file_prefix = f"{subj_session_prefix}_{task}"
        else:
            output_file_prefix = f"{subj_session_prefix}_{task}_{run}"

        if not os.path.exists(f"{output_dir}/"):
            os.makedirs(f"{output_dir}/")
        np.save(f"{output_dir}/{output_file_prefix}_time-series.npy", time_series)
        np.save(f"{output_dir}/{output_file_prefix}_task-data.npy", task_data)


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

    # Read dataset info
    with open(dataset_info_file, "r") as f:
        dataset_info = json.load(f)

    print(
        f"subject-level ROI signal extraction CODE started running ... for subject: {participant_id} ..."
    )

    TASKS = dataset_info["TASKS"]

    if "SESSIONS" in dataset_info:
        SESSIONS = dataset_info["SESSIONS"]
    else:
        SESSIONS = None
    if SESSIONS is None:
        SESSIONS = [None]

    if "RUNS" in dataset_info:
        RUNS = dataset_info["RUNS"]
    else:
        RUNS = None
    if RUNS is None:
        RUNS = {task: [None] for task in TASKS}

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

    trial_type_label = dataset_info["trial_type_label"]
    rest_labels = dataset_info["rest_labels"]

    for session in SESSIONS:
        for task in TASKS:
            run_roi_signal_extraction(
                subj=participant_id,
                task=task,
                main_root=main_root,
                fmriprep_root=fmriprep_root,
                bold_suffix=dataset_info["bold_suffix"],
                output_root=output_root,
                session=session,
                RUNS=RUNS[task],
                trial_type_label=trial_type_label,
                rest_labels=rest_labels,
            )

    print(
        f"subject-level ROI signal extraction CODE finished running ... for subject: {participant_id} ..."
    )

####################################################################
