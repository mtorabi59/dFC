import argparse
import json
import os
import time
import warnings

import numpy as np

from pydfc import MultiAnalysis, data_loader

warnings.simplefilter("ignore")

os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"

################################# Functions #################################


def run_dFC_assess(
    subj_id,
    task,
    roi_root,
    fitted_measures_root,
    output_root,
):

    # check if the subject has this task in roi_root
    if not os.path.exists(f"{roi_root}/{subj_id}"):
        print(f"Subject {subj_id} not found in {roi_root}")
        return

    ALL_ROI_FILES = os.listdir(f"{roi_root}/{subj_id}/")
    ALL_ROI_FILES = [
        roi_file
        for roi_file in ALL_ROI_FILES
        if ("_time-series.npy" in roi_file) and (task in roi_file)
    ]
    ALL_ROI_FILES.sort()

    # check if "_run" exists in all the task file names
    if all(["_run" in roi_file for roi_file in ALL_ROI_FILES]):
        # find all the runs
        RUNS = [
            roi_file[
                roi_file.find("_run")
                + 1 : roi_file.find("_run")
                + 1
                + roi_file[roi_file.find("_run") + 1 :].find("_")
            ]
            for roi_file in ALL_ROI_FILES
        ]
        # sort
        RUNS.sort()
        print(f"Found multiple runs for {subj_id} {task}: {RUNS}")
    else:
        RUNS = [None]

    for run in RUNS:

        # check if the subject has this task and run in roi_root
        if run is None:
            file_suffix = f"{task}"
            if not os.path.exists(
                f"{roi_root}/{subj_id}/{subj_id}_{file_suffix}_time-series.npy"
            ):
                print(f"Time series file not found for {subj_id} {task}")
                continue
            else:
                print(
                    f"subject-level dFC assessment CODE started running ... for task {task} of subject {subj_id} ..."
                )
                BOLD_file_name = "{subj_id}_{task}_time-series.npy"
        else:
            file_suffix = f"{task}_{run}"
            if not os.path.exists(
                f"{roi_root}/{subj_id}/{subj_id}_{file_suffix}_time-series.npy"
            ):
                print(f"Time series file not found for {subj_id} {task} {run}")
                continue
            else:
                print(
                    f"subject-level dFC assessment CODE started running ... for task {task} and {run} of subject {subj_id} ..."
                )
                BOLD_file_name = "{subj_id}_{task}_{run}_time-series.npy"

        ################################# LOAD FIT MEASURES #################################

        MA = np.load(
            f"{fitted_measures_root}/multi-analysis_{file_suffix}.npy",
            allow_pickle="TRUE",
        ).item()

        ALL_RECORDS = os.listdir(f"{fitted_measures_root}/")
        ALL_RECORDS = [i for i in ALL_RECORDS if ("MEASURE" in i) and (file_suffix in i)]
        ALL_RECORDS.sort()
        MEASURES_fit_lst = list()
        for s in ALL_RECORDS:
            fit_measure = np.load(
                f"{fitted_measures_root}/{s}", allow_pickle="TRUE"
            ).item()
            MEASURES_fit_lst.append(fit_measure)
        MA.set_MEASURES_fit_lst(MEASURES_fit_lst)
        print("fitted MEASURES are loaded ...")

        ################################# LOAD DATA #################################

        BOLD = data_loader.load_TS(
            data_root=roi_root,
            file_name=BOLD_file_name,
            SESSIONs=task,
            subj_id2load=subj_id,
            task=task,
            run=run,
        )

        ################################# dFC ASSESSMENT #################################

        tic = time.time()
        print("Measurement Started ...")

        print("dFC estimation started...")
        dFC_dict = MA.subj_lvl_dFC_assess(time_series=BOLD)
        print("dFC estimation done.")

        print(f"Measurement required {time.time() - tic:0.3f} seconds.")

        ################################# SAVE DATA #################################

        folder = f"{output_root}/{subj_id}"
        if not os.path.exists(folder):
            os.makedirs(folder)

        for dFC_id, dFC in enumerate(dFC_dict["dFC_lst"]):
            np.save(f"{folder}/dFC_{file_suffix}_{dFC_id}.npy", dFC)


#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to assess dFC for a given participant.
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
        f"subject-level dFC assessment CODE started running ... for subject: {participant_id} ..."
    )

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

    if "{main_root}" in dataset_info["fitted_measures_root"]:
        fitted_measures_root = dataset_info["fitted_measures_root"].replace(
            "{main_root}", main_root
        )
    else:
        fitted_measures_root = dataset_info["fitted_measures_root"]

    if "{main_root}" in dataset_info["dFC_root"]:
        output_root = dataset_info["dFC_root"].replace("{main_root}", main_root)
    else:
        output_root = dataset_info["dFC_root"]

    for task in TASKS:
        run_dFC_assess(
            subj_id=participant_id,
            task=task,
            roi_root=roi_root,
            fitted_measures_root=fitted_measures_root,
            output_root=output_root,
        )

    print(
        f"subject-level dFC assessment CODE finished running ... for subject: {participant_id} ..."
    )

#######################################################################################
