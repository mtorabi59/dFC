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

########################################################################################


def run_FCS_estimate(
    params_methods,
    MEASURES_name_lst,
    alter_hparams,
    params_multi_analysis,
    task,
    roi_root,
    output_root,
    session=None,
    run=None,
):
    if session is None:
        output_dir = f"{output_root}"
    else:
        output_dir = f"{output_root}/{session}"

    if run is None:
        print(f"TASK: {task} started ...")
        if session is None:
            BOLD_file_name = "{subj_id}_{task}_time-series.npy"
            file_suffix = f"{task}"
        else:
            BOLD_file_name = "{subj_id}_{session}_{task}_time-series.npy"
            file_suffix = f"{session}_{task}"
    else:
        print(f"TASK: {task}, RUN: {run} started ...")
        if session is None:
            BOLD_file_name = "{subj_id}_{task}_{run}_time-series.npy"
            file_suffix = f"{task}_{run}"
        else:
            BOLD_file_name = "{subj_id}_{session}_{task}_{run}_time-series.npy"
            file_suffix = f"{session}_{task}_{run}"
    ################################# LOAD DATA #################################
    BOLD = data_loader.load_TS(
        data_root=roi_root,
        file_name=BOLD_file_name,
        subj_id2load=None,
        task=task,
        session=session,
        run=run,
    )
    ################################ Measures of dFC #################################

    MA = MultiAnalysis(
        analysis_name=f"task-based-dFC-{file_suffix}", **params_multi_analysis
    )

    MEASURES_lst = MA.measures_initializer(
        MEASURES_name_lst, params_methods, alter_hparams
    )

    tic = time.time()
    print("Measurement Started ...")

    ################################# estimate FCS #################################

    for MEASURE_id, measure in enumerate(MEASURES_lst):

        print("MEASURE: " + measure.measure_name)
        print("FCS estimation started...")

        if measure.is_state_based:
            measure.estimate_FCS(time_series=BOLD)

        print("FCS estimation done.")

        # Save
        if not os.path.exists(f"{output_dir}"):
            os.makedirs(f"{output_dir}")
        np.save(f"{output_dir}/MEASURE_{file_suffix}_{MEASURE_id}.npy", measure)

    print(f"Measurement required {time.time() - tic:0.3f} seconds.")
    np.save(f"{output_dir}/multi-analysis_{file_suffix}.npy", MA)


########################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to fit dFC methods for a given task.
    """

    parser = argparse.ArgumentParser(description=HELPTEXT)

    parser.add_argument("--dataset_info", type=str, help="path to dataset info file")
    parser.add_argument("--methods_config", type=str, help="methods config file")

    args = parser.parse_args()

    dataset_info_file = args.dataset_info
    methods_config_file = args.methods_config

    # Read dataset info
    with open(dataset_info_file, "r") as f:
        dataset_info = json.load(f)

    # Read methods config
    with open(methods_config_file, "r") as f:
        methods_config = json.load(f)

    TASKS = dataset_info["TASKS"]

    job_id = int(os.getenv("SGE_TASK_ID"))
    TASK_id = job_id - 1  # SGE_TASK_ID starts from 1 not 0
    if TASK_id >= len(TASKS):
        print("TASK_id out of TASKS")
        exit()
    task = TASKS[TASK_id]

    print(f"FCS estimation CODE started running ... for task: {task} ...")

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
        RUNS = {task: [None]}

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

    # methods params
    params_methods = methods_config["params_methods"]
    MEASURES_name_lst = methods_config["MEASURES_name_lst"]
    alter_hparams = methods_config["alter_hparams"]
    params_multi_analysis = methods_config["params_multi_analysis"]

    for session in SESSIONS:
        for run in RUNS[task]:
            run_FCS_estimate(
                params_methods=params_methods,
                MEASURES_name_lst=MEASURES_name_lst,
                alter_hparams=alter_hparams,
                params_multi_analysis=params_multi_analysis,
                task=task,
                roi_root=roi_root,
                output_root=fitted_measures_root,
                session=session,
                run=run,
            )

    print(f"FCS estimation CODE finished running ... for task: {task} ...")
#################################################################################
