import os
import time
import warnings

import numpy as np

from pydfc import MultiAnalysis, data_loader

warnings.simplefilter("ignore")

os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"

################################# Parameters #################################
# data paths
dataset = "ds003242"
# main_root = f"../../DATA/{dataset}" # for local
main_root = f"/data/origami/dFC/DATA/task-based/openneuro/{dataset}"  # for server
roi_root = f"{main_root}/derivatives/ROI_timeseries"
output_root = f"{main_root}/derivatives/fitted_MEASURES"

# for consistency we use 0 for resting state
TASKS = ["task-CIC", "task-midloc"]

# default RUNS = None
RUNS = None
RUNS = {
    "task-CIC": ["run-001", "run-002", "run-003", "run-004", "run-005", "run-006"],
    "task-midloc": ["run-001"],
}

job_id = int(os.getenv("SGE_TASK_ID"))
TASK_id = job_id - 1  # SGE_TASK_ID starts from 1 not 0
if TASK_id >= len(TASKS):
    print("TASK_id out of TASKS")
    exit()
task = TASKS[TASK_id]

###### MEASUREMENT PARAMETERS ######

# W is in sec

params_methods = {
    # Sliding Parameters
    "W": 12,
    "n_overlap": 1.0,
    "sw_method": "pear_corr",
    "tapered_window": True,
    # TIME_FREQ
    "TF_method": "WTC",
    # CLUSTERING AND DHMM
    "clstr_base_measure": "SlidingWindow",
    # HMM
    "hmm_iter": 20,
    "dhmm_obs_state_ratio": 16 / 24,
    # State Parameters
    "n_states": 5,
    "n_subj_clstrs": 10,
    # Parallelization Parameters
    "n_jobs": 2,
    "verbose": 0,
    "backend": "loky",
    # SESSION
    "session": task,
    # Hyper Parameters
    "normalization": True,
    "num_subj": None,
    "num_time_point": None,
}

###### HYPER PARAMETERS ALTERNATIVE ######

MEASURES_name_lst = [
    "SlidingWindow",
    "Time-Freq",
    "CAP",
    "ContinuousHMM",
    "Windowless",
    "Clustering",
    "DiscreteHMM",
]

alter_hparams = {
    # 'session': ['Rest1_RL', 'Rest2_LR', 'Rest2_RL'],
    # 'n_overlap': [0, 0.25, 0.75, 1],
    # 'n_states': [6, 16],
    # # 'normalization': [],
    # 'num_subj': [50, 100, 200],
    # 'num_select_nodes': [30, 50, 333],
    # 'num_time_point': [800, 1000],
    # 'Fs_ratio': [0.50, 0.75, 1.5],
    # 'noise_ratio': [1.00, 2.00, 3.00],
    # 'num_realization': []
}

###### MultiAnalysis PARAMETERS ######

params_multi_analysis = {
    # Parallelization Parameters
    "n_jobs": None,
    "verbose": 0,
    "backend": "loky",
}

if RUNS is None:
    RUNS = {task: [None]}
for run in RUNS[task]:
    if run is None:
        print(f"TASK: {task} started ...")
        file_suffix = f"{task}"
        BOLD_file_name = "{subj_id}_{task}_time-series.npy"
    else:
        print(f"TASK: {task}, RUN: {run} started ...")
        file_suffix = f"{task}_{run}"
        BOLD_file_name = "{subj_id}_{task}_{run}_time-series.npy"
    ################################# LOAD DATA #################################
    BOLD = data_loader.load_TS(
        data_root=roi_root,
        file_name=BOLD_file_name,
        SESSIONs=task,
        subj_id2load=None,
        task=task,
        run=run,
    )
    ################################ Measures of dFC #################################

    MA = MultiAnalysis(
        analysis_name=f"task-based-dFC-{dataset}-{file_suffix}", **params_multi_analysis
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
        if not os.path.exists(f"{output_root}"):
            os.makedirs(f"{output_root}")
        np.save(f"{output_root}/MEASURE_{file_suffix}_{MEASURE_id}.npy", measure)

    print(f"Measurement required {time.time() - tic:0.3f} seconds.")
    np.save(f"{output_root}/multi-analysis_{file_suffix}.npy", MA)

#################################################################################
