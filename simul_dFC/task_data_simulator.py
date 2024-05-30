# -*- coding: utf-8 -*-
"""
Created on Wed March 20 2024

@author: mte
"""
import os
import warnings

import numpy as np
from tvb.simulator.lab import *

from pydfc import simul_utils

warnings.simplefilter("ignore")

os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"
################################# Parameters ####################################

# data paths
dataset = "ds000001"
# main_root = f"./DATA/{dataset}" # for local
main_root = f"../../DATA/task-based/simulated/{dataset}"  # for server
output_root = f"{main_root}/derivatives/ROI_timeseries"

# simulation parameters
sim_length = 250e3  # in m sec
onset_time = 20.0  # in seconds
BOLD_period = 500  # in m sec
TAVG_period = 1.0  # in m sec
global_conn_coupling_coef = 0.0126
conn_speed = 1.0
D = 0.001  # noise dispersion
dt = 0.5  # integration step
n_subj = 200  # number of subjects

# create a subject id list
subj_list = [f"sub-{i:04d}" for i in range(1, n_subj + 1)]

job_id = int(os.getenv("SGE_TASK_ID"))
subj_id = subj_list[job_id - 1]  # SGE_TASK_ID starts from 1 not 0

print(f"subject-level simulation started running ... for subject: {subj_id} ...")

all_task_info = {
    "task-lowFreqLongRest": {
        "task_name": "task-lowFreqLongRest",
        "onset_time": onset_time,
        "task_duration": 8.0,
        "task_block_duration": 20.0,
        "sim_length": sim_length,
        "BOLD_period": BOLD_period,
        "TAVG_period": TAVG_period,
        "global_conn_coupling_coef": global_conn_coupling_coef,
        "D": D,
        "conn_speed": conn_speed,
        "dt": dt,
    },
    "task-lowFreqShortRest": {
        "task_name": "task-lowFreqShortRest",
        "onset_time": onset_time,
        "task_duration": 12.0,
        "task_block_duration": 20.0,
        "sim_length": sim_length,
        "BOLD_period": BOLD_period,
        "TAVG_period": TAVG_period,
        "global_conn_coupling_coef": global_conn_coupling_coef,
        "D": D,
        "conn_speed": conn_speed,
        "dt": dt,
    },
    "task-highFreqLongRest": {
        "task_name": "task-highFreqLongRest",
        "onset_time": onset_time,
        "task_duration": 1.0,
        "task_block_duration": 5.0,
        "sim_length": sim_length,
        "BOLD_period": BOLD_period,
        "TAVG_period": TAVG_period,
        "global_conn_coupling_coef": global_conn_coupling_coef,
        "D": D,
        "conn_speed": conn_speed,
        "dt": dt,
    },
    "task-highFreqShortRest": {
        "task_name": "task-highFreqShortRest",
        "onset_time": onset_time,
        "task_duration": 4.0,
        "task_block_duration": 5.0,
        "sim_length": sim_length,
        "BOLD_period": BOLD_period,
        "TAVG_period": TAVG_period,
        "global_conn_coupling_coef": global_conn_coupling_coef,
        "D": D,
        "conn_speed": conn_speed,
        "dt": dt,
    },
    "task-lowFreqShortRestNoisy": {
        "task_name": "task-midFreqMidRestNoisy",
        "onset_time": onset_time,
        "task_duration": 12.0,
        "task_block_duration": 20.0,
        "sim_length": sim_length,
        "BOLD_period": BOLD_period,
        "TAVG_period": TAVG_period,
        "global_conn_coupling_coef": global_conn_coupling_coef,
        "D": D * 100,
        "conn_speed": conn_speed,
        "dt": dt,
    },
}

for task in all_task_info:

    time_series, task_data = simul_utils.simulate_task_data(subj_id, all_task_info[task])

    # save the time series and task data
    output_file_prefix = f"{subj_id}_{task}"
    if not os.path.exists(f"{output_root}/{subj_id}/"):
        os.makedirs(f"{output_root}/{subj_id}/")
    np.save(f"{output_root}/{subj_id}/{output_file_prefix}_time-series.npy", time_series)
    np.save(f"{output_root}/{subj_id}/{output_file_prefix}_task-data.npy", task_data)

print("****************** DONE ******************")
####################################################################################
