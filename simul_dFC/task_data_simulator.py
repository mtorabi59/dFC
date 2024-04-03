# -*- coding: utf-8 -*-
"""
Created on Wed March 20 2024

@author: mte
"""
import os
import warnings

import numpy as np
from tvb.simulator.lab import *

from pydfc import TIME_SERIES, task_utils

warnings.simplefilter("ignore")

os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"
################################# Parameters ####################################

# data paths
dataset = "ds000002"
# main_root = f"./DATA/{dataset}" # for local
main_root = f"../../DATA/task-based/simulated/{dataset}"  # for server
output_root = f"{main_root}/derivatives/ROI_timeseries"

task = "task-pulse"

# simulation parameters
sim_length = 250e3  # in m sec
onset_time = 20.0  # in seconds
task_duration = 12.0  # in seconds
task_block_duration = 30.0  # in seconds
BOLD_period = 500  # in m sec
TAVG_period = 1.0  # in m sec
conn_speed = 1.0
D = 0.001  # noise dispersion
dt = 0.5  # integration step
n_subj = 200  # number of subjects

# create a subject id list
subj_list = [f"sub-{i:04d}" for i in range(1, n_subj + 1)]

job_id = int(os.getenv("SGE_TASK_ID"))
subj_id = subj_list[job_id - 1]  # SGE_TASK_ID starts from 1 not 0

print(f"subject-level simulation started running ... for subject: {subj_id} ...")

# randomize some parameters for each subjects
onset = np.random.normal(loc=onset_time, scale=0.5)  # seconds
global_conn_coupling = np.random.normal(loc=0.0126, scale=0.0075)
rand_weighting = np.array(
    [
        np.random.normal(loc=2.0**-2, scale=0.1 * (2.0**-2)),
        np.random.normal(loc=2.0**-3, scale=0.1 * (2.0**-3)),
        np.random.normal(loc=2.0**-4, scale=0.1 * (2.0**-4)),
        np.random.normal(loc=2.0**-5, scale=0.1 * (2.0**-5)),
        np.random.normal(loc=2.0**-6, scale=0.1 * (2.0**-6)),
    ]
)
conn_speed_rand = np.random.normal(loc=conn_speed, scale=0.1 * conn_speed)
################################# Initialize Simulation ####################################
conn = connectivity.Connectivity.from_file()
conn.speed = np.array([conn_speed_rand])

# configure stimulus spatial pattern
weighting = np.zeros((76,))
weighting[[0, 7, 13, 33, 42]] = rand_weighting
# weighting[[0, 7, 13, 33, 42]] = numpy.array([2.0 ** -2, 2.0 ** -3, 2.0 ** -4, 2.0 ** -5, 2.0 ** -6])

# temporal profile
eqn_t = equations.PulseTrain()
eqn_t.parameters["onset"] = onset * 1e3  # ms
eqn_t.parameters["tau"] = task_duration * 1e3  # ms
eqn_t.parameters["T"] = task_block_duration * 1e3  # ms

stimulus = patterns.StimuliRegion(temporal=eqn_t, connectivity=conn, weight=weighting)

################################# Run Simulation ####################################

# set the global coupling strength
# you can switch between deterministic (without noise) and stochastic integration (with noise)
sim = simulator.Simulator(
    model=models.Generic2dOscillator(a=np.array([0.5])),
    connectivity=conn,
    coupling=coupling.Linear(a=np.array([global_conn_coupling])),
    # integrator=integrators.HeunDeterministic(dt=dt),
    integrator=integrators.HeunStochastic(
        dt=dt, noise=noise.Additive(nsig=np.array([D]))
    ),
    monitors=(
        monitors.TemporalAverage(period=TAVG_period),
        monitors.Bold(period=BOLD_period, hrf_kernel=equations.MixtureOfGammas()),
        monitors.ProgressLogger(period=10e3),
    ),
    stimulus=stimulus,
    simulation_length=sim_length,
).configure()

(tavg_time, tavg_data), (bold_time, bold_data), _ = sim.run()

# # truncate the first 10 seconds of the simulation
# # to avoid transient effects
# truncate_time = 10e3 # in m sec
# bold_truncate_idx = int(truncate_time / BOLD_period)
# bold_time = bold_time[bold_truncate_idx:]
# bold_data = bold_data[bold_truncate_idx:]
# tavg_truncate_idx = int(truncate_time / TAVG_period)
# tavg_time = tavg_time[tavg_truncate_idx:]
# tavg_data = tavg_data[tavg_truncate_idx:]

centres_locs = conn.centres
region_labels = list(conn.region_labels)
TR_mri = BOLD_period * 1e-3  # in seconds

bold_data = bold_data[:, 0, :, 0]
# change time_series.shape to (roi, time)
bold_data = bold_data.T

time_series = TIME_SERIES(
    data=bold_data,
    subj_id=subj_id,
    Fs=1 / TR_mri,
    locs=centres_locs,
    node_labels=region_labels,
    TS_name=f"BOLD_{subj_id}_{task}",
    session_name=task,
)
num_time_mri = time_series.n_time
################################# EXTRACT TASK LABELS #########################
oversampling = 50  # more samples per TR than the func data to have a better event_labels time resolution

events = []
event_types = ["rest", "task"]
TASKS = [task]

# using onset, task_duration, task_block_duration to create the events
events.append(["onset", "duration", "trial_type"])
t = onset
while t < sim_length:
    events.append([t, task_duration, "task"])
    t += task_block_duration
events = np.array(events)

event_labels, Fs_task = task_utils.events_time_to_labels(
    events=events,
    TR_mri=TR_mri,
    num_time_mri=num_time_mri,
    event_types=event_types,
    oversampling=oversampling,
    return_0_1=False,
)
# fill task labels with 0 (rest) and 1 (task's index, here only 1 task is used)
task_labels = np.multiply(event_labels != 0, 1)
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
subj_folder = f"{subj_id}_{task}"
if not os.path.exists(f"{output_root}/{subj_folder}/"):
    os.makedirs(f"{output_root}/{subj_folder}/")
np.save(f"{output_root}/{subj_folder}/time_series.npy", time_series)
np.save(f"{output_root}/{subj_folder}/task_data.npy", task_data)

print("****************** DONE ******************")
####################################################################################
