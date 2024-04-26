# -*- coding: utf-8 -*-
"""
Functions to facilitate dFC simulation.

Created on April 25 2024
@author: Mohammad Torabi
"""

import numpy as np
from tvb.simulator.lab import *

from pydfc import TIME_SERIES, task_utils

################################# Simulation Functions ####################################


def create_random_stimulus_weights(stimulated_regions_list, n_regions=76):
    """
    Create random stimulus weights for the stimulated regions.
    """
    rand_weighting = [
        np.random.normal(loc=2.0 ** (-1 * (2 + i)), scale=0.1 * (2.0**-2))
        for i in range(len(stimulated_regions_list))
    ]

    # configure stimulus spatial pattern
    weighting = np.zeros((n_regions,))
    weighting[stimulated_regions_list] = rand_weighting

    return weighting


def create_stimulus(
    onset,
    task_duration,
    task_block_duration,
    conn,
    region_weighting,
):
    """
    Create a stimulus pattern for the task.
    """
    # temporal profile
    eqn_t = equations.PulseTrain()
    eqn_t.parameters["onset"] = onset * 1e3  # ms
    eqn_t.parameters["tau"] = task_duration * 1e3  # ms
    eqn_t.parameters["T"] = task_block_duration * 1e3  # ms

    stimulus = patterns.StimuliRegion(
        temporal=eqn_t, connectivity=conn, weight=region_weighting
    )

    return stimulus


def simulate_task_BOLD(
    onset_time,
    task_duration,
    task_block_duration,
    sim_length,
    BOLD_period,
    TAVG_period,
    global_conn_coupling_coef=0.0126,
    D=0.001,
    conn_speed=1.0,
    dt=0.5,
    drop_initial_time=False,
):
    """
    Simulate BOLD signal for a task.
    """
    # randomize some parameters for each subjects
    onset = np.random.normal(loc=onset_time, scale=0.5)  # seconds
    global_conn_coupling = np.random.normal(loc=global_conn_coupling_coef, scale=0.0075)
    conn_speed_rand = np.random.normal(loc=conn_speed, scale=0.1 * conn_speed)
    ################################# Initialize Simulation ####################################
    conn = connectivity.Connectivity.from_file()
    conn.speed = np.array([conn_speed_rand])

    # configure stimulus spatial pattern
    weighting = create_random_stimulus_weights(
        stimulated_regions_list=[0, 7, 13, 33, 42], n_regions=76
    )

    stimulus = create_stimulus(
        onset=onset,
        task_duration=task_duration,
        task_block_duration=task_block_duration,
        conn=conn,
        region_weighting=weighting,
    )

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

    if drop_initial_time:
        # truncate the first 10 seconds of the simulation
        # to avoid transient effects
        truncate_time = 10e3  # in m sec
        bold_truncate_idx = int(truncate_time / BOLD_period)
        bold_time = bold_time[bold_truncate_idx:]
        bold_data = bold_data[bold_truncate_idx:]
        tavg_truncate_idx = int(truncate_time / TAVG_period)
        tavg_time = tavg_time[tavg_truncate_idx:]
        tavg_data = tavg_data[tavg_truncate_idx:]

    centres_locs = conn.centres
    region_labels = list(conn.region_labels)
    TR_mri = BOLD_period * 1e-3  # in seconds

    bold_data = bold_data[:, 0, :, 0]
    # change time_series.shape to (roi, time)
    bold_data = bold_data.T

    TAVG_data = tavg_data[:, 0, :, 0]
    # change time_series.shape to (roi, time)
    TAVG_data = TAVG_data.T

    return (
        bold_data,
        bold_time,
        region_labels,
        centres_locs,
        TR_mri,
        TAVG_data,
        tavg_time,
        TAVG_period,
    )


def create_simul_task_info(
    num_time_mri,
    TR_mri,
    task,
    onset,
    task_duration,
    task_block_duration,
    sim_length,
    oversampling=50,
):
    """
    Create a dictionary containing the task data for simulation.

    Parameters
    ----------
    num_time_mri : int
        Number of time points in the BOLD signal.
    TR_mri : float
        The repetition time of the MRI.
    task : str
        The task name.
    onset : float
        The onset time of the task.
    task_duration : float
        The duration of the task.
    task_block_duration : float
        The duration of the task block.
    sim_length : float
        The length of the simulation.
    oversampling : int, optional
        The oversampling factor. The default is 50.
        generate more samples per TR than the func data to have a
        better event_labels time resolution
    """
    ################################# EXTRACT TASK LABELS #########################
    events = []
    event_types = ["rest", "task"]

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
        "event_labels": event_labels,
        "event_types": event_types,
        "events": events,
        "Fs_task": Fs_task,
        "TR_mri": TR_mri,
        "num_time_mri": num_time_mri,
    }

    return task_data


def simulate_task_BOLD_TS(
    subj_id,
    task,
    onset_time,
    task_duration,
    task_block_duration,
    sim_length,
    BOLD_period,
    TAVG_period,
    global_conn_coupling_coef=0.0126,
    D=0.001,
    conn_speed=1.0,
    dt=0.5,
    drop_initial_time=False,
):
    """
    Simulate BOLD signal for a task and return a TIME_SERIES object.
    """
    bold_data, bold_time, region_labels, centres_locs, TR_mri, _, _, _ = (
        simulate_task_BOLD(
            onset_time=onset_time,
            task_duration=task_duration,
            task_block_duration=task_block_duration,
            sim_length=sim_length,
            BOLD_period=BOLD_period,
            TAVG_period=TAVG_period,
            global_conn_coupling_coef=global_conn_coupling_coef,
            D=D,
            conn_speed=conn_speed,
            dt=dt,
            drop_initial_time=drop_initial_time,
        )
    )
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
    task_data = create_simul_task_info(
        num_time_mri=num_time_mri,
        TR_mri=TR_mri,
        task=task,
        onset=onset_time,
        task_duration=task_duration,
        task_block_duration=task_block_duration,
        sim_length=sim_length,
    )

    return time_series, task_data


def simulate_task(subj_id, task_info):
    """
    Simulate task-based BOLD signal for a subject.

    Parameters
    ----------
    subj_id : int
        The subject ID.
    task_info : dict
        A dictionary containing the task information below:
            - task_name: str
                The name of the task.
            - onset_time: float
                The onset time of the task.
            - task_duration: float
                The duration of the task.
            - task_block_duration: float
                The duration of the task block.
            - sim_length: float
                The length of the simulation.
            - BOLD_period: float
                The BOLD period.
            - TAVG_period: float
                The TAVG period.
            - global_conn_coupling_coef: float
                The global connectivity coupling coefficient.
            - D: float
                The noise parameter.
            - conn_speed: float
                The connectivity speed.
            - dt: float
                The simulation time step.
    """
    time_series, task_data = simulate_task(
        subj_id=subj_id,
        task=task_info["task_name"],
        onset_time=task_info["onset_time"],
        task_duration=task_info["task_duration"],
        task_block_duration=task_info["task_block_duration"],
        sim_length=task_info["sim_length"],
        BOLD_period=task_info["BOLD_period"],
        TAVG_period=task_info["TAVG_period"],
        global_conn_coupling_coef=task_info["global_conn_coupling_coef"],
        D=task_info["D"],
        conn_speed=task_info["conn_speed"],
        dt=task_info["dt"],
    )

    return time_series, task_data
