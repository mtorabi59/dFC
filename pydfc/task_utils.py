# -*- coding: utf-8 -*-
"""
Functions to facilitate task-based validation.

Created on Oct 25 2023
@author: Mohammad Torabi
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from nilearn import glm
from scipy import signal
from sklearn.mixture import GaussianMixture

from .dfc_utils import TR_intersection, rank_norm, visualize_conn_mat

################################# Preprocessing Functions ####################################


def events_time_to_labels(
    events,
    TR_mri,
    num_time_mri,
    event_types=None,
    oversampling=50,
    trial_type_label="trial_type",
    rest_labels=["rest", "Rest"],
    return_0_1=False,
):
    """
    event_types is a list of event types to be considered. If None, it will found based on events.
    Assigns the longest event in each TR to that TR (in the interval from last TR to current TR).
    It assumes that the first time point is TR0 which corresponds to [0 sec, TR sec] interval.
    oversampling: number of samples per TR_mri to improve the time resolution of tasks

    if trial_type_label is None, we use event type "unknown" as the trial type
    """

    # find which column is the "onset" in the first row
    onset_idx = np.where(events[0, :] == "onset")[0][0]
    duration_idx = np.where(events[0, :] == "duration")[0][0]
    if trial_type_label is not None:
        trial_type_idx = np.where(events[0, :] == trial_type_label)[0][0]

    assert (
        events[0, onset_idx] == "onset"
    ), "Something went wrong with the events file! The onset column was not found!"
    assert (
        events[0, duration_idx] == "duration"
    ), "Something went wrong with the events file! The duration column was not found!"
    if trial_type_label is not None:
        assert (
            events[0, trial_type_idx] == trial_type_label
        ), "Something went wrong with the events file! The trial_type column was not found!"

    if event_types is None:
        if trial_type_label is None:
            event_types = ["unknown"]
        else:
            event_types = list(np.unique(events[1:, trial_type_idx]))
            # remove all the rest labels
            for rest_label in rest_labels:
                if rest_label in event_types:
                    event_types.remove(rest_label)
        # add the rest label to the beginning for consistency
        event_types = ["rest"] + event_types

    Fs = float(1 / TR_mri) * oversampling
    num_time_task = int(num_time_mri * oversampling)
    event_labels = np.zeros((num_time_task, 1))
    for i in range(events.shape[0]):
        # skip the first row which is the header
        if i == 0:
            continue

        if trial_type_label is None:
            trial_type = "unknown"
        else:
            trial_type = events[i, trial_type_idx]

        if trial_type in event_types:
            # the only rest label that is left in event types is "rest" but we don't want to consider it
            if trial_type == "rest":
                continue
            start_time = float(events[i, onset_idx])
            end_time = float(events[i, onset_idx]) + float(events[i, duration_idx])
            start_timepoint = int(np.rint(start_time * Fs))
            end_timepoint = int(np.rint(end_time * Fs))
            event_labels[start_timepoint:end_timepoint] = event_types.index(trial_type)

    if return_0_1:
        event_labels = np.multiply(event_labels != 0, 1)

    return event_labels, Fs, event_types


################################# Visualization Functions ####################################


def plot_task_dFC(task_presence, dFC_lst, Fs_mri, TR_step=12):
    """
    task_presence: numpy array containing the task presence in the time points of the dFC data
    this function assumes that the task presence has the same Fs as the dFC data, i.e. MRI data
    and that the time points of the task presence are aligned with the time points of the dFC data
    """
    conn_mat_size = 20
    scale_task_plot = 20

    # plot task_data['event_labels']
    plt.figure(figsize=(50, 200))

    ax = plt.gca()

    time = np.arange(0, task_presence.shape[0]) / Fs_mri
    ax.plot(time, task_presence * scale_task_plot, linewidth=4)
    plt.xlabel("Time (s)")

    comman_TRs = TR_intersection(dFC_lst)
    TRs_dFC = comman_TRs[::TR_step]

    for dFC_id, dFC in enumerate(dFC_lst):
        dFC_mat = rank_norm(dFC.get_dFC_mat(), global_norm=True)
        TR_array = dFC.TR_array
        for i in range(0, len(TR_array), 1):

            C = dFC_mat[i, :, :]
            TR = TR_array[i]
            if not TR in TRs_dFC:
                continue
            visualize_conn_mat(
                C=C,
                axis=ax,
                title="",
                cmap="plasma",
                V_MIN=0,
                V_MAX=None,
                node_networks=None,
                title_fontsize=18,
                loc_x=[TR / Fs_mri - conn_mat_size / 2, TR / Fs_mri + conn_mat_size / 2],
                loc_y=[(1 + dFC_id) * conn_mat_size, (2 + dFC_id) * conn_mat_size],
            )

            x1, y1 = [TR / Fs_mri, TR / Fs_mri], [conn_mat_size, 0]
            ax.plot(x1, y1, color="k", linestyle="-", linewidth=2)

    plt.show()


################################# PCA Functions ####################################

# def BOLD


################################# Prediction Functions ####################################

from sklearn.linear_model import LinearRegression


def linear_reg(X, y):
    """
    X = (n_samples, n_features)
    y = (n_samples, n_targets)
    """
    reg = LinearRegression().fit(X, y)
    print(reg.score(X, y))
    return reg.predict(X)


################################# Validation Functions ####################################


def event_conv_hrf(event_signal, TR_mri, TR_task):
    time_length_HRF = 32.0  # in sec
    hrf_model = "spm"  # 'spm' or 'glover'

    TR_HRF = TR_task
    oversampling = (
        TR_mri / TR_HRF
    )  # more samples per TR than the func data to have a better HRF resolution,  same as for event_labels
    if hrf_model == "glover":
        HRF = glm.first_level.glover_hrf(
            tr=TR_mri, oversampling=oversampling, time_length=time_length_HRF, onset=0.0
        )
    elif hrf_model == "spm":
        HRF = glm.first_level.spm_hrf(
            tr=TR_mri, oversampling=oversampling, time_length=time_length_HRF, onset=0.0
        )

    events_hrf = signal.convolve(HRF, event_signal, mode="full")[: len(event_signal)]

    return events_hrf


def event_labels_conv_hrf(event_labels, TR_mri, TR_task, no_hrf=False):
    """
    event_labels: event labels including 0 and event ids at the time each event happens
    TR_mri: TR of MRI
    TR_task: TR of task
    assumes that 0 is the resting state

    return: event labels convolved with HRF for each event type
    the convolved event labels have the same length as the event_labels
    event type i convolved with HRF is in events_hrf[:, i-1]

    events_hrf[:, 0] is the resting state

    if no_hrf is True, the event labels are not convolved with HRF
    """

    event_labels = np.array(event_labels)
    L = event_labels.shape[0]
    event_ids = np.unique(event_labels)
    event_ids = event_ids.astype(int)
    events_hrf = np.zeros((L, len(event_ids)))  # 0 is the resting state
    for i, event_id in enumerate(event_ids):
        # 0 is not an event, is the resting state
        if event_id == 0:
            continue
        event_signal = np.zeros(L)
        event_signal[event_labels[:, 0] == event_id] = 1.0

        if no_hrf:
            events_hrf[:, i] = event_signal
        else:
            events_hrf[:, i] = event_conv_hrf(event_signal, TR_mri, TR_task)

    # the time points that are not in any event are considered as resting state
    events_hrf[np.sum(events_hrf[:, 1:], axis=1) == 0.0, 0] = 1.0

    # time_length_task = len(event_labels)*TR_task

    return events_hrf


def downsample_events_hrf(events_hrf, TR_mri, TR_task, method="uniform"):
    """
    method:
        uniform
        resample
        decimate
    no major difference was observed between these methods
    the shape of events_hrf is (num_time_task, num_event_types) or (num_time_task,)
    the shape of the downsampled events_hrf is (num_time_mri, num_event_types)
    """
    flag = False
    if len(events_hrf.shape) == 1:
        flag = True
        events_hrf = np.expand_dims(events_hrf, axis=1)
    events_hrf_ds = []
    for i in range(events_hrf.shape[1]):
        if method == "uniform":
            events_hrf_ds.append(events_hrf[:: int(TR_mri / TR_task), i])
        elif method == "resample":
            events_hrf_ds.append(
                signal.resample(
                    events_hrf[:, i], int(events_hrf.shape[0] * TR_task / TR_mri)
                )
            )
        elif method == "decimate":
            events_hrf_ds.append(signal.decimate(events_hrf[:, i], int(TR_mri / TR_task)))
    events_hrf_ds = np.array(events_hrf_ds).T

    if flag:
        events_hrf_ds = events_hrf_ds[:, 0]

    return events_hrf_ds


def shifted_binarizing(
    event_labels_all_task_hrf,
    task_presence_ratio=0.5,
    step=0.001,
):
    # find threshold such that the after binarization of event_labels_all_task_hrf,
    # the ratio of 1 to 0 is equal to task_presence_ratio
    for threshold in np.arange(0, np.max(event_labels_all_task_hrf), step):
        # binarize the event_labels_all_task_hrf
        event_labels_all_task_hrf_binarized = np.where(
            event_labels_all_task_hrf > threshold, 1, 0
        )
        # find the ratio of 1 to 0 in event_labels_all_task_hrf_binarized
        new_ratio = np.mean(event_labels_all_task_hrf_binarized)
        if new_ratio <= task_presence_ratio:
            break
    return threshold


def GMM_binarizing(
    event_labels_all_task_hrf,
    threshold=0.1,
    downsample=True,
    TR_mri=None,
    TR_task=None,
    TR_array=None,
):
    event_labels_all_task_hrf = event_labels_all_task_hrf.copy()
    event_labels_all_task_hrf_reshaped = event_labels_all_task_hrf.reshape(-1, 1)
    # Fit GMM
    gmm = GaussianMixture(n_components=2, n_init=5).fit(
        event_labels_all_task_hrf_reshaped
    )
    # downsample to MRI TR
    if downsample:
        event_labels_all_task_hrf_reshaped = downsample_events_hrf(
            event_labels_all_task_hrf_reshaped, TR_mri, TR_task
        )
    # some dFC measures (window-based) have a different TR than the task data
    if TR_array is not None:
        event_labels_all_task_hrf_reshaped = event_labels_all_task_hrf_reshaped[TR_array]
    # now predict on vs. off for the downsampled time points
    probs = gmm.predict_proba(event_labels_all_task_hrf_reshaped)
    # Identify which component corresponds to "on" (higher mean)
    # Each component has a mean, and in this case:
    # The "off" state should have a lower mean (closer to baseline).
    # The "on" state should have a higher mean (HRF-convolved signal is elevated during task).
    means = gmm.means_.flatten()
    on_component = np.argmax(means)
    # Get probability of being in the "on" state
    p_on = probs[:, on_component]
    # Create a binarized signal with transition points discarded
    indices = np.where((p_on <= threshold) | (p_on >= (1 - threshold)))[0]
    task_presence = np.where(p_on >= (1 - threshold), 1, 0)

    return task_presence, indices


def extract_task_presence(
    event_labels,
    TR_task,
    TR_mri,
    TR_array=None,
    binary=True,
    binarizing_method="GMM",
    no_hrf=False,
):
    """
    event_labels: event labels including 0 and event ids at the time each event happens
    TR_task: TR of task
    TR_array: the time points of the dFC data, optional
    TR_mri: TR of MRI

    This function extracts the task presence from the event labels and returns it in the same time points as the dFC data
    It also downsamples the task presence to the time points of the dFC data
    if binary is True, the task presence is binarized using the mean of the task presence
    binarizing_method: 'median' or 'mean' or 'shift' or 'GMM'
    if binarizing_method is 'shift', the task presence is binarized such that the ratio of 1 to 0 is equal to the task presence ratio

    if no_hrf is True, the task presence is not convolved with HRF
    """

    # event_labels_all_task is all conditions together, rest vs. task times
    event_labels_all_task = np.multiply(event_labels != 0, 1)

    event_labels_all_task_hrf = event_labels_conv_hrf(
        event_labels=event_labels_all_task, TR_mri=TR_mri, TR_task=TR_task, no_hrf=no_hrf
    )

    # keep the task signal of events_hrf_0_1_ds
    if event_labels_all_task_hrf.shape[1] == 1:
        # rest
        # raise error if no task
        raise ValueError("No task signal in the event data")
    else:
        # other tasks
        event_labels_all_task_hrf = event_labels_all_task_hrf[:, 1]

    if binary:
        if binarizing_method == "median":
            threshold = np.median(event_labels_all_task_hrf)
            task_presence = np.where(event_labels_all_task_hrf > threshold, 1, 0)
            task_presence = downsample_events_hrf(task_presence, TR_mri, TR_task)
            # some dFC measures (window-based) have a different TR than the task data
            if TR_array is not None:
                task_presence = task_presence[TR_array]
            indices = np.arange(task_presence.shape[0])
        elif binarizing_method == "mean":
            threshold = np.mean(event_labels_all_task_hrf)
            task_presence = np.where(event_labels_all_task_hrf > threshold, 1, 0)
            task_presence = downsample_events_hrf(task_presence, TR_mri, TR_task)
            # some dFC measures (window-based) have a different TR than the task data
            if TR_array is not None:
                task_presence = task_presence[TR_array]
            indices = np.arange(task_presence.shape[0])
        elif binarizing_method == "shift":
            task_presence_ratio = np.mean(event_labels_all_task)
            threshold = shifted_binarizing(
                event_labels_all_task_hrf=event_labels_all_task_hrf,
                task_presence_ratio=task_presence_ratio,
            )
            task_presence = np.where(event_labels_all_task_hrf > threshold, 1, 0)
            task_presence = downsample_events_hrf(task_presence, TR_mri, TR_task)
            # some dFC measures (window-based) have a different TR than the task data
            if TR_array is not None:
                task_presence = task_presence[TR_array]
            indices = np.arange(task_presence.shape[0])
        elif binarizing_method == "GMM":
            task_presence, indices = GMM_binarizing(
                event_labels_all_task_hrf,
                threshold=0.01,
                downsample=True,
                TR_mri=TR_mri,
                TR_task=TR_task,
                TR_array=TR_array,
            )
        else:
            raise ValueError(
                "binarizing_method should be 'median', 'mean', 'shift', or 'GMM'"
            )
    else:
        task_presence = event_labels_all_task_hrf
        task_presence = downsample_events_hrf(task_presence, TR_mri, TR_task)
        # some dFC measures (window-based) have a different TR than the task data
        if TR_array is not None:
            task_presence = task_presence[TR_array]
        indices = np.arange(task_presence.shape[0])

    return task_presence, indices


################################# Task Features ####################################


def calc_relative_task_on(task_presence):
    """
    task_presence: 0, 1 array
    return: relative_task_on
    """
    return np.sum(task_presence) / len(task_presence)


def calc_task_duration(task_presence, TR_mri):
    """
    task_presence: 0, 1 array
    return: avg_task_duration, var_task_duration
    """
    task_durations = list()
    for i in range(1, len(task_presence)):
        if task_presence[i] == 1 and task_presence[i - 1] == 0:
            start = i
        if task_presence[i] == 0 and task_presence[i - 1] == 1:
            end = i
            task_durations.append((end - start) * TR_mri)
            start = None
    task_durations = np.array(task_durations)
    # find mean and variance of task durations with division error handling
    if len(task_durations) == 0:
        return 0, 0
    return np.mean(task_durations), np.var(task_durations)


def calc_rest_duration(task_presence, TR_mri):
    """
    task_presence: 0, 1 array
    return: avg_rest_duration, var_rest_duration
    """
    rest_durations = list()
    if task_presence[0] == 0:
        start = 0
    for i in range(1, len(task_presence)):
        if task_presence[i] == 0 and task_presence[i - 1] == 1:
            start = i
        if task_presence[i] == 1 and task_presence[i - 1] == 0:
            end = i
            rest_durations.append((end - start) * TR_mri)
            start = None
    if task_presence[-1] == 0:
        end = len(task_presence)
        rest_durations.append((end - start) * TR_mri)
    rest_durations = np.array(rest_durations)
    # find mean and variance of rest durations with division error handling
    if len(rest_durations) == 0:
        return 0, 0
    return np.mean(rest_durations), np.var(rest_durations)


def calc_transition_freq(task_presence):
    """
    task_presence: 0, 1 array
    return: num_of_transitions, relative_transition_freq
    """
    transitions = np.abs(np.diff(task_presence))
    num_of_transitions = np.sum(transitions)
    relative_transition_freq = num_of_transitions / len(task_presence)
    return num_of_transitions, relative_transition_freq
