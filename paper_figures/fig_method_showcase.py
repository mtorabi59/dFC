#!/usr/bin/env python3
"""
Figure A: Multi-Method dFC Showcase Panel
==========================================
Runs all 7 dFC methods on demo resting-state fMRI data (OpenNeuro AOMIC-PIOP1,
2 subjects) and displays a grid of connectivity matrices (methods × time windows).
Row labels are color-coded by method family.

Usage (from repo root):
    python paper_figures/fig_method_showcase.py

Output:
    paper_figures/output/fig_method_showcase.png
"""

import os
import subprocess
import sys
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "sample_data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Make pydfc importable when running from paper_figures/ directly
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Download demo data ─────────────────────────────────────────────────────────
BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep"
PREPROC_SUFFIX = "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
CONFOUND_SUFFIX = "desc-confounds_regressors.tsv"

DOWNLOAD_INFO = {
    "sub-0001": {
        "nifti_url": (
            f"{BASE_URL}/sub-0001/func/sub-0001_task-restingstate_acq-mb3_{PREPROC_SUFFIX}"
            "?versionId=UfCs4xtwIEPDgmb32qFbtMokl_jxLUKr"
        ),
        "confounds_url": (
            f"{BASE_URL}/sub-0001/func/sub-0001_task-restingstate_acq-mb3_{CONFOUND_SUFFIX}"
            "?versionId=biaIJGNQ22P1l1xEsajVzUW6cnu1_8lD"
        ),
    },
    "sub-0002": {
        "nifti_url": (
            f"{BASE_URL}/sub-0002/func/sub-0002_task-restingstate_acq-mb3_{PREPROC_SUFFIX}"
            "?versionId=fUBWmUTg6vfe2n.ywDNms4mOAW3r6E9Y"
        ),
        "confounds_url": (
            f"{BASE_URL}/sub-0002/func/sub-0002_task-restingstate_acq-mb3_{CONFOUND_SUFFIX}"
            "?versionId=2zWQIugU.J6ilTFObWGznJdSABbaTx9F"
        ),
    },
}


def download_demo_data():
    for subj, info in DOWNLOAD_INFO.items():
        for kind, url in [
            ("nifti", info["nifti_url"]),
            ("confounds", info["confounds_url"]),
        ]:
            suffix = PREPROC_SUFFIX if kind == "nifti" else CONFOUND_SUFFIX
            dest = os.path.join(DATA_DIR, f"{subj}_task-restingstate_acq-mb3_{suffix}")
            if not os.path.exists(dest):
                print(f"Downloading {subj} {kind} ...")
                subprocess.run(f"curl -L '{url}' -o '{dest}'", shell=True, check=False)
            else:
                print(f"  Found: {os.path.basename(dest)}")


# ── Parameters ─────────────────────────────────────────────────────────────────
PARAMS = {
    "W": 44,
    "n_overlap": 0.5,
    "sw_method": "pear_corr",
    "tapered_window": True,
    "TF_method": "WTC",
    "clstr_base_measure": "SlidingWindow",
    "hmm_iter": 30,
    "dhmm_obs_state_ratio": 16 / 24,
    "n_states": 12,
    "n_subj_clstrs": 13,
    "n_jobs": None,
    "verbose": 0,
    "backend": None,
    "session": "rest",
    "normalization": True,
}

# Ordered display list (state-free first, then state-based-smooth, then instantaneous)
METHODS = [
    "SlidingWindow",
    "Time-Freq",
    "Clustering",
    "ContinuousHMM",
    "DiscreteHMM",
    "CAP",
    "Windowless",
]

METHOD_DISPLAY = {
    "SlidingWindow": "Sliding Window (SW)",
    "Time-Freq": "Time-Frequency (TF)",
    "Clustering": "SW + Clustering (SWC)",
    "ContinuousHMM": "Continuous HMM (CHMM)",
    "DiscreteHMM": "Discrete HMM (DHMM)",
    "CAP": "Co-Activation Patterns (CAP)",
    "Windowless": "Windowless (WL)",
}

# 0 = state-free, 1 = state-based temporal, 2 = state-based instantaneous
METHOD_FAMILY = {
    "SlidingWindow": 0,
    "Time-Freq": 0,
    "Clustering": 1,
    "ContinuousHMM": 1,
    "DiscreteHMM": 1,
    "CAP": 2,
    "Windowless": 2,
}

FAMILY_COLORS = {
    0: "#4C72B0",  # blue  — state-free
    1: "#DD8452",  # orange — state-based (temporal)
    2: "#55A868",  # green  — state-based (instantaneous)
}

FAMILY_LABELS = {
    0: "State-Free",
    1: "State-Based (temporal)",
    2: "State-Based (instantaneous)",
}

N_WINDOWS = 5  # number of time windows to display per method


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Figure A: Multi-Method dFC Showcase")
    print("=" * 60)

    download_demo_data()

    from pydfc import data_loader, multi_analysis_utils
    from pydfc.dfc_utils import TR_intersection

    # ── Load data ──────────────────────────────────────────────────────────────
    subj_id_list = ["sub-0001", "sub-0002"]
    nifti_files_list = [
        os.path.join(
            DATA_DIR,
            f"{sid}_task-restingstate_acq-mb3_{PREPROC_SUFFIX}",
        )
        for sid in subj_id_list
    ]
    print("\nLoading time series...")
    BOLD_multi = data_loader.multi_nifti2timeseries(
        nifti_files_list,
        subj_id_list,
        n_rois=100,
        Fs=1 / 0.75,
        confound_strategy="no_motion",
        standardize=False,
    )

    # ── Initialize methods ─────────────────────────────────────────────────────
    print("Initializing methods...")
    MEASURES_lst, hyper_param_info = multi_analysis_utils.measures_initializer(
        METHODS, PARAMS, alter_hparams={}
    )

    # ── Fit group-level states ─────────────────────────────────────────────────
    print("Fitting group-level states (this may take a few minutes)...")
    MEASURES_fit_lst = multi_analysis_utils.estimate_group_FCS(
        time_series=BOLD_multi,
        MEASURES_lst=MEASURES_lst,
        n_jobs=None,
        verbose=0,
        backend=None,
    )

    # ── Estimate dFC for sub-0001 ──────────────────────────────────────────────
    print("Estimating dFC for sub-0001...")
    BOLD_subj = BOLD_multi.get_subj_ts(subjs_id="sub-0001")
    dFC_dict = multi_analysis_utils.subj_lvl_dFC_assess(
        time_series=BOLD_subj,
        MEASURES_fit_lst=MEASURES_fit_lst,
        n_jobs=None,
        verbose=0,
        backend=None,
    )
    dFC_lst = dFC_dict["dFC_lst"]

    # ── Pick time windows ──────────────────────────────────────────────────────
    common_TRs = TR_intersection(dFC_lst)
    step = max(1, len(common_TRs) // (N_WINDOWS + 1))
    chosen_TRs = common_TRs[step::step][:N_WINDOWS]
    print(f"Using {len(chosen_TRs)} time windows from {len(common_TRs)} common TRs.")

    # ── Collect dFC matrices ───────────────────────────────────────────────────
    dFC_matrices = {}
    for dfc in dFC_lst:
        name = dfc.measure.measure_name
        if name in METHODS:
            mats = dfc.get_dFC_mat(TRs=chosen_TRs)  # (N_WINDOWS, n_regions, n_regions)
            dFC_matrices[name] = mats

    missing = [m for m in METHODS if m not in dFC_matrices]
    if missing:
        print(f"Warning: missing outputs for methods: {missing}")

    # ── Build figure ───────────────────────────────────────────────────────────
    print("Building figure...")
    n_rows = len(METHODS)
    n_cols = N_WINDOWS

    # Label column + matrix columns
    label_col_ratio = 2.8
    mat_col_w = 1.6  # inches
    row_h = 1.6  # inches
    top_pad = 0.55  # inches for family legend
    bottom_pad = 0.2
    right_pad = 0.15

    fig_w = label_col_ratio * mat_col_w + n_cols * mat_col_w + right_pad
    fig_h = n_rows * row_h + top_pad + bottom_pad

    fig, all_axes = plt.subplots(
        n_rows,
        n_cols + 1,
        figsize=(fig_w, fig_h),
        facecolor="white",
        gridspec_kw={
            "width_ratios": [label_col_ratio] + [1] * n_cols,
            "wspace": 0.06,
            "hspace": 0.10,
            "left": 0.01,
            "right": 0.99,
            "bottom": bottom_pad / fig_h,
            "top": 1.0 - top_pad / fig_h,
        },
    )

    for r, method_key in enumerate(METHODS):
        # ── Label cell (column 0) ──────────────────────────────────────────────
        ax_label = all_axes[r, 0]
        ax_label.set_axis_off()
        family = METHOD_FAMILY[method_key]
        color = FAMILY_COLORS[family]

        # Colored rectangle as family indicator
        rect = mpatches.FancyBboxPatch(
            (0.02, 0.15),
            0.04,
            0.70,
            boxstyle="round,pad=0.01",
            facecolor=color,
            edgecolor="none",
            transform=ax_label.transAxes,
            clip_on=False,
        )
        ax_label.add_patch(rect)

        # Method name label (right-aligned, colored)
        ax_label.text(
            0.92,
            0.5,
            METHOD_DISPLAY.get(method_key, method_key),
            ha="right",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=color,
            transform=ax_label.transAxes,
        )

        # ── Matrix cells ───────────────────────────────────────────────────────
        mats = dFC_matrices.get(method_key)
        if mats is None:
            for c in range(n_cols):
                all_axes[r, c + 1].set_axis_off()
            continue

        v_max = float(np.percentile(np.abs(mats), 97))
        v_max = max(v_max, 1e-6)

        for c in range(n_cols):
            ax = all_axes[r, c + 1]
            ax.imshow(
                mats[c],
                cmap="seismic",
                vmin=-v_max,
                vmax=v_max,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#AAAAAA")

            # Column index label on first row
            if r == 0:
                ax.set_title(
                    f"$t_{{{c+1}}}$",
                    fontsize=9,
                    pad=3,
                    color="#333333",
                )

    # ── Family legend ──────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(
            facecolor=FAMILY_COLORS[k], edgecolor="none", label=FAMILY_LABELS[k]
        )
        for k in sorted(FAMILY_COLORS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.62, 0.995),
        ncol=3,
        fontsize=8.5,
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        handlelength=1.2,
        handleheight=0.9,
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, "fig_method_showcase.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
