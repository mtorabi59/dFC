# PydFC Paper — Representative Use Cases (Section 6.8 draft)

Suggested placement: **Section 6.8 "Representative Use Cases"**
(or as a subsection of the Discussion under "Applications and Extensibility")

Cross-references to add:
- Use Cases 1 and 2 → reference Figure A (multi-method showcase panel)
- Use Case 5 → reference `task_dFC/nifti_to_roi_signal.py` and `task_dFC/ML.py`
- Use Case 6 → reference `simul_dFC/task_data_simulator.py`

---

## 6.8 Representative Use Cases

To illustrate the breadth of scientific questions that PydFC is designed to support, we describe six representative use cases drawn from common neuroimaging research workflows. Each maps the relevant software entry points to a concrete scientific goal.

---

### Use Case 1 — End-to-End dFC Pipeline and Method-Robust Feature Extraction

**Scientific question:** How does one go from raw fMRI data to analysis-ready dFC features, and how can one assess whether a biomarker reflects genuine biology rather than an artifact of the chosen method or pipeline?

A recurring barrier in dFC research is the absence of standardized, end-to-end pipelines that connect neuroimaging inputs to analysis-ready outputs under reproducible, documented conditions. Different groups implement this chain differently — varying parcellations, normalization strategies, output formats, and feature definitions — making it difficult to compare or replicate findings even when the same estimator family is nominally used. Beyond the pipeline itself, a deeper epistemic challenge is that dFC-derived biomarkers may reflect the assumptions of the chosen estimator or hyperparameter configuration rather than any underlying biological phenomenon. A biomarker that emerges only from one method with one window length but not from others is difficult to interpret as biologically meaningful. PydFC addresses both levels: it provides a unified pipeline from NIfTI images to dFC objects, and — critically — it lets users vary both methods *and* hyperparameters within the same framework, enabling a principled assessment of which findings are robust and which are analysis-contingent.

The full pipeline proceeds as follows.

**Step 1 — Convert preprocessed fMRI to parcellated time series.**
`nifti2timeseries()` (single subject) or `multi_nifti2timeseries()` (cohort) from `pydfc/data_loader.py` converts preprocessed NIfTI files to `TIME_SERIES` objects under a user-specified parcellation atlas. Confound removal, TR information, and subject identifiers are embedded in the object:
```python
from pydfc import data_loader
BOLD = data_loader.multi_nifti2timeseries(
    nifti_files_list, subj_id_list, n_rois=100, Fs=1/TR
)
```

**Step 2 — Initialize methods and hyperparameter variants.**
`measures_initializer()` from `pydfc/multi_analysis_utils.py` generates all method objects from a single shared parameter dictionary, ensuring matched hyperparameters across methods by default. The `alter_hparams` argument simultaneously explores hyperparameter sensitivity — sweeping `n_states` or window length — so biomarker robustness can be evaluated across both the estimator axis and the hyperparameter axis in one unified pipeline:
```python
MEASURES_lst, hyper_param_info = multi_analysis_utils.measures_initializer(
    MEASURES_name_lst=["SlidingWindow", "Time-Freq", "CAP", "ContinuousHMM",
                       "Windowless", "Clustering", "DiscreteHMM"],
    params_methods={"W": 44, "n_overlap": 0.5, "n_states": 12, ...},
    alter_hparams={"n_states": [6, 12, 16], "W": [30, 44, 60]}
)
```
`MEASURES_lst` contains one object per (method, hyperparameter) combination; `hyper_param_info` tracks which configuration each object corresponds to.

**Step 3 — Fit group-level states.**
`estimate_group_FCS()` fits state-based methods on the training cohort, learning a shared state space that ensures states are comparable across subjects and that downstream features carry the same semantic meaning for every individual:
```python
MEASURES_fit_lst = multi_analysis_utils.estimate_group_FCS(
    time_series=BOLD, MEASURES_lst=MEASURES_lst, n_jobs=-1
)
```

**Step 4 — Compute subject-level dFC and extract features.**
`subj_lvl_dFC_assess()` applies all fitted methods to each subject, returning a list of `DFC` objects that expose a unified API regardless of the underlying method.

*Raw connectivity features* — retrieved with `dFC.get_dFC_mat()` as a 3D array of shape `(n_time, n_regions, n_regions)`. This representation preserves the complete spatial and temporal structure and can be vectorized for any downstream regression, classification, or dimensionality reduction.

*State-based summary features* — available for state-based methods and particularly compact and interpretable:
- **State probability time series** (`dFC.FCS_proba`, shape `(n_time, n_states)`): soft probability of occupying each state at each TR.
- **State time course** (`dFC.state_TC()`): hard state label sequence, from which scalar summaries — fractional occupancy, mean dwell time, state-transition frequency — are computed via `SimilarityAssessment.FO_calc()` and `SimilarityAssessment.transition_stats()`.
- **State spatial patterns** (`dFC.FCSs`): connectivity matrices defining each state, usable as spatial templates or features.

These features support any downstream goal: predicting clinical or behavioral variables, classifying groups, tracking longitudinal changes, or characterizing population-level variability.

**Step 5 — Assess robustness across methods and hyperparameters.**
`logistic_regression_classify()` and `SVM_classify()` from `pydfc/ml_utils.py` use `StratifiedGroupKFold` when subject labels are provided, preventing subject leakage. `get_permutation_scores()` provides a subject-aware null distribution. Running this pipeline across all method–hyperparameter combinations produces a robustness profile: a biomarker that holds across methods from all three families *and* across multiple hyperparameter values is substantially more credible as a biological signal than one that appears only under specific analytical choices. Conversely, a finding sensitive to hyperparameters or present only in one method family should be reported with explicit caveats — a transparency that PydFC's unified interface makes straightforward rather than burdensome.

---

### Use Case 2 — Brain State Repertoire and Transition Structure Analysis

**Scientific question:** What recurring connectivity states does a population occupy, how long do subjects remain in each, and how often do they transition between them?

Characterizing the brain's state repertoire — spatial patterns, dwell times, and transition structure — is a core goal in resting-state fMRI, widely used to study aging, disease, and cognitive variability. Running this analysis across multiple state-based methods simultaneously reveals which state properties are method-invariant (and therefore more biologically interpretable) and which are method-specific.

After running `estimate_group_FCS()` and `subj_lvl_dFC_assess()`, the state repertoire is accessible through the `DFC` object API. The spatial pattern of each state is retrieved via `dFC.FCSs` (a dict of `(n_regions, n_regions)` connectivity matrices); the state time course is retrieved with `dFC.state_TC()`; and fractional occupancy and transition statistics are computed via the comparison module:
```python
from pydfc.comparison import SimilarityAssessment
sim = SimilarityAssessment(dFC_lst)
FO_list    = sim.FO_calc(dFC_lst)           # fractional occupancy per method
trans_list = sim.transition_stats(dFC_lst)  # dwell times and transition counts
```
`visualize_FCS()` from `pydfc/dfc_utils.py` produces a publication-ready figure showing connectivity matrices for each state (top row) and mean region activations on a brain surface (bottom row) via nilearn's `plot_markers`. Running this across CAP, SWC, CHMM, DHMM, and WL with the same `n_states` directly tests whether recovered state patterns are consistent across methods.

---

### Use Case 3 — Hyperparameter Sensitivity Profiling

**Scientific question:** How sensitive are dFC results to the choice of window length or number of states?

Hyperparameter choice is a major but underreported source of variability in dFC analysis. Window length for SW governs the temporal resolution/statistical stability tradeoff; `n_states` defines the granularity of the state space. Neither has a universally optimal value, and results can change substantially across plausible choices.

`measures_initializer()` accepts lists of alternative values via `alter_hparams`:
```python
alter_hparams = {"n_states": [4, 6, 8, 12, 16, 20]}
MEASURES_lst, hyper_param_info = multi_analysis_utils.measures_initializer(
    ["CAP", "ContinuousHMM", "Clustering"],
    params_methods, alter_hparams
)
```
This generates one measure object per combination. After running the full pipeline, cross-variant similarity is computed via `SimilarityAssessment.assess_similarity()`, quantifying how much dFC estimates change as `n_states` increases — turning a hidden design choice into a transparent, reported sensitivity curve and directly addressing the hyperparameter sensitivity limitation discussed in Section 10.2.

---

### Use Case 4 — Subject-Level dFC Fingerprinting and Individual Identifiability

**Scientific question:** How reliably does dFC capture individual-specific connectivity patterns, and does this vary by method?

Connectivity fingerprinting — identifying individuals from their connectivity patterns — has emerged as a benchmark for the reliability and individual specificity of FC measures (Finn et al., 2015). Whether dFC estimates are as individually identifiable as static FC, and whether identifiability is method-dependent, are open questions with direct implications for the use of dFC as a precision biomarker.

The inter-subject similarity level in `SimilarityAssessment.assess_similarity()` quantifies how distinguishable subjects are from each other based on their dFC patterns. After running `subj_lvl_dFC_assess()` across a multi-session dataset, the inter-subject correlation matrix is computed for each method. A method that produces highly individual-specific dFC estimates will show strong diagonal dominance (self-correlation across sessions exceeding cross-subject correlation). Comparing this structure across all 7 methods with a single call to `assess_similarity()` benchmarks individual identifiability as a function of method choice — an analysis that previously required constructing custom pipelines for each estimator separately.

---

### Use Case 5 — Task-Informed dFC Analysis: Aligning Connectivity Dynamics with Experimental Design

**Scientific question:** How does connectivity change in relation to task onset and offset, and which brain state is recruited during task performance?

Task-based fMRI provides a rare opportunity to ground dFC analysis in an external, experimentally controlled reference signal. Comparing the temporal structure of estimated connectivity dynamics with task timing enables assessment of state recruitment, state-to-task alignment, and the degree to which different methods detect task-evoked connectivity changes — one of the most direct ways to assess whether dFC estimates reflect meaningful neural dynamics rather than noise or method artifacts.

`pydfc/task_utils.py` provides utilities for parsing TSV-format event files and converting them to binary task-presence labels aligned with the MRI TR. `extract_task_presence()` convolves event onsets and durations with a hemodynamic response function and downsamples to the MRI sampling rate:
```python
from pydfc import task_utils
task_labels, valid_TRs = task_utils.extract_task_presence(
    event_labels, TR_task=1/task_Fs, TR_mri=TR, binary=True, binarizing_method="GMM"
)
```
This label vector is directly comparable to `dFC.state_TC()` or `dFC.FCS_proba`. Users can compute the correlation between task presence and state probability to identify which states are preferentially recruited during task performance, or compute transition frequency and dwell time via `SimilarityAssessment.transition_stats()` separately for task and rest periods to characterize task-induced changes in state dynamics.

The `task_dFC/nifti_to_roi_signal.py` script combines NIfTI-to-ROI conversion with automatic task-label alignment in a single step, producing matched (TIME_SERIES, task_labels) pairs ready for dFC estimation. The `task_dFC/ML.py` script provides a full task-decoding pipeline from dFC features to classification results for the specific case where task presence is the prediction target.

---

### Use Case 6 — Simulation-Based Method Recovery Benchmarking

**Scientific question:** How accurately does each method recover known dynamic connectivity patterns?

In resting-state analysis, there is no observable ground truth for dFC, making direct validation against biology impossible. Simulation provides a principled alternative: generate synthetic BOLD data with known connectivity dynamics and quantify each method's ability to recover them.

`pydfc/simul_utils.py` and `simul_dFC/task_data_simulator.py` use The Virtual Brain (TVB) to generate synthetic BOLD time series with user-specified, time-varying connectivity. Task-evoked transitions are introduced via the `CustomStimuli` class and `simulate_task_BOLD()`, producing data where the ground-truth state sequence is fully known. Running all 7 dFC methods on such data via `measures_initializer()` + `estimate_group_FCS()` + `subj_lvl_dFC_assess()` yields a controlled benchmark: the recovered `state_TC()` or `get_dFC_mat()` output can be compared directly against the known ground truth. This provides empirical guidance on which method family is appropriate for which type of connectivity dynamics, complementing the theoretical assumptions discussion in Section 4 and directly addressing the epistemic limitation noted in Section 10.1.
