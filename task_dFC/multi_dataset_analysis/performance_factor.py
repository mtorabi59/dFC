import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper_functions import (  # pyright: ignore[reportMissingImports]
    RDoC_MAP,
    canon_task,
    savefig_pub,
    setup_pub_style,
)

LEVEL = "group_lvl"
GROUP = "test"

CLASSIFIER_METRIC_MAP = {
    "Logistic regression": "Logistic regression balanced accuracy",
    "SVM": "SVM balanced accuracy",
}

TIMING_FEATURES = [
    "task_ratio_avg",
    "transition_freq_avg",
    "rest_durations_median",
    "task_durations_median",
    "rest_durations_iqr",
    "task_durations_iqr",
]

COHEN_FEATURES = [
    "CohensD_max",
    "CohensD_mean",
]

TSNR_FEATURES = [
    "median_tsnr_avg_over_subjects",
]

CORR_EXCLUDE_COLUMNS = {
    "RDoC",
    "task",
    "run",
    "dFC assessment method",
    "classifier model",
    "embedding",
    "classification_balanced_accuracy",
}


def parse_args():
    helptext = """
    Build a unified run-level dataframe linking ML performance to task factors.
    """
    parser = argparse.ArgumentParser(description=helptext)
    parser.add_argument(
        "--multi_dataset_info",
        type=str,
        required=True,
        help="path to multi-dataset info file",
    )
    parser.add_argument(
        "--simul_or_real",
        type=str,
        required=True,
        choices=["simulated", "real"],
        help="Specify 'simulated' or 'real' data",
    )
    return parser.parse_args()


def read_json(json_file):
    with open(json_file, "r") as file_obj:
        return json.load(file_obj)


def load_npy_dict(path, label):
    assert os.path.exists(path), f"{label} file does not exist: {path}"
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray):
        loaded = loaded.item()
    assert isinstance(loaded, dict), f"{label} must be a dictionary. Got {type(loaded)}"
    return loaded


def assert_required_keys(data_dict, required_keys, label):
    missing = [key for key in required_keys if key not in data_dict]
    assert not missing, f"Missing required keys in {label}: {missing}"


def dict_to_df(data_dict, label):
    lengths = {key: len(value) for key, value in data_dict.items()}
    unique_lengths = set(lengths.values())
    assert len(unique_lengths) == 1, (
        f"Inconsistent column lengths in {label}: {lengths}. "
        "All arrays/lists must have equal length."
    )
    return pd.DataFrame.from_dict(data_dict)


def normalize_run(value):
    if value is None:
        return "none"
    if isinstance(value, float) and np.isnan(value):
        return "none"
    # TSV empty cells are read by pandas as NaN (float) handled above,
    # but guard against empty strings too (e.g. after manual editing).
    if str(value).strip() == "":
        return "none"
    return str(value).strip().lower()


def add_join_keys(df):
    assert "task" in df.columns, "Expected column 'task'"
    assert "run" in df.columns, "Expected column 'run'"
    df = df.copy()
    df["task_key"] = df["task"].astype(str).map(canon_task)
    df["run_key"] = df["run"].map(normalize_run)
    assert (df["task_key"].str.len() > 0).all(), "Found empty normalized task key"
    return df


def get_paths(multi_dataset_info, simul_or_real):
    output_root = multi_dataset_info["output_root"]
    return {
        "ml": f"{output_root}/ML_results/ALL_ML_SCORES_{simul_or_real}.npy",
        "timing": f"{output_root}/task_timing_stats/{simul_or_real}/task_timing_stats_{simul_or_real}.npy",
        "cohensd": f"{output_root}/CohensD/{simul_or_real}/CohensD_ML_{simul_or_real}.npy",
        "tsnr": f"{output_root}/t-SNR/tsnr_summary_grouped.tsv",
        "out_dir": f"{output_root}/performance_factor/{simul_or_real}",
    }


def prepare_ml_df(ml_scores_all):
    ml_scores = ml_scores_all

    required_keys = [
        "task",
        "run",
        "embedding",
        "dFC method",
        "group",
        *CLASSIFIER_METRIC_MAP.values(),
    ]
    assert_required_keys(ml_scores, required_keys, "ALL_ML_SCORES")

    df_ml_wide = dict_to_df(ml_scores, "ALL_ML_SCORES")
    df_ml_wide = df_ml_wide[df_ml_wide["group"] == GROUP].copy()
    assert not df_ml_wide.empty, f"No ML rows found for group='{GROUP}'"

    if "dataset" in df_ml_wide.columns:
        id_cols = ["dataset", "task", "run", "embedding", "dFC method", "group"]
    else:
        id_cols = ["task", "run", "embedding", "dFC method", "group"]

    classifier_frames = []
    for classifier, metric_key in CLASSIFIER_METRIC_MAP.items():
        frame = df_ml_wide[id_cols + [metric_key]].copy()
        frame["classifier model"] = classifier
        frame = frame.rename(columns={metric_key: "classification_balanced_accuracy"})
        classifier_frames.append(frame)

    df_ml = pd.concat(classifier_frames, ignore_index=True)
    df_ml = df_ml.rename(columns={"dFC method": "dFC assessment method"})

    score = df_ml["classification_balanced_accuracy"].astype(float)
    assert np.isfinite(score).all(), "ML performance contains NaN/Inf values"
    assert ((score >= 0.0) & (score <= 1.0)).all(), (
        "Expected balanced accuracy in [0, 1]. "
        f"Observed min={score.min()}, max={score.max()}"
    )

    return add_join_keys(df_ml)


def prepare_timing_df(timing_dict):
    required_keys = ["task", "run", *TIMING_FEATURES]
    assert_required_keys(timing_dict, required_keys, "task_timing_stats")
    df_timing = dict_to_df(timing_dict, "task_timing_stats")

    keep_cols = ["task", "run", *TIMING_FEATURES]
    if "dataset" in df_timing.columns:
        keep_cols = ["dataset", *keep_cols]
    df_timing = df_timing[keep_cols].copy()

    for col in TIMING_FEATURES:
        values = df_timing[col].astype(float)
        assert np.isfinite(values).all(), f"Timing feature '{col}' contains NaN/Inf"

    return add_join_keys(df_timing)


def prepare_cohensd_df(cohensd_dict):
    required_keys = ["task", "run", *COHEN_FEATURES]
    assert_required_keys(cohensd_dict, required_keys, "CohensD_ML")
    df_cohensd = dict_to_df(cohensd_dict, "CohensD_ML")

    keep_cols = ["task", "run", *COHEN_FEATURES]
    if "dataset" in df_cohensd.columns:
        keep_cols = ["dataset", *keep_cols]
    df_cohensd = df_cohensd[keep_cols].copy()

    for col in COHEN_FEATURES:
        values = df_cohensd[col].astype(float)
        assert np.isfinite(values).all(), f"Cohen's D feature '{col}' contains NaN/Inf"

    return add_join_keys(df_cohensd)


def prepare_tsnr_df(tsnr_path):
    assert os.path.exists(tsnr_path), f"tSNR file does not exist: {tsnr_path}"
    df_tsnr = pd.read_csv(tsnr_path, sep="\t")

    required_cols = ["dataset", "task", "run", *TSNR_FEATURES]
    missing = [col for col in required_cols if col not in df_tsnr.columns]
    assert not missing, f"Missing required columns in tsnr_summary_grouped.tsv: {missing}"

    df_tsnr = df_tsnr[["dataset", "task", "run", *TSNR_FEATURES]].copy()

    # Validate tSNR values (allow NaN — runs with no data are left empty as specified)
    tsnr_vals = df_tsnr["median_tsnr_avg_over_subjects"].astype(float)
    assert (
        tsnr_vals.dropna() > 0
    ).all(), (
        "median_tsnr_avg_over_subjects contains non-positive values where data is present"
    )

    return add_join_keys(df_tsnr)


def choose_join_keys(df_ml, df_timing, df_cohensd, df_tsnr):
    has_dataset_everywhere = all(
        "dataset" in df.columns for df in [df_ml, df_timing, df_cohensd, df_tsnr]
    )

    base_keys = ["task_key", "run_key"]
    dataset_keys = ["dataset", *base_keys]

    timing_dupes_base = df_timing.duplicated(subset=base_keys).sum()
    cohensd_dupes_base = df_cohensd.duplicated(subset=base_keys).sum()
    tsnr_dupes_base = df_tsnr.duplicated(subset=base_keys).sum()

    if timing_dupes_base == 0 and cohensd_dupes_base == 0 and tsnr_dupes_base == 0:
        return base_keys

    if has_dataset_everywhere:
        timing_dupes_dataset = df_timing.duplicated(subset=dataset_keys).sum()
        cohensd_dupes_dataset = df_cohensd.duplicated(subset=dataset_keys).sum()
        tsnr_dupes_dataset = df_tsnr.duplicated(subset=dataset_keys).sum()
        assert timing_dupes_dataset == 0, (
            "task_timing_stats still has duplicate rows per dataset/task/run after "
            f"normalization. duplicate_count={timing_dupes_dataset}"
        )
        assert cohensd_dupes_dataset == 0, (
            "CohensD_ML still has duplicate rows per dataset/task/run after "
            f"normalization. duplicate_count={cohensd_dupes_dataset}"
        )
        assert tsnr_dupes_dataset == 0, (
            "tsnr_summary_grouped still has duplicate rows per dataset/task/run after "
            f"normalization. duplicate_count={tsnr_dupes_dataset}"
        )
        return dataset_keys

    raise AssertionError(
        "Ambiguous join on task/run (duplicates found), and dataset is not available "
        "in all sources to disambiguate."
    )


def merge_with_checks(df_ml, df_timing, df_cohensd, df_tsnr, join_keys):
    timing_cols = join_keys + TIMING_FEATURES
    cohensd_cols = join_keys + COHEN_FEATURES
    tsnr_cols = join_keys + TSNR_FEATURES

    df_merged = df_ml.merge(
        df_timing[timing_cols],
        on=join_keys,
        how="left",
        validate="many_to_one",
        indicator="timing_merge",
    )
    timing_unmatched = (df_merged["timing_merge"] != "both").sum()
    assert (
        timing_unmatched == 0
    ), f"Could not match timing stats for {timing_unmatched} ML rows using keys {join_keys}"
    df_merged = df_merged.drop(columns=["timing_merge"])

    df_merged = df_merged.merge(
        df_cohensd[cohensd_cols],
        on=join_keys,
        how="left",
        validate="many_to_one",
        indicator="cohensd_merge",
    )
    cohensd_unmatched = (df_merged["cohensd_merge"] != "both").sum()
    assert (
        cohensd_unmatched == 0
    ), f"Could not match Cohen's D stats for {cohensd_unmatched} ML rows using keys {join_keys}"
    df_merged = df_merged.drop(columns=["cohensd_merge"])

    # tSNR: left join — rows with no tSNR data (e.g. None-run datasets not in file)
    # will have NaN, which is acceptable as specified.
    df_merged = df_merged.merge(
        df_tsnr[tsnr_cols],
        on=join_keys,
        how="left",
        validate="many_to_one",
    )

    return df_merged


def add_rdoc(df, simul_or_real):
    task_to_domain = RDoC_MAP[simul_or_real]["TASK2DOMAIN"]
    df = df.copy()
    df["RDoC"] = df["task_key"].map(task_to_domain)

    missing_mask = df["RDoC"].isna()
    if missing_mask.any():
        missing_tasks = sorted(df.loc[missing_mask, "task"].astype(str).unique())
        raise AssertionError(
            "Missing RDoC mapping for tasks (after canonicalization): "
            f"{missing_tasks}. Update helper_functions.RDoC_MAP if needed."
        )

    return df


def finalize_columns(df):
    cols = []
    if "dataset" in df.columns:
        cols.append("dataset")
    cols += [
        "task",
        "run",
        "RDoC",
        *TIMING_FEATURES,
        *COHEN_FEATURES,
        *TSNR_FEATURES,
        "dFC assessment method",
        "classifier model",
        "embedding",
        "classification_balanced_accuracy",
    ]

    missing = [col for col in cols if col not in df.columns]
    assert not missing, f"Missing expected final columns: {missing}"

    out = df[cols].copy()
    sort_cols = ["task", "run", "dFC assessment method", "classifier model", "embedding"]
    if "dataset" in out.columns:
        sort_cols = ["dataset", *sort_cols]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def save_outputs(df, out_dir, simul_or_real):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/performance_factor_{simul_or_real}.csv"
    pkl_path = f"{out_dir}/performance_factor_{simul_or_real}.pkl"
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    return csv_path, pkl_path


def build_correlation_table(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    factor_cols = [
        col
        for col in numeric_cols
        if col not in CORR_EXCLUDE_COLUMNS and col != "classification_balanced_accuracy"
    ]
    assert factor_cols, "No numeric factor columns available for correlation analysis"

    rows = []
    for method, group_df in df.groupby("dFC assessment method", observed=True):
        for factor in factor_cols:
            pair_df = group_df[[factor, "classification_balanced_accuracy"]].dropna()
            n_samples = len(pair_df)

            if (
                n_samples < 3
                or pair_df[factor].nunique(dropna=True) < 2
                or pair_df["classification_balanced_accuracy"].nunique(dropna=True) < 2
            ):
                corr = np.nan
            else:
                corr = pair_df[factor].corr(
                    pair_df["classification_balanced_accuracy"], method="pearson"
                )

            rows.append(
                {
                    "factor": factor,
                    "dFC assessment method": method,
                    "correlation": corr,
                    "n_samples": n_samples,
                }
            )

    corr_df = pd.DataFrame(rows)
    corr_df["factor"] = pd.Categorical(
        corr_df["factor"], categories=factor_cols, ordered=True
    )
    corr_df = corr_df.sort_values(["factor", "dFC assessment method"]).reset_index(
        drop=True
    )
    return corr_df


def plot_factor_correlation_pointplot(corr_df, out_dir, simul_or_real):
    valid_df = corr_df.dropna(subset=["correlation"]).copy()
    assert (
        not valid_df.empty
    ), "All factor correlations are NaN; cannot generate correlation pointplot"

    n_factors = valid_df["factor"].nunique()
    width = max(10, 0.75 * n_factors)
    height = 5.5

    figure, ax = plt.subplots(figsize=(width, height))
    sns.pointplot(
        data=valid_df,
        x="factor",
        y="correlation",
        hue="dFC assessment method",
        dodge=0.4,
        errorbar=None,
        markers="o",
        linestyles="",
        ax=ax,
    )

    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Factor")
    ax.set_ylabel("Pearson correlation with classification balanced accuracy")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(title="dFC assessment method", frameon=True)
    sns.despine(ax=ax, top=True, right=True)
    figure.tight_layout()

    fig_path = f"{out_dir}/performance_factor_correlation_pointplot_{simul_or_real}.png"
    savefig_pub(fig_path)
    plt.close(figure)
    return fig_path


def main():
    args = parse_args()
    setup_pub_style()

    multi_dataset_info = read_json(args.multi_dataset_info)
    paths = get_paths(multi_dataset_info, args.simul_or_real)

    ml_scores_all = load_npy_dict(paths["ml"], "ALL_ML_SCORES")
    timing_dict = load_npy_dict(paths["timing"], "task_timing_stats")
    cohensd_dict = load_npy_dict(paths["cohensd"], "CohensD_ML")

    df_ml = prepare_ml_df(ml_scores_all)
    df_timing = prepare_timing_df(timing_dict)
    df_cohensd = prepare_cohensd_df(cohensd_dict)
    df_tsnr = prepare_tsnr_df(paths["tsnr"])

    join_keys = choose_join_keys(df_ml, df_timing, df_cohensd, df_tsnr)
    print(f"Using join keys: {join_keys}")

    df = merge_with_checks(df_ml, df_timing, df_cohensd, df_tsnr, join_keys)
    df = add_rdoc(df, args.simul_or_real)
    df = finalize_columns(df)

    csv_path, pkl_path = save_outputs(df, paths["out_dir"], args.simul_or_real)

    corr_df = build_correlation_table(df)
    corr_csv_path = (
        f"{paths['out_dir']}/performance_factor_correlations_{args.simul_or_real}.csv"
    )
    corr_df.to_csv(corr_csv_path, index=False)
    corr_fig_path = plot_factor_correlation_pointplot(
        corr_df, paths["out_dir"], args.simul_or_real
    )

    print(f"Saved dataframe with shape: {df.shape}")
    print(f"CSV: {csv_path}")
    print(f"PKL: {pkl_path}")
    print(f"Correlation CSV: {corr_csv_path}")
    print(f"Correlation figure: {corr_fig_path}")


if __name__ == "__main__":
    main()
