import argparse
import json
import os
import traceback

import numpy as np

from pydfc.ml_utils import (
    cluster_for_visual,
    extract_task_features,
    task_paradigm_clustering,
    task_presence_classification,
    task_presence_clustering,
)

#######################################################################################


def function_f():
    pass


#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to run across-dataset analysis on dFC results.
    """

    parser = argparse.ArgumentParser(description=HELPTEXT)

    parser.add_argument(
        "--multi_dataset_info", type=str, help="path to multi-dataset info file"
    )

    args = parser.parse_args()

    multi_dataset_info = args.multi_dataset_info

    # Read dataset info
    with open(multi_dataset_info, "r") as f:
        multi_dataset_info = json.load(f)

    print("Multi-Dataset Analysis started ...")

    main_root = multi_dataset_info["main_root"]
    DATASETS = multi_dataset_info["DATASETS"]

    try:
        function_f()
    except Exception as e:
        print(f"Error in task features extraction: {e}")
        traceback.print_exc()
    print("Task features extraction finished.")

    print("Multi-Dataset Analysis finished.")

#######################################################################################
