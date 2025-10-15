#!/bin/sh
#
#SBATCH --job-name=across_dataset_analysis
#SBATCH --output=logs/%x_out.txt
#SBATCH --error=logs/%x_err.txt
#SBATCH --time=05:00:00
#SBATCH --mem=32G

# === Activate virtual environment ===
source "/home/mt00/venvs/pydfc/bin/activate"

# === Global variable ===
MULTI_DATASET_INFO="/home/mt00/pydfc/dFC/task_dFC/run_scripts_slurm/multi_dataset_info.json"

# === Arguments ===
SCRIPT_NAME=$1        # e.g., ml_results.py
SIMUL_OR_REAL=$2      # e.g., real or simulated
SCRIPT_DIR="/home/mt00/pydfc/dFC/task_dFC/multi_dataset_analysis"

# === Safety checks ===
if [ -z "$SCRIPT_NAME" ]; then
    echo "Usage: sbatch run_analysis.sh <script_name> [real|simulated]"
    exit 1
fi

if [ -z "$SIMUL_OR_REAL" ]; then
    SIMUL_OR_REAL="real"  # default
fi

SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script '$SCRIPT_PATH' not found."
    exit 1
fi

# === Run based on script name ===
case "$SCRIPT_NAME" in
    ml_results.py | dfc_visualization.py | LE_embedding_visualization.py | sample_matrix_visualization.py | task_presence_binarization.py | task_timing_stats.py)
        python "$SCRIPT_PATH" --multi_dataset_info "$MULTI_DATASET_INFO" --simul_or_real "$SIMUL_OR_REAL"
        ;;
    cohensd.py)
        python "$SCRIPT_PATH" --multi_dataset_info "$MULTI_DATASET_INFO"
        ;;
    *)
        echo "Unknown script: $SCRIPT_NAME"
        exit 1
        ;;
esac

# === Deactivate virtual environment ===
deactivate
