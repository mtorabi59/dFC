#!/bin/sh
#
#SBATCH --job-name=fit_fcs_job   # Optional: Name of your job
#SBATCH --output=logs/fcs_out.txt  # Standard output log
#SBATCH --error=logs/fcs_err.txt   # Standard error log
#SBATCH --time=7-00:00:00                # Walltime for each task (7 days)
#SBATCH --mem=64G                     # Memory request per node

DATASET_INFO="./dataset_info.json"
METHODS_CONFIG="./methods_config.json"

# Activate  virtual environment
source "/home/mt00/venvs/pydfc/bin/activate"

python "/home/mt00/pydfc/dFC/task_dFC/FCS_estimate.py" \
--dataset_info $DATASET_INFO \
--methods_config $METHODS_CONFIG

deactivate
