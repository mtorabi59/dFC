#!/bin/sh
#
#SBATCH --job-name=ML_job   # Optional: Name of your job
#SBATCH --output=logs/ML_out.txt  # Standard output log
#SBATCH --error=logs/ML_err.txt   # Standard error log
#SBATCH --time=4-00:00:00                # Walltime for each task (4 days)
#SBATCH --mem=128G                     # Memory request per node

DATASET_INFO="./dataset_info.json"

# Activate  virtual environment
source "/home/mt00/venvs/pydfc/bin/activate"

python "/home/mt00/pydfc/dFC/task_dFC/ML.py" \
--dataset_info $DATASET_INFO

deactivate
