#!/bin/sh
#
#SBATCH --job-name=ML_job   # Optional: Name of your job
#SBATCH --output=logs/ML_out.txt  # Standard output log
#SBATCH --error=logs/ML_err.txt   # Standard error log
#SBATCH --account=def-jbpoline           # Account
#SBATCH --time=72:00:00                # Walltime for each task (72 hours)
#SBATCH --mem=64G                     # Memory request per node

DATASET_INFO="./dataset_info.json"

# Activate  virtual environment
source "/home/mt00/venvs/pydfc/bin/activate"

python "/home/mt00/pydfc/dFC/task_dFC/ML.py" \
--dataset_info $DATASET_INFO

deactivate
