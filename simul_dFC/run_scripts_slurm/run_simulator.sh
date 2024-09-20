#!/bin/bash
#
#SBATCH --job-name=simul_dfc_job   # Optional: Name of your job
#SBATCH --output=logs/simul_out.txt  # Standard output log
#SBATCH --error=logs/simul_err.txt   # Standard error log
#SBATCH --account=def-jbpoline           # Account
#SBATCH --time=24:00:00                # Walltime for each task (24 hours)
#SBATCH --mem=8G                     # Memory request per node
#SBATCH --array=1-200                # Task array specification

DATASET_INFO="./dataset_info.json"

# Activate  virtual environment
source "/home/mt00/venvs/pydfc/bin/activate"

# Run Python script
python "/home/mt00/pydfc/dFC/simul_dFC/task_data_simulator.py" \
--dataset_info $DATASET_INFO

# Deactivate environment
deactivate
