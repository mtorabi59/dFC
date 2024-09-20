#!/bin/bash
#
#SBATCH --job-name=simul_dfc_job   # Optional: Name of your job
#SBATCH --output=logs/simul_out.txt  # Standard output log
#SBATCH --error=logs/simul_err.txt   # Standard error log
#SBATCH --account=def-jbpoline           # Account
#SBATCH --mem=8G                     # Memory request per node
#SBATCH --array=1-200                # Task array specification

# Activate  virtual environment
source "/home/mt00/pydfc/bin/activate"

# Run Python script
python "/home/mt00/pydfc/dFC/simul_dFC/task_data_simulator.py"

# Deactivate environment
deactivate
