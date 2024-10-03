#!/bin/bash
#
#SBATCH --job-name=fmriprep_job       # Name of the job
#SBATCH --output=logs/fmriprep_out.log  # Standard output log
#SBATCH --error=logs/fmriprep_err.log   # Standard error log
#SBATCH --time=24:00:00                # Walltime (24 hours)
#SBATCH --mem=64G                      # Memory (64 GB)
#SBATCH --cpus-per-task=8              # Number of CPU cores (increase based on availability)
#SBATCH --account=def-jbpoline           # Account
#SBATCH --tmp=100G                     # Allocate 100GB of temporary space

module load apptainer

source "/home/mt00/venvs/nipoppy_env/bin/activate"

SUBJECT_LIST="./subj_list.txt"

echo "Number subjects found: $(wc -l < $SUBJECT_LIST)"

SUBJECT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
echo "Subject ID: $SUBJECT_ID"

nipoppy run \
--pipeline fmriprep \
--dataset-root "$(dirname "$(pwd)")" \
--participant-id $SUBJECT_ID

deactivate
