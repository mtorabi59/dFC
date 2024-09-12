#!/bin/bash
#
#SBATCH --job-name=fmriprep_job       # Name of the job
#SBATCH --output=logs/fmriprep_out.log  # Standard output log
#SBATCH --error=logs/fmriprep_err.log   # Standard error log
#SBATCH --time=24:00:00                # Walltime (24 hours)
#SBATCH --mem=32G                      # Memory (32 GB)
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --account=rrg-jbpoline           # Account

source "/home/mt00/projects/rrg-jbpoline/mt00/venvs/nipoppy_env/bin/activate"

SUBJECT_LIST="./subj_list.txt"

echo "Number subjects found: $(wc -l < $SUBJECT_LIST)"

SUBJECT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
echo "Subject ID: $SUBJECT_ID"

nipoppy run \
--pipeline fmriprep \
--dataset-root "$(dirname "$(pwd)")" \
--participant-id $SUBJECT_ID

deactivate
