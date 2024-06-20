#!/bin/sh
#
#$ -cwd
#$ -o logs/report_out.txt
#$ -e logs/report_err.txt
#$ -l h_vmem=16G
#$ -q origami.q

DATASET_INFO="./dataset_info.json"

source /data/origami/dFC/anaconda3/etc/profile.d/conda.sh
conda activate pydfc
python "/data/origami/dFC/CODEs/pydfc/dFC/task_dFC/generate_report.py" \
--dataset_info $DATASET_INFO

conda deactivate
