#!/bin/bash
#SBATCH --job-name=fit_st_denoise
#SBATCH --output=/home/mrenke/logs/fit_st_denoise_stim2_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --mem=16G 
#SBATCH --time=2:00:00

. $HOME/.bashrc

source activate numrefields

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

python $HOME/git/numrisk/numrisk/fmri_analysis/glm_denoise/fit_glm_denoise.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/mrenke/ds-dnumrisk
