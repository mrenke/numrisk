#!/bin/bash
#SBATCH --job-name=task_fit_cv
#SBATCH --output=/home/mrenke/logs/fit_nPRF-cv_dnumr_halfdata_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --gres gpu:1
#SBATCH --time=6:00:00

source /etc/profile.d/lmod.sh
module load gpu cuda

# . $HOME/init_conda.sh

. $HOME/.bashrc.sh

source activate numrefields

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate tf2-gpu
python $HOME/git/numrisk/numrisk/fmri_analysis/encoding_model/fit_nprf_cv.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/mrenke/ds-dnumrisk --denoise --split_data run_123   
python $HOME/git/numrisk/numrisk/fmri_analysis/encoding_model/fit_nprf_cv.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/mrenke/ds-dnumrisk --denoise --split_data run_456   
