#!/bin/bash
#SBATCH --job-name=task_fit_cv
#SBATCH --output=/home/mrenke/logs/fit_nPRF-2D_dnumr_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --gres gpu:1
#SBATCH --time=6:00:00

source /etc/profile.d/lmod.sh
module load gpu cuda

# . $HOME/init_conda.sh

. $HOME/.bashrc.sh

module load mamba
conda init
source activate numrefields

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

python $HOME/git/numrisk/numrisk/fmri_analysis/encoding_model/fit_2D_prf.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/mrenke/ds-dnumrisk --mixture_model --same_rfs --smoothed
