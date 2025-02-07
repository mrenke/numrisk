#!/bin/bash
#SBATCH --job-name=task_fit_cv
#SBATCH --output=/home/mrenke/logs/decode_2D_dnumr_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=6:00:00

source /etc/profile.d/lmod.sh
module load gpu cuda

# . $HOME/init_conda.sh

. $HOME/.bashrc

module load mamba
conda init
source activate numrefields

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

python $HOME/git/numrisk/numrisk/fmri_analysis/encoding_model/decode_2D.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/mrenke/ds-dnumrisk --two_dimensional --mixture_model --same_rfs --n_voxels 100
