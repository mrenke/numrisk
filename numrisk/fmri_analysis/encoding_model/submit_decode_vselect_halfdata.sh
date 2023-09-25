#!/bin/bash
#SBATCH --job-name=decode_volume
#SBATCH --output=/home/mrenke/logs/decode_svox_dnumr_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --gres gpu:1
#SBATCH --time=3:00:00

source /etc/profile.d/lmod.sh
module load gpu cuda

# . $HOME/init_conda.sh

. $HOME/.bashrc.sh

source activate numrefields

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate tf2-gpu

python $HOME/git/numrisk/numrisk/fmri_analysis/encoding_model/decode_select_voxels_cv.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/mrenke/ds-dnumrisk --mask NPC_R  --denoise --split_data run_123
python $HOME/git/numrisk/numrisk/fmri_analysis/encoding_model/decode_select_voxels_cv.py $PARTICIPANT_LABEL --bids_folder /shares/zne.uzh/mrenke/ds-dnumrisk --mask NPC_R  --denoise --split_data run_456