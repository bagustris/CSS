#!/bin/bash
#$ -o logs/
#$ -l rt_G.small=1
#$ -l h_rt=12:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.7 cuda/10.2/10.2.89 cudnn/8.0/8.0.5
source /home/aad13432ni/github/CSS/venv-py38-pt17/bin/activate
python train.py -c configs/Experiments_Final_holdout/exp_abci.yaml
deactivate
