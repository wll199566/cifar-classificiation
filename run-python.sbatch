#!/bin/bash
#SBATCH --partition titanx-short
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 20GB
#SBATCH --time 1:00:00
#SBATCH --job-name python
#SBATCH --output slurm-%J.out 
#SBATCH -e slurm-%J.err
#SBATCH --mail-type END
#SBATCH --mail-user huiwei@umass.edu


source activate mldl
sleep 1
python train.py --model resnet --n 3 --epochs 20 --lr 0.001 --batch_size 256 --weight_decay 0 --momentum 0
#python model.py
