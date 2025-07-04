#!/bin/bash
#SBATCH --job-name=dit_inference
#SBATCH --output=dit_inference.out
#SBATCH --error=dit_inference.err
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --partition=8v100-32

conda init
conda activate DiT
cd /work/sustcsc_11/DiT-SUSTCSC
python sample.py
