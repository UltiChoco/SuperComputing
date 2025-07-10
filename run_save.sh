#!/bin/bash
#SBATCH --job-name=save_dit_weights
#SBATCH --output=save_dit_weights.out
#SBATCH --error=save_dit_weights.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --partition=8v100-32

# 激活 conda 环境
source ~/.bashrc
conda activate DiT

# 切换到代码目录
cd /work/sustcsc_11/DiT-SUSTCSC

# 执行脚本
python save_dit_weights.py
