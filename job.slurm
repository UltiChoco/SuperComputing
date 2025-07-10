#!/bin/bash 
#SBATCH --job-name=dit_inference
#SBATCH --output=dit_inference.out
#SBATCH --error=dit_inference.err
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=8v100-32

# 使用conda初始化
conda init bash 
source ~/.bashrc  # 加载 .bashrc，确保 conda 正常工作
conda activate DiT

# 将预训练模型路径添加到 PYTHONPATH
export PYTHONPATH=/work/sustcsc_11/DiT-SUSTCSC/pretrained_models:$PYTHONPATH

# 进入项目目录
cd /work/sustcsc_11/DiT-SUSTCSC

# 执行代码
python -m torch.distributed.run --nproc_per_node=4 sample_ddp.py --job-id ${SLURM_JOB_ID}


