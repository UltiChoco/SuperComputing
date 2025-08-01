#!/bin/bash 
#SBATCH --job-name=dit_inference
#SBATCH --output=dit_inference.out
#SBATCH --error=dit_inference.err
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --partition=8v100-32

# 使用conda初始化
conda init bash 
source ~/.bashrc  # 加载 .bashrc，确保 conda 正常工作
conda activate DiT

# 进入项目目录
cd /work/sustcsc_11/DiT-SUSTCSC

# 处理 torch.compile 的缓存
export TORCHINDUCTOR_CACHE_DIR=/work/sustcsc_11/DiT-SUSTCSC/torch_cache

# 使用 torchrun 启动4个进程（每张卡1个进程）
torchrun --nproc_per_node=4 --master_addr=$(hostname) --master_port=29500 sample_02_profiler.py --job-id ${SLURM_JOB_ID}           


