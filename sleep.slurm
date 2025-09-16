#!/bin/bash
#SBATCH --job-name=sleep_job
#SBATCH --time=00:01:00   # 设置休眠时间为 1 分钟
#SBATCH --ntasks=1         # 运行 1 个任务
#SBATCH --cpus-per-task=1  # 每个任务 1 个 CPU

# 休眠 60 秒
sleep 60