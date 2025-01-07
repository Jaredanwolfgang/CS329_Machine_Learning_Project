#!/bin/bash
#SBATCH --job-name="YOLO"               # 作业名称
#SBATCH --cpus-per-task=16              # 每个任务使用的 CPU 核心数
#SBATCH --gres=gpu:8                    # 请求 8 个 GPU
#SBATCH --partition=rtx2080ti           # 使用 rtx2080ti 分区
#SBATCH --qos=rtx2080ti                 # 请求 rtx2080ti QOS
#SBATCH --time=96:00:00                 # 作业最大运行时间（96小时)
#SBATCH --output=log_%j.out             # 输出文件名，%j 会被替换为作业ID

# 切换到项目目录
cd /home/turing_lab/cse12210414/projects/RT-DETR/rtdetrv2_pytorch

# 第四部分：在 Singularity 环境中执行
echo "Activating conda environment and starting second training stage"
singularity exec --nv /home/turing_lab/cse12210414/singularity/cuda_12.1.0_sandbox bash -c "conda init && source ~/.bashrc && conda activate rtdetr && sh scripts/train.sh > log_AL_augmented_0106_random_gain.txt"
