# 项目运行说明文档

## 1. 环境配置及依赖安装

### 1.1 操作系统信息
- 操作系统：`Linux`
- 版本：`4.18.0-372.32.1.el8_6.x86_64`
- 架构：`x86_64`

### 1.2 Python 版本
- Python 版本：`3.10.18`

### 1.3 Python 依赖

- Python 依赖：

- **深度学习相关库**：
  torch==2.7.0
  torchmetrics==1.7.3
  torchvision==0.22.0
  timm==1.0.15
  triton==3.3.0
  diffusers==0.33.1
  safetensors==0.5.3

- **CUDA相关库**：
  nvidia-cuda-cu12==12.6.4.1
  nvidia-cudnn-cu12==9.5.1.17
  nvidia-nccl-cu12==2.26.2
  nvidia-cuda-nvrtc-cu12==12.6.77


- **常见工具和库**：
  numpy==1.24.4
  requests==2.32.4
  psutil==7.0.0
  protobuf==6.31.1
  tensorboard==2.19.0
  Jinja2==3.1.6
  Markdown==3.8.2
  pillow==11.2.1
  packaging==25.0

查看所有已安装的依赖：

```bash
`pip freeze`
```

### 1.4 conda环境
- `base` 环境：位于 `/work/sustcsc_11/miniconda3`
- `DiT` 环境：位于 `/work/sustcsc_11/miniconda3/envs/DiT`
- `myenv` 环境：位于 `/work/sustcsc_11/miniconda3/envs/myenv`

### 1.5 GPU环境
- GPU型号：`NVIDIA V100 GPU (32GB)`
- CUDA版本：`11.8`

## 2. 依赖安装
以下是本项目的Python依赖，确保通过`pip`或`conda`安装以下包：

diffusers==0.33.1
numpy==1.24.4
Pillow==11.2.1
timm==1.0.15
torch==2.7.0
torchmetrics==1.7.3
torchvision==0.22.0
tqdm==4.66.4

## 3.编译步骤

本项目不涉及额外的编译步骤，直接依赖 Python 环境及深度学习框架。

## 4.运行步骤

### 4.1 如何启动程序
利用`job.slurm`（作业提交脚本）提交代码，本项目中有多个优化版本的代码，若想运行某一版本，于`job.slurm`最后一行修改运行的文件。现在默认是采用的最优版本。
注意，`baseline`代码为`sample_baseline.py`
目前（截至提交）最优版本代码为`sample_best.py`

### 4.2 运行结果
-每一份提交的作业都会`新生成以作业号命名的目录`，存放性能日志和生成的图像。都根据作业号命名。
-最优版本中已删除profiler。若使用未删除profiler的版本，性能日志将生成在对应作业的目录下。
  结果目录结构如下：

  ```shell
  <DiT-SUSTCSC>
  ├── job_<job_id>/
  │   ├── log_<job_id>/       # 这个目录保存了性能分析的日志文件
  │   │   ├── <log_files>     # 包含性能日志的文件（例如：.pt.trace.json等）
  │   └── sample_<job_id>.png # 生成的采样图片

```

-测试过程中使用`profetto`和`TensorBoard`分析，上传profetto之前需先更改文件权限：
```bash
chmod 644 <file_path>
```
























