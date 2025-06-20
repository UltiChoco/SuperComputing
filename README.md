# SUSTCSC-DiT 图像生成挑战说明

> 本Repo Fork自「Scalable Diffusion Models with Transformers」的[官方实现库](https://github.com/facebookresearch/DiT)，遵循原Repo的[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)协议。

## 代码结构

``` shell
.
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE.txt
├── README.md
├── README_Original.md
├── diffusion
│   ├── __init__.py
│   ├── diffusion_utils.py
│   ├── gaussian_diffusion.py
│   ├── respace.py
│   └── timestep_sampler.py
├── evaluate.py # Used for Evaluation, calulate the four metrics: python evaluate.py sample_baseline.png sample_modified.png 
├── models.py
├── pretrained_models
│   ├── DiT-XL-2-512x512.pt # You can copy the model from /work/share/dit
│   ├── download.py
│   └── sd-vae-ft-ema # You can copy the VAE model from /work/share/dit/sd-vae-ft-ema
├── requirements.txt # You can build the conda environment using this.
├── sample.py # Used for Baseline, rename the file to sample_baseline.png and later use for evaluation. (The picture should contains 8 subfigures in total.)
└── sample_ddp.py
```

## 环境配置

推荐使用miniconda进行环境管理，你可以通过以下指令创建环境，需要注意的是，在超算环境中，登录节点通常是没有GPU的，你需要获取到计算节点所支持的cuda版本信息（通过提交脚本的方式）

``` shell
conda env create -n sustc-dit python=3.10
conda activate sustc-dit
pip install -r requirements.txt # 不一定可以直接使用，请根据集群环境配置可以使用的环境（环境管理毕竟是超算入门第一课）
```

## 代码修改说明

你可以对代码进行修改，主要修改区域集中在 `models.py` 文件中，也可以增加新的文件来支持模型的分布式推理，无论你对代码做出如何修改，请在该Repo根目录撰写一份 `code.md` 文件来说明你做出更改的部分，以下部分不可以修改：

1. 模型参数设置
2. 模型精度设置
3. 推理过程中的Batch大小（固定为8）
4. seed种子不可修改

## 提交说明

请将作业系统生成的 `.out` `.log` 等文件统一放置在该Repo根目录下的 `log` 文件夹中，并标注哪个文件中包含了最终提交结果，最后请将整个文件除了 `pretrained_models` 文件夹以外的部分整理为一个压缩包，命名格式为 `teamid_DiT.tar.gz`，邮件发送至指定邮箱：`xiaoyc2022@mail.sustech.edu.cn`，邮件标题请以以下格式提交`[SUSTCSC-TeamID] DiT挑战赛提交`，在结果验证之后将会将分数实时更新到Board中

## Issue说明

如果对赛题有任何疑问，请在本Repo的Issue区域发布Issue