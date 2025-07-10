# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from pretrained_models.download import find_model
from models import DiT_models
import argparse

import torch.profiler
import os
import time

# ============ 获取作业号并构造输出目录 ============ #
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
parser.add_argument("--num-classes", type=int, default=1000)
parser.add_argument("--cfg-scale", type=float, default=4.0)
parser.add_argument("--num-sampling-steps", type=int, default=250)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--ckpt", type=str, default=None,
                    help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
parser.add_argument("--job-id", type=str, default=os.environ.get("SLURM_JOB_ID", "nojob"), help="Job ID for naming outputs")
args = parser.parse_args()

job_id = args.job_id
job_dir = os.path.join(".", f"job_{job_id}")
log_dir = os.path.join(job_dir, f"log_{job_id}")
os.makedirs(log_dir, exist_ok=True)

# ===================================================

print(">> Running sample_01.py with job_id =", args.job_id, flush=True)


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # which GPU you are using
    print("Using device:", device, flush=True)
    if device == "cuda":
        print("CUDA device name:", torch.cuda.get_device_name(), flush=True)

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

   # === 加载标准 DiT 权重 ===
    ckpt_path = args.ckpt or f"pretrained_models/DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    print(f"Loading checkpoint from: {ckpt_path}", flush=True)
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()


    
    # === TorchScript 编译 ===
    try:
        scripted_model = torch.jit.script(model)
        print("✅ TorchScript 编译成功")
        # 保存脚本模型
        torch.jit.save(scripted_model, os.path.join(job_dir, f"{args.model.replace('/', '-')}_scripted.pt"))
        print("📦 脚本模型已保存")
        model = scripted_model
    except Exception as e:
        print("❌ TorchScript 编译失败:", e)
        return
    
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"pretrained_models/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    time_total = []

    # with torch.profiler.profile(
    #         activities=[
    #             torch.profiler.ProfilerActivity.CPU,  # CPU性能分析
    #             torch.profiler.ProfilerActivity.CUDA  # GPU性能分析
    #         ],
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),  # 保存性能数据到log文件
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True
    #     ) as prof:

    
    start_time = time.time()

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Run sampling:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device
    )
    samples, _ = samples.chunk(2, dim=0) # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    end_time = time.time()

    # prof.step()

    time_total.append(end_time - start_time)
    
    out_path = os.path.join(job_dir, f"job_{job_id}.png")
    save_image(samples, out_path, nrow=4, normalize=True, value_range=(-1, 1))
   
    print(f"Image saved: {out_path}", flush=True)
    print(f"Profiler logs saved to: {log_dir}", flush=True)
    print(f"⏱️ sampling time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main(args)