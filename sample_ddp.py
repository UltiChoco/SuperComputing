import torch
import torch.distributed as dist
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from pretrained_models.download import find_model
from models import DiT_models
import argparse
import os
import math
import torch.profiler
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

# Add the path to the pretrained_models directory to the Python path
sys.path.append('/work/sustcsc_11/DiT-SUSTCSC/pretrained_models')  # Add the path to Python path

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
print(">> Running sample.py with job_id =", args.job_id, flush=True)

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get local_rank from environment variable (passed by torchrun)
    local_rank = int(os.environ['LOCAL_RANK'])  # torchrun passes local_rank automatically
    torch.cuda.set_device(local_rank)  # Set the device for the current process

    # Initialize Distributed Data Parallel (DDP)
    dist.init_process_group("nccl")  # DDP initialization

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

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"pretrained_models/DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"pretrained_models/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Setup the output folder:
    model_string_name = args.model.replace("/", "-")
    folder_name = f"{model_string_name}-{args.ckpt or 'pretrained'}-size-{args.image_size}-vae-{args.vae}-cfg-{args.cfg_scale}-seed-{args.seed}"
    sample_folder_dir = os.path.join(job_dir, folder_name)
    if local_rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples to {sample_folder_dir}")

    dist.barrier()  # Synchronize all processes

    all_samples = []  # List to store all generated samples from all GPUs

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,  # CPU性能分析
                    torch.profiler.ProfilerActivity.CUDA],  # GPU性能分析
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),  # 指定 TensorBoard 输出路径
        record_shapes=True,  # 记录张量的形状，用于分析内存占用
        profile_memory=True,  # 关闭内存跟踪，以减少性能开销
        with_stack=True,  # 获取堆栈信息
        with_flops=True,  # 计算FLOPs
        with_modules=True  # 记录模块信息
    ) as prof:
        # Calculate the number of iterations required for sampling
        total_samples = 50_000
        global_batch_size = n * dist.get_world_size()
        iterations = int(math.ceil(total_samples / global_batch_size))
        pbar = tqdm(range(iterations)) if local_rank == 0 else range(iterations)

        total = 0
        for _ in pbar:
            # Sample inputs:
            z = torch.randn(n, 4, latent_size, latent_size, device=device)
            y = torch.randint(0, args.num_classes, (n,), device=device)

            # Setup classifier-free guidance:
            if args.cfg_scale > 1.0:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                sample_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                sample_fn = model.forward

            # Sample images:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )

            if args.cfg_scale > 1.0:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            samples = vae.decode(samples / 0.18215).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Add the samples to the list to concatenate them later
            all_samples.append(samples)

            # Record profiler step
            prof.step()

        dist.barrier()  # Ensure all processes finish before the final step

        if local_rank == 0:
            # Concatenate all samples from all GPUs into one large image
            all_samples = np.concatenate(all_samples, axis=0)

            # 2 行 4 列的排列方式：8张图，分布在2行4列
            big_image_row_1 = np.concatenate(all_samples[:4], axis=1)  # 第一行
            big_image_row_2 = np.concatenate(all_samples[4:], axis=1)  # 第二行

            # 通过垂直拼接这两行，最终形成一张大图
            big_image = np.concatenate([big_image_row_1, big_image_row_2], axis=0)

            # Save the big image
            big_image_path = os.path.join(sample_folder_dir, "big_image.png")
            Image.fromarray(big_image).save(big_image_path)
            print(f"Big image saved to {big_image_path}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main(args)
