import torch
import torch.distributed as dist
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from pretrained_models.download import find_model
from models import DiT_models
import argparse
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
output_dir = '/work/sustcsc_11/DiT-SUSTCSC/output'
job_dir = os.path.join(output_dir, f"job_{job_id}")
log_dir = os.path.join(job_dir, f"log_{job_id}")
os.makedirs(log_dir, exist_ok=True)
sample_folder_dir = os.path.join(job_dir, "samples")
os.makedirs(sample_folder_dir, exist_ok=True)


# ===================================================

print(">> Running sample_02.py with job_id =", args.job_id, flush=True)

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False) # 禁用梯度计算
    
     # Debugging: Print before initializing DDP
    print("Initializing process group...")
     # 打印子进程中的环境变量（这一步才是有效的验证）
    print("\n===== 子进程环境变量 =====", flush=True)
    print(f"RANK: {os.environ.get('RANK')}", flush=True)
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}", flush=True)
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}", flush=True)
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}", flush=True)
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}", flush=True)
    print("==========================\n", flush=True)
    # Setup DDP:
    dist.init_process_group("nccl",init_method="env://")

      # Debugging: Print after initializing DDP
    print("Process group initialized.")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={args.seed}, world_size={dist.get_world_size()}.")

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

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    labels_per_gpu = 2
    # 每个GPU根据rank分配标签
    start_idx = rank * labels_per_gpu
    end_idx = (rank + 1) * labels_per_gpu
    # 获取每个GPU应该处理的标签
    assigned_labels = class_labels[start_idx:end_idx]

    # 输出分配情况
    print(f"Class labels: {class_labels}")
    print(f"Rank {rank} assigned labels: {assigned_labels}")
        
    iterations = 1
    pbar = tqdm(range(iterations)) if dist.get_rank() == 0 else range(iterations)


    start_time = time.time()
    for _ in pbar:
        # Create sampling noise:
        z = torch.randn(2, 4, latent_size, latent_size, device=device)
        y = torch.tensor(assigned_labels, device=device)

       # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * 2, device=device)
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
        
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample
        
        for i, sample in enumerate(samples):
            img_path = os.path.join(sample_folder_dir, f"rank{rank}_img{i}.png")
            save_image(sample, img_path, normalize=True, value_range=(-1, 1))

    # 同步所有进程
    dist.barrier()
    end_time = time.time()
    
    # 在rank == 0 拼接图像并保存
    if rank == 0:
        print(f"Rank {rank} is saving the final image.")
        # 使用 make_grid 拼接图像
        print(f"⏱️ sampling time: {end_time - start_time:.4f} seconds")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main(args)

    




