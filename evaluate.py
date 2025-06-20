import numpy as np
from PIL import Image
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
import torch
import torchvision.transforms as T
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_image(img1_path, img2_path):
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    return img1, img2

def image_diff(img1, img2):
    img1_np = np.asarray(img1)
    img2_np = np.asarray(img2)
    diff = np.abs(img1_np.astype(np.int32) - img2_np.astype(np.int32)).mean()
    return diff

def metrics_score(img1, img2):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    
    transform = T.ToTensor()
    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)
    
    lpips_score = lpips(img1_tensor, img2_tensor).item()
    psnr_score = psnr(img1_tensor, img2_tensor).item()
    ssim_score = ssim(img1_tensor, img2_tensor).item()
    
    return lpips_score, psnr_score, ssim_score

def main():
    parser = argparse.ArgumentParser(description="Compare two images using LPIPS, PSNR, SSIM, and pixel diff.")
    parser.add_argument("image1", type=str, help="Path to the first image")
    parser.add_argument("image2", type=str, help="Path to the second image")
    args = parser.parse_args()

    img1, img2 = read_image(args.image1, args.image2)
    
    print(f"Comparing:\n - {args.image1}\n - {args.image2}")
    
    diff = image_diff(img1, img2)
    lpips_score, psnr_score, ssim_score = metrics_score(img1, img2)

    print("\n=== Image Comparison Metrics ===")
    print(f"Pixel-wise Mean Absolute Difference: {diff:.4f}")
    print(f"LPIPS (VGG):                        {lpips_score:.4f}")
    print(f"PSNR:                              {psnr_score:.2f} dB")
    print(f"SSIM:                              {ssim_score:.4f}")

if __name__ == "__main__":
    main()