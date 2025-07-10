import torch
from models import DiT_models  # 确保这一行导入你的 DiT 模型构建函数

# 1. 初始化模型（替换为你实际使用的模型名）
model = DiT_models['DiT-XL/2'](
    input_size=64,          # 或512等，确保和训练/推理时一致
    in_channels=4,
    num_classes=1000,
    learn_sigma=True
).cuda()  # 如果有 GPU，可加 .cuda()

# 2. 设置为 eval 模式（可选）
model.eval()

# 3. 构造随机输入（与模型一致的 shape）
N = 1  # batch size
dummy_input = torch.randn(N, 4, 64, 64).cuda()          # latent space image
dummy_t = torch.randint(0, 1000, (N,), dtype=torch.long).cuda()  # timestep
dummy_y = torch.randint(0, 1000, (N,), dtype=torch.long).cuda()  # class label

# 4. 前向传播一次（必要，触发 lazy init）
with torch.no_grad():
    _ = model(dummy_input, dummy_t, dummy_y)

# 5. 保存当前模型的参数
torch.save(model.state_dict(), "DiT_flash_initialized_512.pth")
print("✅ 权重已保存为 DiT_flash_initialized_512.pth")
