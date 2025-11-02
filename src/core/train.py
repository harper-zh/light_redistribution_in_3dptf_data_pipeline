import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorToImage(nn.Module):
    def __init__(self, input_dim=6, output_size=640):
        super().__init__()
        self.output_size = output_size

        # 全连接层扩展到 16×16×32
        self.dense = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 16 * 16 * 32),
            nn.BatchNorm1d(16 * 16 * 32),
            nn.ReLU(inplace=True)
        )

        # 上采样模块（使用PixelShuffle替代部分转置卷积）
        self.deconv = nn.Sequential(
            # 初始输入: (32, 16, 16)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 上采样到32x32
            nn.PixelShuffle(2),  # 64 -> (64//(2^2)=16, 16*2=32)
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 上采样到64x64
            nn.PixelShuffle(2),  # 32 -> (32//4=8, 32*2=64)
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 上采样到128x128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            # 上采样到256x256
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            
            # 上采样到512x512
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(4, 2, 3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            
            # 最终调整到640x640
            nn.Upsample(size=(output_size, output_size), mode='bilinear', align_corners=False),
            nn.Conv2d(2, 1, 1),  # 1x1卷积调整通道数
            nn.Sigmoid()
        )

    def forward(self, x):
        # 线性变换到 16×16×32
        x = self.dense(x)
        x = x.view(-1, 32, 16, 16)
        
        # 上采样生成图像
        x = self.deconv(x)
        return x  # (batch, 1, 640, 640)

# 测试代码
if __name__ == "__main__":
    model = VectorToImage()
    input_vec = torch.randn(2, 6)
    output_img = model(input_vec)
    
    print(f"Input shape: {input_vec.shape}")  # (2, 6)
    print(f"Output shape: {output_img.shape}")  # (2, 1, 640, 640)