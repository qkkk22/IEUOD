import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 用于从原始图像提取特征的卷积层
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        
    def forward(self, img, features):
        """
        计算原始输入图像和模型中间特征之间的余弦相似度损失
        Args:
            img: 原始输入图像, shape [B, 3, H, W]
            features: 模型的中间特征图, shape [B, C, H', W']
        Returns:
            loss: 余弦相似度损失
        """
        # 对原始图像应用卷积
        img_features = self.conv(img)  # [B, 64, H//2, W//2]
        
        # 调整特征图尺寸（如果必要）
        # if img_features.shape[2:] != features.shape[2:]:
        #     features = F.interpolate(features, size=img_features.shape[2:], mode='bilinear', align_corners=False)
        
        # 池化操作：平均池化 + 最大池化
        img_avg = F.adaptive_avg_pool2d(img_features, (1, 1))
        img_max = F.adaptive_max_pool2d(img_features, (1, 1))
        img_pool = img_avg + img_max
        
        feat_avg = F.adaptive_avg_pool2d(features, (1, 1))
        feat_max = F.adaptive_max_pool2d(features, (1, 1))
        feat_pool = feat_avg + feat_max
        
        # 展平
        img_flat = img_pool.view(img_pool.size(0), -1)  # [B, 64]
        feat_flat = feat_pool.view(feat_pool.size(0), -1)  # [B, C]
        
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(img_flat, feat_flat, dim=1)  # [B]
        loss = (1 - cos_sim).mean()  # 标量
        
        return loss