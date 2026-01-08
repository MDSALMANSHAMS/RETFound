import torch
import torch.nn as nn
import torch.nn.functional as F
from models_vit import RETFound_mae


class SegmentationHead(nn.Module):
    def __init__(self, hidden_dim, num_classes, img_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.h = img_size // patch_size
        self.w = img_size // patch_size
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_classes, 1),
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.h, self.w, C).permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=self.patch_size, mode="bilinear", align_corners=False)
        return self.conv(x)


class RETFoundSegmentation(nn.Module):
    def __init__(self, img_size=512, patch_size=16, hidden_dim=1024, num_classes=2, drop_path=0.2):
        super().__init__()
        self.encoder = RETFound_mae(img_size=img_size, num_classes=num_classes,
                                    drop_path_rate=drop_path, global_pool=False)
        self.seg_head = SegmentationHead(hidden_dim, num_classes, img_size, patch_size)

    def forward(self, x):
        B = x.size(0)
        x = self.encoder.patch_embed(x)
        cls = self.encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.encoder.pos_embed
        x = self.encoder.pos_drop(x)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        return self.seg_head(x[:, 1:])
