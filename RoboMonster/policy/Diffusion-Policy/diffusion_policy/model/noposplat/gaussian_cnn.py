import torch
import torch.nn as nn

class GaussianConvEncoder(nn.Module):
    def __init__(self, in_channels, pre_fuse):
        super().__init__()
        if pre_fuse:
            self.gau_conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=1, padding=0),
                nn.GroupNorm(num_groups=2, num_channels=32),
                nn.ReLU(),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=2, num_channels=16),
                nn.ReLU(),
                nn.Conv2d(16, 3, kernel_size=1, padding=0),
                nn.ReLU(),
            )
            chs_in = 6
        else:
            chs_in = in_channels
        self.head = nn.Sequential(
            nn.Conv2d(chs_in, 32, kernel_size=1, padding=0),
            nn.GroupNorm(num_groups=2, num_channels=32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=2, num_channels=16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=1, padding=0),
            nn.ReLU(),
        )
        self.pre_fuse = pre_fuse

    def forward(self, gau, img):
        """
        gau: [B, T, N, C', H, W]
        img: [B, T, N, 3, H, W]
        return: [T, B, N, 3, H, W]
        """
        B, T, N, C, H, W = gau.shape
        if self.pre_fuse:
            gau = gau.reshape(B * T * N, C, H, W)       # reshape to [T*B*N, C', H, W]
            img = img.reshape(B * T * N, 3, H, W)
            gau = self.gau_conv(gau)
            feat = torch.cat([gau, img], dim=1)
        else:
            feat = gau.reshape(B * T * N, C, H, W)        

        feat = self.head(feat)                           # [T*B*N, 3, H, W]
        feat = feat.reshape(B, T, N, 3, H, W)            # reshape back

        return feat
