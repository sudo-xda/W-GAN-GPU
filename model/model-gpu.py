import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletTransform(nn.Module):
    """
    2D Haar Wavelet Transform for edge extraction.
    Works fully on GPU using grouped convolutions.
    Args:
        wave (str): Currently supports only 'haar'
    """
    def __init__(self, wave: str = 'haar'):
        super(WaveletTransform, self).__init__()
        self.wave = wave.lower()
        if self.wave == 'haar':
            ll = torch.tensor([[0.5, 0.5],
                               [0.5, 0.5]])
            lh = torch.tensor([[0.5, 0.5],
                               [-0.5, -0.5]])
            hl = torch.tensor([[0.5, -0.5],
                               [0.5, -0.5]])
            hh = torch.tensor([[0.5, -0.5],
                               [-0.5, 0.5]])
        else:
            raise NotImplementedError(f"Wavelet {self.wave} not implemented yet.")
        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor [B, C, H, W]
        Returns:
            Tensor: Edge map [B, C, H/2, W/2]
        """
        B, C, H, W = x.shape
        filt = self.filters.repeat(C, 1, 1, 1)
        y = F.conv2d(x, filt, stride=2, padding=0, groups=C)
        y = y.view(B, C, 4, H // 2, W // 2)
        LL, LH, HL, HH = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        edges = LH + HL + HH
        return edges

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.wavelet = WaveletTransform()

    def forward(self, x):
        edge = self.wavelet(x)
        edge = F.interpolate(edge, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x + self.block(x) + edge

class Generator(nn.Module):
    def __init__(self, in_channels=1, num_res_blocks=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 9, padding=4),
            nn.PReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.output = nn.Conv2d(64, in_channels, 9, padding=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.res_blocks(x1)
        x3 = self.mid_conv(x2)
        x = x1 + x3
        x = self.upsample(x)
        return self.output(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1)
            # REMOVED: torch.sigmoid() - now returns logits for BCEWithLogitsLoss
        )

    def forward(self, x):
        batch_size = x.size(0)
        return self.net(x).view(batch_size)  # Return raw logits, not sigmoid
