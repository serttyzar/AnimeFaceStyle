import torch.nn as nn



class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, num_residual=6):
        super().__init__()
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3),  # [B, 32, 128, 128]
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [B, 64, 64, 64]
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # [B, 128, 32, 32]
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(num_residual)]
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # [B, 64, 64, 64]
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # [B, 32, 128, 128]
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, 7, padding=3),  # [B, 3, 128, 128]
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # [B, 32, 64, 64]
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [B, 64, 32, 32]
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # [B, 128, 16, 16]
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, padding=1),  # [B, 256, 13, 13]
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, 4, padding=1)  # [B, 1, 10, 10]
        )

    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)