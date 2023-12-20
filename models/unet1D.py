import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log as ln

# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, n_channels):
        super(PositionalEncoding, self).__init__()
        self.n_channels = n_channels
        self.linear_scale = 5e3

    def forward(self, noise_level):
        if len(noise_level.shape) > 1:
            noise_level = noise_level.squeeze(-1)
        half_dim = self.n_channels // 2
        exponents = torch.arange(half_dim, dtype=torch.float32).to(noise_level) / float(half_dim)
        exponents = 1e-4 ** exponents
        exponents = self.linear_scale * noise_level.unsqueeze(1) * exponents.unsqueeze(0)
        return torch.cat([exponents.sin(), exponents.cos()], dim=-1)

  
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.embedding = PositionalEncoding(out_channels)
        torch.nn.init.orthogonal_(self.conv.weight.data, gain=1)
        self.activation = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x, noise_level):

        x = self.conv(x)
        x = self.batchnorm(x)
        if noise_level is None:
            return self.activation(x)
        else:
            gamma = self.embedding(noise_level)
            return self.activation(x) * gamma.unsqueeze(-1)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x, noise_level):
        x = self.conv1(x, noise_level)
        x = self.conv2(x, noise_level)
        x = self.pool(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        # The upconv layer should halve the number of channels and double the spatial dimension
        self.upconv = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After concatenation with the skip connection, the number of channels will be in_channels // 2 + skip_channels
        # The first ConvBlock should take this into account
        self.conv1 = ConvBlock(in_channels // 2 + in_channels // 2, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, noise_level, skip):
        x = self.upconv(x)
        # Before concatenation, make sure that the dimensions are compatible
        if x.size(2) != skip.size(2):
            # Use interpolate to adjust the size if necessary
            x = F.interpolate(x, size=skip.size(2), mode='nearest')
        # Concatenate along the channel dimension
        x = torch.cat((x, skip), dim=1)
        x = self.conv1(x, noise_level)
        x = self.conv2(x, noise_level)
        return x

class ConditionalUNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super(ConditionalUNet1D, self).__init__()
        self.first_conv = ConvBlock(in_channels, base_channels)
        self.encoder1 = Encoder(base_channels, base_channels*2)
        self.encoder2 = Encoder(base_channels*2, base_channels*4)
        self.encoder3 = Encoder(base_channels*4, base_channels*8)
        self.bottleneck = Encoder(base_channels * 8, base_channels * 16)

        self.decoder3 = Decoder(base_channels * 16, base_channels * 8)
        self.decoder2 = Decoder(base_channels * 8, base_channels * 4)
        self.decoder1 = Decoder(base_channels * 4, base_channels * 2)
        self.decoder0 = Decoder(base_channels * 2, base_channels)
        self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, noise_level):
        # xc = torch.concat([x, conditional_vector], dim=-1)
        # x = self.mlp(xc)
        x = x.unsqueeze(-2)
        skip0 = self.first_conv(x, noise_level)
        skip1 = self.encoder1(skip0, noise_level)
        skip2 = self.encoder2(skip1, noise_level)
        skip3 = self.encoder3(skip2, noise_level)
        bottle = self.bottleneck(skip3, noise_level)
        x3 = self.decoder3(bottle, noise_level, skip3)
        x2 = self.decoder2(x3, noise_level, skip2)
        x1 = self.decoder1(x2, noise_level, skip1)
        x0 = self.decoder0(x1, noise_level, skip0)
        out = self.final_conv(x0)
        return out.squeeze(-2)