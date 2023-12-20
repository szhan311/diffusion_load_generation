import torch.nn.functional as F
from torch import nn
import torch
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

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)

    def forward(self, x, y):
        out = self.lin(x)
        gamma = y
        out = gamma * out
        return out
class ConditionalConv1d(nn.Module):
    def __init__(self, **kwargs):
        super(ConditionalConv1d, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        self.embedding = PositionalEncoding(kwargs.get('out_channels'))
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x, y):
        out = self.conv1d(x);
        gamma = self.embedding(y)
        return out * gamma.unsqueeze(-1)
class ConditionalModel(nn.Module):
    def __init__(self,input_dim):
        super(ConditionalModel, self).__init__()
        self.conv1 = ConditionalConv1d(in_channels=1, out_channels=64, kernel_size=16, padding=8)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = ConditionalConv1d(in_channels=64, out_channels=64, kernel_size=16, padding=16, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=16, padding=16, dilation=2)
        self.lin1 = ConditionalLinear(input_dim+5, 512)
        self.lin2 = nn.Linear(512, input_dim)
    
    def forward(self, x, y):
        x = x.unsqueeze(1)
        x = F.leaky_relu(self.bn1(self.conv1(x, y)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x, y)), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.squeeze(1)
        x = F.softplus(self.lin1(x, y))
        return self.lin2(x)

class Clinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()  # Corrected super call
        self.linear = nn.Linear(input_dim, output_dim)
        self.embedding = PositionalEncoding(input_dim)  # Make sure PositionalEncoding is defined

    def forward(self, x, y):
        encoder = self.embedding(y)
        x = x + encoder
        return F.leaky_relu(self.linear(x))  # Pass x to self.linear


class Attention(nn.Module):
    def __init__(self,input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim, num_layers=1, batch_first=True)
        self.embedding = PositionalEncoding(hidden_dim)  # Make sure PositionalEncoding is defined
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim
        )
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.output_projector = self.conv1d_with_init(hidden_dim, 1, 1)
        self.linear = nn.Linear(hidden_dim, 96)

    def conv1d_with_init(self, in_channels, out_channels, kernel_size):
        conv1d_layer = nn.Conv1d(in_channels, out_channels, kernel_size)
        nn.init.kaiming_normal_(conv1d_layer.weight)
        return conv1d_layer

    
    
    def forward(self, x, y):
        # x = x.unsqueeze(-1)
        # y = y.unsqueeze(-1)
        hid_enc, (_, _) = self.lstm(x)  # (B, L, hidden_dim)
        time_emb = self.embedding(y)
        hid_enc = hid_enc + time_emb
        trans_enc = self.trans_encoder(hid_enc)
        # out = self.output_projector(trans_enc).permute(0, 2, 1)
        return self.linear(trans_enc)
