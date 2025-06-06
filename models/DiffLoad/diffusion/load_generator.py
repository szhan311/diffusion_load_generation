import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------- small helpers -------------------------------------------------- #
class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, depth: int = 3):
        super().__init__()
        layers = []
        for i in range(depth):
            layers += [
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.GELU()
            ]
        self.main = nn.Sequential(*layers)
        self.skip = (
            nn.Identity()
            if in_dim == hidden_dim
            else nn.Linear(in_dim, hidden_dim, bias=False)
        )

    def forward(self, x):
        return self.main(x) + self.skip(x)


class PositionalEncoding(nn.Module):
    """Classic sin–cos position enc. returned as (B, L, D)."""
    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-torch.log(torch.tensor(10_000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)                # (max_len, D)

    def forward(self, length: int, batch: int) -> torch.Tensor:
        # -> (B, L, D)  (pe is not trainable, so repeat is cheap)
        return self.pe[:length].unsqueeze(0).repeat(batch, 1, 1)


# ---------- main backbone -------------------------------------------------- #
class DiffusionLoadGenerator(nn.Module):
    """
    • local temporal encoding via LSTM
    • noise-level + sine–cos positional encoding
    • optional condition & PV embeddings
    • Transformer refinement
    • head to original input dimension
    """
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim

        # 1. LSTM
        self.lstm = nn.LSTM(
            input_size=args.input_dim,
            hidden_size=args.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # 2. condition & PV embedders
        self.cond_embedder = ResidualMLP(args.cond_dim, args.hidden_dim, depth=4)
        self.pv_embedder   = (
            ResidualMLP(args.input_dim, args.hidden_dim, depth=2)
            if getattr(args, "use_pv_embedder", False) else None
        )

        # 3. positional encoding for sequence + scalar noise encoding
        self.pos_enc = PositionalEncoding(args.hidden_dim)
        self.noise_mlp = ResidualMLP(1, args.hidden_dim, depth=2)

        # 4. Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_dim,
            nhead=args.nhead,
            dim_feedforward=args.ff_dim,
            dropout=args.dropout,
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=args.n_layers)

        # 5. projection head
        self.head = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(args.hidden_dim),
            nn.Linear(args.hidden_dim, args.input_dim),
        )

    # --------------------------------------------------------------------- #
    # public
    def forward(
        self,
        x: torch.Tensor,                   # (B, L, in_dim)
        noise: torch.Tensor,               # flexible (see below)
        cond: Optional[torch.Tensor] = None,   # (B, cond_dim)
        PV_base: Optional[torch.Tensor] = None # (B, L, in_dim) or (B, in_dim)
    ):
        B, L, _ = x.shape

        # 0) make noise embedding the right shape
        noise = self._expand_noise(noise, B, L)            # (B, L, 1)
        noise_emb = self.noise_mlp(noise)                  # (B, L, H)

        # 1) LSTM
        h, _ = self.lstm(x)                                # (B, L, H)

        # 2) add position + noise
        h = h + self.pos_enc(L, B) + noise_emb

        # 3) condition embedding
        if cond is not None:
            cond_emb = self.cond_embedder(cond).unsqueeze(1)  # (B,1,H)
            h = h + cond_emb

        # 4) PV embedding
        if PV_base is not None:
            if PV_base.dim() == 2:                          # (B, in_dim)
                PV_base = PV_base.unsqueeze(1).expand(-1, L, -1)
            if self.pv_embedder is not None:
                pv_emb = self.pv_embedder(PV_base)          # (B,L,H)
                h = h + pv_emb
            else:
                h = h + PV_base[..., : self.hidden_dim]     # naive add

        # 5) Transformer
        h = self.transformer(h)                             # (B, L, H)

        # 6) projection
        return self.head(h)                                 # (B, L, in_dim)

    # ------------------------------------------------------------------ #
    # private utilities
    def _expand_noise(self, noise: torch.Tensor, B: int, L: int):
        """
        Accept a variety of noise shapes and expand to (B, L, 1)
        """
        if noise.dim() == 1:                # (B,)
            noise = noise[:, None, None]    # -> (B,1,1)
        elif noise.dim() == 2:              # (B,1) or (B,?)
            noise = noise[:, None]          # -> (B,1,?)
        elif noise.dim() == 3 and noise.size(1) == 1:  # (B,1,?)
            pass
        elif noise.dim() == 3 and noise.size(1) == L:  # already (B,L,?)
            return noise[..., :1] if noise.size(-1) > 1 else noise
        else:
            raise ValueError("Unrecognised noise shape")

        # now (B,1,?) – keep only first channel, repeat along L
        return noise[..., :1].expand(B, L, 1)
