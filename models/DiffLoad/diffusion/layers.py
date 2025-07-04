from torch import nn
import torch
import torch.nn.functional as F

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


class CondModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.use_MLP = args.use_MLP
        self.lstm = nn.LSTM(args.input_dim,args.hidden_dim, num_layers=1, batch_first=True)
        self.cond_embedder = nn.Sequential(
            nn.Linear(args.cond_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh()
        )
        self.cond_embedder2 = nn.Sequential(
            nn.Linear(args.cond_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh()
        )
        self.cond_embedder3 = nn.Sequential(
            nn.Linear(args.cond_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, 7),
            nn.Softmax()
        )
        self.embedding = PositionalEncoding(args.hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_dim, nhead=4, dim_feedforward=args.hidden_dim
        )
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.linear1 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.linear2 = nn.Linear(args.hidden_dim, args.input_dim)

    def forward(self, x, noise, cond = None, PV_base = None):
        hid_enc, (_, _) = self.lstm(x)
        noise_emb = self.embedding(noise)
        hid_enc = hid_enc + noise_emb
        # if cond is not None:
        #     cond_emb = self.cond_embedder(cond)
        #     hid_enc = hid_enc + cond_emb
        
        trans_enc = self.trans_encoder(hid_enc) 
        # out = F.leaky_relu(self.linear1(trans_enc)) + self.cond_embedder2(cond)
        out = F.leaky_relu(self.linear1(trans_enc))
        
        if PV_base is not None:
            ## calculate solar gen
            coef = self.cond_embedder3(cond).unsqueeze(1) # [B,1, 7]
            solar_gen = torch.bmm(coef, PV_base)
            solar_gen = solar_gen.squeeze(1)
            return self.linear2(out) + solar_gen
        else: 
            return self.linear2(out)
        # out = F.leaky_relu(self.linear1(trans_enc)) + self.cond_embedder2(cond)

        

class CondModel_v2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_MLP = args.use_MLP
        self.lstm = nn.LSTM(args.input_dim,args.hidden_dim, num_layers=1, batch_first=True)
        self.cond_embedder = nn.Sequential(
            nn.Linear(args.cond_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh()
        )
        self.cond_embedder2 = nn.Sequential(
            nn.Linear(args.cond_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
             nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
             nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh()
        )
        self.cond_embedder3 = nn.Sequential(
            nn.Linear(args.cond_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Softmax()
        )
        self.solar_embedder = nn.Sequential(
            nn.Linear(args.input_dim * 7, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh()
        )
        self.init_weights()
        self.embedding = PositionalEncoding(args.hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_dim, nhead=4, dim_feedforward=args.hidden_dim
        )
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.linear1 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.linear2 = nn.Linear(args.hidden_dim, args.input_dim)
    
    def init_weights(self):
        for module in self.cond_embedder3:
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0)  # Set weights to 0
                nn.init.constant_(module.bias, 0)  # Set biases to 0

    def forward(self, x, noise, cond = None, PV_base = None):
        hid_enc, (_, _) = self.lstm(x)
        noise_emb = self.embedding(noise)
        hid_enc = hid_enc + noise_emb
        # if cond is not None:
        #     cond_emb = self.cond_embedder(cond)
        #     hid_enc = hid_enc + cond_emb
            
        if PV_base is not None:
            solar_gen = PV_base.reshape(-1, PV_base.shape[-1] * PV_base.shape[-2]) 
            solar_enc = self.solar_embedder(solar_gen)
            hid_enc = hid_enc + solar_enc * self.cond_embedder(cond)

        trans_enc = self.trans_encoder(hid_enc) 

        out = F.leaky_relu(self.linear1(trans_enc))
        if self.use_MLP: out = out + self.cond_embedder2(cond)
        return self.linear2(out)
    


# conditional model without PV
class Attention(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.lstm = nn.LSTM(args.input_dim,args.hidden_dim, num_layers=1, batch_first=True)
        self.cond_embedder = nn.Sequential(
            nn.Linear(args.cond_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Tanh()
        )
        # self.cond_embedder2 = nn.Sequential(
        #     nn.Linear(args.cond_dim, args.hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(args.hidden_dim, args.hidden_dim),
        #     nn.Tanh()
        # )
        # self.cond_embedder3 = nn.Sequential(
        #     nn.Linear(args.cond_dim, args.hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(args.hidden_dim, 7),
        #     nn.Softmax()
        # )
        self.embedding = PositionalEncoding(args.hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_dim, nhead=4, dim_feedforward=args.hidden_dim
        )
        self.trans_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.linear1 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.linear2 = nn.Linear(args.hidden_dim, args.input_dim)

    def forward(self, x, noise, cond = None, PV_base = None):
        hid_enc, (_, _) = self.lstm(x)
        noise_emb = self.embedding(noise)
        hid_enc = hid_enc + noise_emb
        if cond is not None:
            cond_emb = self.cond_embedder(cond)
            hid_enc = hid_enc + cond_emb
        
        trans_enc = self.trans_encoder(hid_enc) 
        out = F.leaky_relu(self.linear1(trans_enc))
        
        return self.linear2(out)