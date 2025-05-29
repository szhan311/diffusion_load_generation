import torch
import numpy as np
import torch.nn as nn
from utils.helper import extract
import torch.nn.functional as F

# For time series; use noise level
class DDPM1d(nn.Module):
    """
    Encapsulates the denoising diffusion process into a single function.

    Parameters:
    - betas: Tensor containing beta values for the diffusion process.
    - n_steps: Integer for the number of diffusion steps.
    - model: PyTorch model for noise reconstruction.
    - x_0: Tensor of data to be denoised.

    Returns:
    - loss: The computed loss after the denoising process.
    """
    def __init__(
        self,
        model,
        betas,
        n_steps,
        signal_size,
        loss_type="l1",
    ):
        super().__init__()
        self.clip_denoised = True
        self.betas = betas
        self.alphas = 1 - betas
        self.model = model
        self.device = betas.device
        self.n_steps = n_steps
        self.loss_type = loss_type
        self.signal_size = signal_size
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        # a = torch.tensor_full([1]).float()
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float().to(self.device), self.alphas_prod[:-1]], 0)
        self.alphas_prod_p_sqrt = self.alphas_prod_p.sqrt()
        self.sqrt_recip_alphas_cumprod = (1.0 / self.alphas_prod).sqrt()
        self.sqrt_alphas_cumprod_m1 = torch.sqrt(1.0 - self.alphas_prod) * (1.0 / self.alphas_prod).sqrt()
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.posterior_mean_coef_1 = (self.betas * self.alphas_bar_sqrt / (1.0 - self.alphas_prod))
        self.posterior_mean_coef_2 = ((1.0 - self.alphas_prod_p) * torch.sqrt(self.alphas) / (1.0 - self.alphas_prod))
        posterior_variance = betas * (1.0 - self.alphas_prod_p) / (1.0 - self.alphas_prod)
        self.posterior_log_variance_clipped = torch.log(torch.cat((posterior_variance[1].view(1, 1), posterior_variance[1:].view(-1, 1)), 0)).view(-1)

    def sample_continuous_noise_level(self, batch_size):
        t = np.random.choice(range(1, self.n_steps), size=batch_size)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.alphas_prod_p_sqrt[t-1].cpu().numpy(),
                self.alphas_prod_p_sqrt[t].cpu().numpy(),
                size=batch_size
            )).to(self.alphas_prod_p.device)
        return continuous_sqrt_alpha_cumprod.unsqueeze(-1)


    # Sample continuous noise level

    def forward(self, x, cond = None, PV_base = None):
        batch_size = x.shape[0]
        noise_level = self.sample_continuous_noise_level(batch_size)
        eps = torch.randn_like(x)
        # Diffuse the signal
        y_noisy = noise_level * x + (1 - noise_level**2).sqrt() * eps
        # Reconstruct the added noise
        eps_recon = self.model(y_noisy, noise_level, cond, PV_base)
        loss = torch.nn.L1Loss()(eps_recon, eps) if self.loss_type=="l1" else F.mse_loss(eps_recon, eps)
        return loss
    
    @torch.no_grad()
    def _p_sample(self, x, t, cond, PV_base, stable = False):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.alphas_prod_p_sqrt[t+1]]).repeat(batch_size, 1).to(self.device)
        eps_recon = self.model(x, noise_level, cond, PV_base) if stable is False else self.model.stable_comp(cond, PV_base)
        # eps_recon = self.model(x, noise_level, cond, PV_base)
        x_recon = self.sqrt_recip_alphas_cumprod[t] * x - self.sqrt_alphas_cumprod_m1[t] * eps_recon

        if self.clip_denoised and stable is False:
            x_recon.clamp_(-1.0, 1.0)

        model_mean = self.posterior_mean_coef_1[t] * x_recon + self.posterior_mean_coef_2[t] * x
        model_log_variance = self.posterior_log_variance_clipped[t]
        eps = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        out = model_mean + eps * (0.5 * model_log_variance).exp() if stable is False else model_mean
        return out
    
    @torch.no_grad()
    def sample_seq(self, batch_size, cond=None, PV_base = None, stable = False):
        if stable:
            cur_x = torch.zeros(batch_size,  *self.signal_size).to(self.device)
        else:
            cur_x = torch.randn(batch_size,  *self.signal_size).to(self.device)
        x_seq = [cur_x]

        for i in reversed(range(self.n_steps - 1)):
            cur_x = self._p_sample(cur_x, i, cond, PV_base, stable)
            x_seq.append(cur_x)

        return torch.stack(x_seq)