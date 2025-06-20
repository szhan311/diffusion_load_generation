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
            # x_recon.clamp_(-1.0, 1.0)
            x_recon.clamp_(0.0, 1.0)

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
    
    @torch.no_grad()
    def sample_ddim(
        self,
        batch_size,
        cond=None,
        PV_base=None,
        ddim_eta: float = 0.0,          # η = 0 → deterministic DDIM
        ddim_timesteps: list | None = None,
        return_intermediates: bool = False
    ):
        """
        DDIM sampling method for faster generation.
        
        Args:
            batch_size: Number of samples to generate
            cond: Conditional information for the model
            PV_base: Additional conditioning parameter
            ddim_eta: Stochasticity parameter (0 = deterministic, 1 = DDPM-like)
            ddim_timesteps: Custom timestep schedule. If None, uses uniform spacing
            return_intermediates: Whether to return all intermediate states
        
        Returns:
            Generated samples (and optionally intermediate states)
        """
        
        # Set up timestep schedule
        if ddim_timesteps is None:
            # Default: uniform spacing with fewer steps for faster sampling
            skip = self.n_steps // 50  # Use 50 steps by default (adjust as needed)
            ddim_timesteps = list(range(0, self.n_steps, skip))
            ddim_timesteps = ddim_timesteps[::-1]  # Reverse for denoising direction
        
        # Start from pure noise
        x = torch.randn(batch_size, *self.signal_size).to(self.device)
        
        if return_intermediates:
            intermediates = [x.clone()]
        
        # DDIM sampling loop
        for i, t in enumerate(ddim_timesteps[:-1]):
            t_next = ddim_timesteps[i + 1]
            
            # Get noise level for current timestep
            noise_level = torch.FloatTensor([self.alphas_prod_p_sqrt[t+1]]).repeat(batch_size, 1).to(self.device)
            
            # Predict noise using the model
            eps_pred = self.model(x, noise_level, cond, PV_base)
            
            # Get alpha values for current and next timesteps
            alpha_prod_t = self.alphas_prod[t]
            alpha_prod_t_next = self.alphas_prod[t_next] if t_next >= 0 else torch.tensor(1.0).to(self.device)
            
            # Predict x_0 from current x_t and predicted noise
            sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
            sqrt_one_minus_alpha_prod_t = torch.sqrt(1.0 - alpha_prod_t)
            
            pred_x0 = (x - sqrt_one_minus_alpha_prod_t * eps_pred) / sqrt_alpha_prod_t
            
            # Clip predicted x_0 if specified
            if self.clip_denoised:
                pred_x0 = pred_x0.clamp(-1.0, 1.0)
            
            # Compute variance for stochastic sampling
            if ddim_eta > 0 and t_next > 0:
                # Variance schedule for DDIM
                sigma_t = ddim_eta * torch.sqrt(
                    (1.0 - alpha_prod_t_next) / (1.0 - alpha_prod_t) * 
                    (1.0 - alpha_prod_t / alpha_prod_t_next)
                )
            else:
                sigma_t = 0.0
            
            # Compute the coefficient for the predicted noise direction
            sqrt_alpha_prod_t_next = torch.sqrt(alpha_prod_t_next)
            sqrt_one_minus_alpha_prod_t_next = torch.sqrt(1.0 - alpha_prod_t_next - sigma_t**2)
            
            # DDIM update step
            x = (
                sqrt_alpha_prod_t_next * pred_x0 +
                sqrt_one_minus_alpha_prod_t_next * eps_pred
            )
            
            # Add stochastic noise if eta > 0
            if ddim_eta > 0 and t_next > 0:
                noise = torch.randn_like(x)
                x = x + sigma_t * noise
            
            if return_intermediates:
                intermediates.append(x.clone())
        
        if return_intermediates:
            return x, torch.stack(intermediates)
        else:
            return x