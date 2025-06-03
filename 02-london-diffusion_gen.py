import torch
import matplotlib.pyplot as plt
from utils.plots import hdr_plot_style, plot_compare, plot_determ_comp
from utils.helper import ObjectView
from models.DiffLoad.ddpm import DDPM1d

import os
from tqdm import tqdm

def get_x_hat(checkpoint, cond, PV_base = None):
    ddpm = checkpoint['ddpm']
    device = ddpm.device
    cond = cond.to(device)
    if PV_base is not None:
        PV_base = PV_base.to(device)
    X_test_hat = ddpm.sample_seq(batch_size=len(cond), cond=cond, PV_base = PV_base)[-1]
    return X_test_hat.to("cpu")

def main(args):
    data_dir = f"./data/london/{args.num_users}"
    X_test = torch.load(f"{data_dir}/X_test.pt", weights_only=True)
    y_test = torch.load(f"{data_dir}/y_test.pt", weights_only=True)
    print(len(y_test))
    # checkpoint = torch.load(f'./result/ckpts/london_{args.num_users}/{args.epochs}.pth', weights_only=False)
    
    # y_test = y_test * 2 - 1
    checkpoint = torch.load(f'./result/ckpts/london_{args.num_users}/{args.epochs}.pth', weights_only=False)
    print(f"length of test samples: {len(X_test)}")
    device = torch.device("cuda")
    
    ddpm = checkpoint['ddpm']
    model = ddpm.model
    betas = ddpm.betas
    n_steps = ddpm.n_steps
    ddpm = DDPM1d(model, betas, n_steps, ddpm.signal_size, loss_type='l2')
    loss = checkpoint['Loss']
    print(loss)
    
    X_test_hat_diffusion_base = []
    cond = X_test
    for i in tqdm(range(2)):
        X_test_hat1 = get_x_hat(checkpoint, cond)
        X_test_hat_diffusion_base.append(X_test_hat1)

    X_test_hat_diffusion_base = torch.stack(X_test_hat_diffusion_base)
    print(X_test_hat_diffusion_base.shape)
    torch.save(X_test_hat_diffusion_base.permute(1, 0, 2), "./result/data/london/load_hat_diff_base.pt")

        



if __name__ == '__main__':
    config = {
        'num_users':200,
        'epochs':2000
    }
    args = ObjectView(config)
    main(args)