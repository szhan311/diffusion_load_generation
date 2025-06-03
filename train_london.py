import torch
import torch.optim as optim
from models.DiffLoad.diffusion.layers import Attention
from matplotlib import pyplot as plt
from utils.helper import make_beta_schedule, EMA, ObjectView
from utils.plots import hdr_plot_style
hdr_plot_style()
from tqdm import tqdm
from models.DiffLoad.ddpm import DDPM1d
import datetime
import os

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def main(args):
    data_dir = f"./data/london/{args.num_users}"
    X_tr = torch.load(f"{data_dir}/X_tr.pt")
    X_val = torch.load(f"{data_dir}/X_val.pt")
    y_tr = torch.load(f"{data_dir}/y_tr.pt")
    y_val = torch.load(f"{data_dir}/y_val.pt")
    # Select betas
    n_steps = args.n_steps
    args.cond_dim = X_tr.shape[-1]
    betas = make_beta_schedule(schedule='linear', n_timesteps=n_steps, start=args.beta_start, end=args.beta_end)
    betas = betas.to(device)
    model = Attention(args)
    model = model.to(device)
    dataset = y_tr
    cond = X_tr
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    ddpm = DDPM1d(model, betas, n_steps, (args.input_dim,), loss_type='l2')
    # Create EMA model
    ema = EMA(args.ema_decay)
    ema.register(model)

    Loss = []
    for j in tqdm(range(args.epoch)):
        # X is a torch Variable
        permutation = torch.randperm(dataset.size()[0])
        for i in range(0, dataset.size()[0], args.batch_size):
            # Retrieve current batch 
            indices = permutation[i:i+args.batch_size]
            batch_x = dataset[indices].to(device)
            batch_cond = cond[indices].to(device)
            # Compute the loss.
            loss = ddpm(batch_x, batch_cond)
            # Before the backward pass, zero all of the network gradients
            optimizer.zero_grad()
            # Backward pass: compute gradient of the loss with respect to parameters
            loss.backward()
            # Perform gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            # Calling the step function to update the parameters
            optimizer.step()
            scheduler.step()
            # Update the exponential moving average
            ema.update(model)
            ddpm.model = model
        if (j+1) % 10 == 0:
            Loss.append(loss.item())
        if (j+1) % 10 == 0:
            print("loss: ", loss.item())
        if (j+1) % 500 == 0:
            checkpoint = {
                'config': args,
                'dataset': dataset,
                'cond': cond,
                'ddpm': ddpm,
                'Loss': Loss
            }
            save_dir = f"./result/ckpts/london_{args.num_users}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(checkpoint, f"{save_dir}/{j+1}.pth")
        


if __name__ == "__main__":
    config = {
    'n_steps': 500,
    'input_dim': 48,
    'hidden_dim': 2000,
    'nhead': 4,
    'cond_dim': 251,
    'epoch': 10000,
    'batch_size': 5000,
    'learning_rate': 1e-4,
    'lr_decay': 0.9,
    'lr_decay_step':1000,
    'ema_decay': 0.9,
    'beta_start': 1e-6,
    'beta_end': 2e-2,
    'loss_type': 'l2',
    'num_class': 50,
    'num_users': 500,
    }
    args = ObjectView(config)
    print(args.n_steps)
    
    main(args)