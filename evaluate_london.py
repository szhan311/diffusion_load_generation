import torch
import matplotlib.pyplot as plt
from utils.plots import hdr_plot_style, plot_compare, plot_determ_comp
from utils.helper import ObjectView
import os
from tqdm import tqdm

def main(args):
    data_dir = f"./data/london/{args.num_users}"
    X_val = torch.load(f"{data_dir}/X_val.pt")
    y_val = torch.load(f"{data_dir}/y_val.pt")
    print(f"length of test samples: {len(X_val)}")
    device = torch.device("cuda")
    # Use safe_globals to load checkpoint securely
    from utils.helper import ObjectView
    checkpoint = torch.load(f'./result/ckpts/london_{args.num_users}/{args.epochs}.pth', weights_only=False)
    print(checkpoint['config'].n_steps)
    print(checkpoint['config'].epoch)
    print(checkpoint['config'].hidden_dim)
    print(checkpoint['config'].learning_rate)
    print(checkpoint['config'].lr_decay)
    print(checkpoint['config'].lr_decay_step)
    print(checkpoint['config'].ema_decay)
    ddpm = checkpoint['ddpm']
    
    # Sample for 10 users
    for k in tqdm(range(10)):
        X_val_k = X_val[X_val[:, k] == 1]
        y_val_k = y_val[X_val[:, k] == 1]
        cond_s = X_val_k.to(device)
        
        x_seq = ddpm.sample_seq(batch_size=len(X_val_k), cond=cond_s)
        y_fake_k = x_seq[-1].to("cpu")
        # y_fake_k = ddpm.sample_ddim(batch_size=len(X_val_k), cond=cond_s, ddim_eta=0.7).to("cpu")
        
        # Compute mean and variance across samples for each time step
        real_mean = torch.mean(y_val_k, dim=0)
        real_var = torch.var(y_val_k, dim=0)
        fake_mean = torch.mean(y_fake_k, dim=0)
        fake_var = torch.var(y_fake_k, dim=0)
        
        # Plotting
        plt.figure(figsize=(7, 5), dpi=300)
        
        # Subplot 1: Real data individual samples
        plt.subplot(2, 2, 1)
        for i in range(len(y_val_k)):
            plt.plot(y_val_k[i], alpha=0.5)
        plt.title("Real Data")
        plt.ylim(0, 1)
        plt.xlabel("Time")
        plt.ylabel("Value")
        
        # Subplot 2: Fake data individual samples
        plt.subplot(2, 2, 2)
        for i in range(len(y_fake_k)):
            plt.plot(y_fake_k[i], alpha=0.5)
        plt.title("Fake Data")
        plt.ylim(0, 1)
        plt.xlabel("Time")
        plt.ylabel("Value")
        
        # Subplot 3: Mean across time
        plt.subplot(2, 2, 3)
        plt.plot(real_mean, label="Real Mean", color="blue")
        plt.plot(fake_mean, label="Fake Mean", color="orange")
        plt.title("Mean Across Time")
        plt.ylim(0, 1)
        plt.xlabel("Time")
        plt.ylabel("Mean")
        plt.legend()
        
        # Subplot 4: Variance across time
        plt.subplot(2, 2, 4)
        plt.plot(real_var, label="Real Variance", color="blue")
        plt.plot(fake_var, label="Fake Variance", color="orange")
        plt.title("Variance Across Time")
        plt.xlabel("Time")
        plt.ylabel("Variance")
        plt.legend()
        
        # Save plot
        save_dir = f'result/imgs/{args.num_users}'
        os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(f'{save_dir}/imgs_{k}.jpg')
        plt.close()  # Close figure to free memory

if __name__ == '__main__':
    config = {
        'num_users': 200,
        'epochs': 2000
    }
    args = ObjectView(config)
    main(args)