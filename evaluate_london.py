import torch
import matplotlib.pyplot as plt
from utils.plots import hdr_plot_style, plot_compare, plot_determ_comp
from utils.helper import ObjectView
import os
from tqdm import tqdm

def main(args):
    data_dir = f"./data/london/{args.num_users}"
    X_val = torch.load(f"{data_dir}/X_val.pt", weights_only=True)
    y_val = torch.load(f"{data_dir}/y_val.pt", weights_only=True)
    print(f"length of test samples: {len(X_val)}")
    device = torch.device("cuda")
    checkpoint = torch.load(f'./result/ckpts/london_{args.num_users}/{args.epochs}.pth')
    ddpm = checkpoint['ddpm']
    
    # sample for 10 users
    for k in range(10):
        X_val_k = X_val[X_val[:, k] == 1]
        y_val_k = y_val[X_val[:, k] == 1]
        cond_s = X_val_k.to(device)
        
        x_seq = ddpm.sample_seq(batch_size=len(X_val_k), cond=cond_s)
        y_fake_k = x_seq[-1].to("cpu")
        # y_fake_k = ddpm.sample_ddim(batch_size=len(X_val_k), cond=cond_s, ddim_eta=0.7).to("cpu")
        
        
        plt.figure(figsize=(7,3), dpi=300)
        plt.subplot(1,2,1)
        for i in range(len(y_val_k)):
            plt.plot(y_val_k[i])
        plt.title("real data")
        plt.ylim(-1, 1)
        plt.subplot(1,2,2)
        for i in range(len(y_val_k)):
            plt.plot(y_fake_k[i])
        plt.title("fake data")
        plt.ylim(-1, 1)
        
        save_dir = f'result/imgs/{args.num_users}'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/imgs_{k}.jpg')


if __name__ == '__main__':
    config = {
        'num_users':200,
        'epochs':1000
    }
    args = ObjectView(config)
    main(args)