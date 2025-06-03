import torch
import matplotlib.pyplot as plt
from utils.plots import hdr_plot_style, plot_compare, plot_determ_comp
from utils.helper import ObjectView
import os

def main(args):
    data_dir = f"./data/london/{args.num_users}"
    X_val = torch.load(f"{data_dir}/X_val.pt")
    y_val = torch.load(f"{data_dir}/y_val.pt")
    device = torch.device("cuda")
    checkpoint = torch.load(f'./result/ckpts/london_{args.num_users}/2000.pth')
    ddpm = checkpoint['ddpm']
    Loss = checkpoint['Loss']
        
    for k in range(10):
        X_val_k = X_val[X_val[:, k] == 1]
        y_val_k = y_val[X_val[:, k] == 1]
        cond_s = X_val_k.to(device)
        x_seq = ddpm.sample_seq(batch_size=len(X_val_k), cond=cond_s)
        y_fake_k = x_seq[-1].to("cpu")
        
        plt.figure(figsize=(7,3), dpi=300)
        plt.subplot(1,2,1)
        for i in range(len(y_val_k)):
            plt.plot(y_val_k[i])
        plt.title("real data")
        plt.ylim(0, 1)
        plt.subplot(1,2,2)
        for i in range(len(y_val_k)):
            plt.plot(y_fake_k[i])
        plt.title("fake data")
        plt.ylim(0, 1)
        
        save_dir = f'result/imgs/{args.num_users}'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/imgs_{k}.jpg')


if __name__ == '__main__':
    config = {
        'num_users':200
    }
    args = ObjectView(config)
    main(args)