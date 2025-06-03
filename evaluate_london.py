import torch
import matplotlib.pyplot as plt
from utils.plots import hdr_plot_style, plot_compare, plot_determ_comp
from utils.helper import ObjectView
import os

def main(args):
    data_dir = f"./data/london/{args.num_users}"
    X_val = torch.load(f"{data_dir}/X_val.pt")
    y_val = torch.load(f"{data_dir}/y_val.pt")
    for k in range(10):
        y_val_0 = y_val[X_val[:, k] == 1]
        print(y_val_0.shape)
        print(y_val_0.max(), y_val_0.min())
        plt.figure(figsize=(3,3), dpi=300)
        for i in range(len(y_val_0)):
            plt.plot(y_val_0[i])
        plt.title("real data")
        save_dir = f'result/imgs/{args.num_users}'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/real_{k}.jpg')


if __name__ == '__main__':
    config = {
        'num_users':200
    }
    args = ObjectView(config)
    main(args)