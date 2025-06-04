import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from utils.config import config_dataset_london
from tqdm import tqdm
from utils.helper import ObjectView

def get_x_hat(checkpoint, cond):
    ddpm = checkpoint['ddpm']
    device = ddpm.device
    cond = cond.to(device)
    X_test_hat = ddpm.sample_seq(batch_size=len(cond), cond=cond)[-1]
    return X_test_hat.to("cpu")
    
def main(args):
    # load conditional data
    data_dir = f"./data/london/{args.num_users}"
    cond_test = torch.load(f"{data_dir}/X_val.pt", weights_only=True)
    cond_test = cond_test.to(args.device)
    # load checkpoint
    checkpoint = torch.load(f'./result/ckpts/london_{args.num_users}/{args.epochs}.pth')

    # ddpm = checkpoint['ddpm']
    # x_seq = ddpm.sample_seq(batch_size=10, cond=cond_test[:10])
    # sampling
    print('Sampling...')
    X_test_hat_diffusion_base = []
    for i in tqdm(range(20)):
        
        X_test_hat1 = get_x_hat(checkpoint, cond_test[1000])
        X_test_hat_diffusion_base.append(X_test_hat1)
    X_test_hat_diffusion_base = torch.stack(X_test_hat_diffusion_base)
    print(X_test_hat_diffusion_base.shape)
    # save results
    torch.save(X_test_hat_diffusion_base.permute(1, 0, 2), "./result/data/london/load_hat_diff_base.pt")



if __name__ == '__main__':
    config = {
        'num_users': 500,
        'epochs': 1000,
        'device':'cuda'
    }
    args = ObjectView(config)
    main(args)