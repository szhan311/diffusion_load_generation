import torch
from utils.config import config_dataset_london
from tqdm import tqdm

X_train = torch.load("../data/london/tensor_small/y_tr.pt", weights_only=True)
X_test = torch.load("../data/london/tensor_small/y_val.pt", weights_only=True)
cond_train = torch.load("../data/london/tensor_small/X_tr.pt", weights_only=True)
cond_test = torch.load("../data/london/tensor_small/X_val.pt", weights_only=True)

def get_x_hat(checkpoint, cond, PV_base = None):
    ddpm = checkpoint['ddpm']
    device = ddpm.device
    cond = cond.to(device)
    if PV_base is not None:
        PV_base = PV_base.to(device)
    X_test_hat = ddpm.sample_seq(batch_size=len(cond), cond=cond, PV_base = PV_base)[-1]
    return X_test_hat.to("cpu")

model1 = "diff_base"
checkpoint1 = torch.load('../result/models/london/{}.pth'.format(model1))

X_test_hat_diffusion_base = []
cond = cond_test
for i in tqdm(range(20)):
    X_test_hat1 = get_x_hat(checkpoint1, cond)
    X_test_hat_diffusion_base.append(X_test_hat1)

X_test_hat_diffusion_base = torch.stack(X_test_hat_diffusion_base)
print(X_test_hat_diffusion_base.shape)

torch.save(X_test_hat_diffusion_base.permute(1, 0, 2), "../result/data/london/load_hat_diff_base.pt")
