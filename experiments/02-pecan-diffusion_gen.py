import torch
from utils.config import config_dataset
from tqdm import tqdm

X_train = torch.load("../data/Pecan Street Smart Meter Data (large) (tensor)/X_train.pt", weights_only=True)
X_test = torch.load("../data/Pecan Street Smart Meter Data (large) (tensor)/X_test.pt", weights_only=True)
cond_train = torch.load("../data/Pecan Street Smart Meter Data (large) (tensor)/cond_train.pt", weights_only=True)
cond_test = torch.load("../data/Pecan Street Smart Meter Data (large) (tensor)/cond_test.pt", weights_only=True)
PV_base_train = torch.load("../data/Pecan Street Smart Meter Data (large) (tensor)/PV_base_train.pt", weights_only=True)
PV_base_test = torch.load("../data/Pecan Street Smart Meter Data (large) (tensor)/PV_base_test.pt", weights_only=True)


num_class = config_dataset["num_class"]
# num_class = 5

def denorm(x):
    loads = torch.load("../data/Pecan Street Smart Meter Data (large) (tensor)/loads_raw.pt")
    lmin = []
    lmax = []
    for i in range(loads.shape[0]):
        load = loads[i]
        lmin.append(load.min())
        lmax.append(load.max())
    lmin = torch.stack(lmin)[:num_class]
    lmax = torch.stack(lmax)[:num_class]
    lmin = lmin.unsqueeze(1).repeat(1, int(x.shape[0]/num_class))
    lmax = lmax.unsqueeze(1).repeat(1, int(x.shape[0]/num_class))
    lmin = lmin.reshape((-1, 1, 1))
    lmax = lmax.reshape((-1, 1, 1))
    return (x + 1) * (lmax - lmin)/ 2 + lmin

def get_x_hat(checkpoint, cond, PV_base = None):
    ddpm = checkpoint['ddpm']
    device = ddpm.device
    cond = cond.to(device)
    if PV_base is not None:
        PV_base = PV_base.to(device)
    X_test_hat = ddpm.sample_seq(batch_size=len(cond), cond=cond, PV_base = PV_base)[-1]
    return X_test_hat.to("cpu")

model1 = "diff_base"
model2 = "diff_phy_80000"
checkpoint1 = torch.load('../result/models/pecan/{}.pth'.format(model1))
checkpoint2 = torch.load('../result/models/pecan/{}.pth'.format(model2))

X_test_hat_diffusion_base = []
X_test_hat_diffusion_phy = []
cond = cond_test
PV_base = PV_base_test
for i in tqdm(range(20)):
    X_test_hat1 = get_x_hat(checkpoint1, cond)
    X_test_hat2 = get_x_hat(checkpoint2, cond, PV_base)
    X_test_hat_diffusion_base.append(X_test_hat1)
    X_test_hat_diffusion_phy.append(X_test_hat2)

X_test_hat_diffusion_base = torch.stack(X_test_hat_diffusion_base)
X_test_hat_diffusion_phy = torch.stack(X_test_hat_diffusion_phy)

print(X_test_hat_diffusion_base.shape)
print(X_test_hat_diffusion_phy.shape)

torch.save(X_test_hat_diffusion_base.permute(1, 0, 2), "../result/data/pecan/load_hat_diff_base_norm.pt")
torch.save(X_test_hat_diffusion_phy.permute(1, 0, 2), "../result/data/pecan/load_hat_diff_phy_norm.pt")

torch.save(denorm(X_test_hat_diffusion_base.permute(1, 0, 2)), "../result/data/pecan/load_hat_diff_base.pt")
torch.save(denorm(X_test_hat_diffusion_phy.permute(1, 0, 2)), "../result/data/pecan/load_hat_diff_phy.pt")
