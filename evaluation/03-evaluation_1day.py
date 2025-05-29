import torch
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
from tqdm import tqdm

X_test = torch.load("./data/load_test.pt")
X_test_hat_gan = torch.load("./result/data/load_hat_gan.pt")
X_test_hat_vae = torch.load("./result/data/load_hat_vae.pt")
X_test_hat_nf = torch.load("./result/data/load_hat_nf.pt")
X_test_hat_diff_base = torch.load("./result/data/load_hat_diff_base.pt")
X_test_hat_diff_phy = torch.load("./result/data/load_hat_diff_phy.pt")


num_sample = 20
N = 100 # X_test.shape[0]
ymin = -7
ymax = 7
# for i in tqdm(range(N)):
#     plt.figure(figsize=(15, 2), dpi=300)
#     plt.subplot(1, 5, 1)
#     for j in range(num_sample):
#         plt.plot(X_test_hat_gan[i][j], color="gray")
#     plt.plot(X_test[i])
#     plt.ylim([ymin, ymax])
#     plt.xticks([])

#     # plt.title("(a)")
#     plt.subplot(1, 5, 2)
#     for j in range(num_sample):
#         plt.plot(X_test_hat_vae[i][j], color="gray")
#     plt.plot(X_test[i])
#     plt.ylim([ymin, ymax])
#     plt.xticks([])
#     plt.yticks([])  # Removes y-axis ticks
#     # plt.title("(b)")
#     plt.subplot(1, 5, 3)
#     for j in range(num_sample):
#         plt.plot(X_test_hat_nf[i][j], color="gray")
#     plt.plot(X_test[i])
#     plt.ylim([ymin, ymax])
#     plt.xticks([])
#     plt.yticks([])  # Removes y-axis ticks
#     # plt.title("(c)")
#     plt.subplot(1, 5, 4)
#     for j in range(num_sample):
#         plt.plot(X_test_hat_diff_base[i][j], color="gray")
#     plt.plot(X_test[i])
#     plt.ylim([ymin, ymax])
#     plt.xticks([])
#     plt.yticks([])  # Removes y-axis ticks
#     # plt.title("(s)")
#     plt.subplot(1, 5, 5)
#     for j in range(num_sample):
#         plt.plot(X_test_hat_diff_phy[i][j], color="gray")
#     plt.plot(X_test[i])
#     plt.ylim([ymin, ymax])
#     plt.yticks([])  # Removes y-axis ticks
#     plt.xticks([])
#     # plt.title("(e)")
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.savefig("./result/imgs/1day_{}.jpg".format(i))

# List of methods for looping
# List of methods for looping
num_users=5
methods = [
    ("GAN", X_test_hat_gan),
    ("VAE", X_test_hat_vae),
    ("NF", X_test_hat_nf),
    ("BDM", X_test_hat_diff_base),
    ("PDM", X_test_hat_diff_phy),
]

# Plot 5x5 grid
plt.figure(figsize=(10, 5), dpi=300)
user_indeces = [11,12,15,20,2]
for row in range(num_users):
    user_index = user_indeces[row]  # You can modify the user index selection here
    
    for col, (method_name, X_test_hat) in enumerate(methods):
        plt.subplot(num_users, 5, row * 5 + col + 1)
        
        # Plot synthetic samples
        for j in range(num_sample):
            plt.plot(X_test_hat[user_index][j], color="gray", alpha=0.5)
        
        # Plot actual load
        plt.plot(X_test[user_index], color="blue", linewidth=1)
        
        plt.ylim([ymin, ymax])
        plt.xticks([])
        plt.yticks([])
        
        # Add titles for the first row
        if row == 0:
            plt.title(method_name)

# Remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)

# Save the figure
plt.savefig("./result/imgs/5x5_load_profiles.jpg", bbox_inches='tight')
plt.show()