import torch
import matplotlib.pyplot as plt
filename = 'dif_mnist0.sav'
model = torch.load(filename, map_location=torch.device('cpu'))
print(model.loss_values[-1])
print(model.w.network_dimensions)
with torch.no_grad():
    sample = model.sample_model(36)
    print(torch.max(sample))
n_row =6
n_col =6
_, axs = plt.subplots(n_row, n_col, figsize=(24,24))
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    ax.imshow(sample[i].reshape(28,28).numpy(), vmin = 0, vmax=1, cmap = 'gray')
plt.savefig('test_0.png')
