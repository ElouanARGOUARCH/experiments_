import torch
import matplotlib.pyplot as plt
filename = 'dif_mnist.sav'
model = torch.load(filename)
print(model.loss_values[-1])
with torch.no_grad():
    sample = model.sample_model(50)
n_row = 5
n_col = 10
_, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.imshow(sample[i].reshape(28,28).numpy())
plt.savefig('test.png')
