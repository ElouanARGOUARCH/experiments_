import torch
import matplotlib.pyplot as plt
filename = 'dif_mnist_best.sav'
model = torch.load(filename, map_location=torch.device('cpu'))
print('train negative_log_likelihood = ' + str(model.loss_values[-1]))

import torchvision.datasets as datasets
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
images = mnist_testset.data.flatten(start_dim=1)
extracted = images.float()
test_samples = (extracted + torch.rand_like(extracted))/256
print(test_samples.shape)
print('test negative_log_likelihood = ' + str(model.loss(test_samples)))

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
plt.savefig('test_dif_best.png')
