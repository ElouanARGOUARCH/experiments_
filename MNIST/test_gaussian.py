import torch
from torch import nn
class gaussianVDE(nn.Module):
    def __init__(self, target_samples):
        self.target_samples = target_samples
        self.mean = torch.mean(target_samples, dim = 0)
        self.cov = torch.cov(target_samples.T)

    def log_prob(self,x):
        return torch.distributions.MultivariateNormal(self.mean, self.cov).log_prob(x)

    def sample(self, num_sample):
        return torch.distributions.MultivariateNormal(self.mean, self.cov).sample([num_sample])

###MNIST###

import torchvision.datasets as datasets
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
images = mnist_trainset.data.flatten(start_dim=1).float()
temp = (images + torch.rand_like(images))/256

def pre_process(x, lbda):
    return torch.logit(lbda*torch.ones_like(x) + x*(1-2*lbda))

def inverse_pre_process(x, lbda):
    return torch.sigmoid((x- lbda*torch.ones_like(x))/(1-2*lbda))

lbda = 1e-6
target_samples = pre_process(temp, lbda)

gaussian = gaussianVDE(target_samples)
print(torch.mean(gaussian.log_prob(target_samples)))
import matplotlib.pyplot as plt
with torch.no_grad():
    sample = gaussian.sample(36)
    true_samples = inverse_pre_process(sample, 1e-6)
n_row =6
n_col =6
_, axs = plt.subplots(n_row, n_col, figsize=(24,24))
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    ax.imshow(true_samples[i].reshape(28,28).numpy(), vmin = 0, vmax=1, cmap = 'gray')
plt.savefig('test_gaussian_best.png')