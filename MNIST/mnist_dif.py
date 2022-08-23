import torch
from models_dif import SoftmaxWeight, DIFDensityEstimator

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

for i in range(10):
    p = target_samples.shape[-1]
    K = 60
    dif = DIFDensityEstimator(target_samples, K)
    dif.w = SoftmaxWeight(K,p, [512,512,256,256,128,128])
    dif.train(2000, 6000)
    filename = 'dif_mnist_best'+str(i)+'.sav'
    torch.save(dif, filename)
