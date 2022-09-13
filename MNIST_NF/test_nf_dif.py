import torch
from models_nf import MAFLayer, MixedModelDensityEstimator, DIFDensityEstimatorLayer
from models_dif import SoftmaxWeight

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
p = target_samples.shape[-1]

C = 9
K = 50
structure = [[MAFLayer, [128,128,128]] for i in range(C)]
structure.append([DIFDensityEstimatorLayer, K])
initial_w = SoftmaxWeight(K,p, [128,128,128])
model = MixedModelDensityEstimator(target_samples, structure)
model.model[-1].w = initial_w
model.train(1000,6000)