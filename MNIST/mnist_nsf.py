import torch
from models_nf import NeuralSplineFlow


import torchvision.datasets as datasets
import matplotlib.pyplot as plt
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
images = mnist_trainset.data.flatten(start_dim=1)
targets = mnist_trainset.targets

digit = 'all'
if digit != 'all':
    extracted = images[targets == digit].float()
else:
    extracted = images.float()
target_samples = extracted + torch.rand_like(extracted)/256

num_samples = target_samples.shape[0]
print('number of samples = ' + str(num_samples))
p = target_samples.shape[-1]

nsf = NeuralSplineFlow(target_samples, 20,128,3)

epochs = 1000
batch_size = 6000
nsf.train(epochs, batch_size)

filename = 'nsf_mnist.sav'
torch.save(nsf, filename)


