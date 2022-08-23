import torch
from models_nf import MixedModelDensityEstimator, MAFLayer


import torchvision.datasets as datasets
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
images = mnist_trainset.data.flatten(start_dim=1)
targets = mnist_trainset.targets

digit = 'all'
if digit != 'all':
    extracted = images[targets == digit].float()
else:
    extracted = images.float()
target_samples = (extracted + torch.rand_like(extracted))/256

num_samples = target_samples.shape[0]
print('number of samples = ' + str(num_samples))
p = target_samples.shape[-1]

structure = [[MAFLayer,[256,256,256]] for i in range(10)]
rnvp = MixedModelDensityEstimator(target_samples, structure)
print(rnvp.compute_number_params())

epochs = 10000
batch_size = 6000
rnvp.train(epochs, batch_size)

filename = 'maf_mnist_best.sav'
torch.save(rnvp, filename)


