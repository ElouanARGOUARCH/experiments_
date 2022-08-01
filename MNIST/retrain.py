import torch

model = torch.load("dif_mnist.sav")
print(model.loss_values[-1])
epochs = 1000
batch_size = 6000
model.train(epochs, batch_size)
filename = 'dif_mnist.sav'
torch.save(model, filename)