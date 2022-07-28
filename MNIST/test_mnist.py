import torch
import matplotlib.pyplot as plt
filename = 'dif_mnist.sav'
model = torch.load(filename)
print(model.loss_values[-1])
with torch.no_grad():
    sample = model.sample_model(1)
    fig = plt.figure()
    plt.imshow(sample.reshape(28,28).numpy())
    plt.show()