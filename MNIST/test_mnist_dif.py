import torch
import matplotlib.pyplot as plt
list_test_log_likelihood = []
for i in range(10):
    filename = 'dif_mnist_second_model_0.sav'
    model = torch.load(filename, map_location=torch.device('cpu'))
    print('train negative_log_likelihood = ' + str(model.loss_values[-1]))

    def pre_process(x, lbda):
        return torch.logit(lbda*torch.ones_like(x) + x*(1-2*lbda))
    def inverse_pre_process(x, lbda):
        return torch.sigmoid((x- lbda*torch.ones_like(x))/(1-2*lbda))

    import torchvision.datasets as datasets
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    images = mnist_testset.data.flatten(start_dim=1)
    extracted = images.float()
    test_samples = pre_process((extracted + torch.rand_like(extracted))/256, 1e-6)
    print(test_samples.shape)
    loss = model.loss(test_samples)
    list_test_log_likelihood.append(loss)
    print('test negative_log_likelihood = ' + str(model.loss(test_samples)))

print(str(list_test_log_likelihood.mean()))
print(str(list_test_log_likelihood.std()))

with torch.no_grad():
    sample = model.sample_model(36)
    true_samples = inverse_pre_process(sample, 1e-6)
n_row =6
n_col =6
_, axs = plt.subplots(n_row, n_col, figsize=(24,24))
axs = axs.flatten()
for i, ax in enumerate(axs):
    ax.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    ax.imshow(true_samples[i].reshape(28,28).numpy(), vmin = 0, vmax=1, cmap = 'gray')
plt.savefig('test_dif_best.png')
