import numpy as np
import torch
from matplotlib import image
import matplotlib.pyplot as plt
torch.manual_seed(0)
number_runs = 20

from models_nf import NeuralSplineFlow
rgb = image.imread("euler.jpg")
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
grey = torch.tensor(rgb2gray(rgb))
list_score_nll = []

for i in range(number_runs):
    #Sample data according to image
    vector_density = grey.flatten()
    vector_density = vector_density/torch.sum(vector_density)
    lines, columns = grey.shape

    num_samples = 300000
    cat = torch.distributions.Categorical(probs = vector_density)
    categorical_samples = cat.sample([num_samples])
    target_samples = torch.cat([((categorical_samples // columns + torch.rand(num_samples)) / lines).unsqueeze(-1),((categorical_samples % columns + torch.rand(num_samples)) / columns).unsqueeze(-1)],dim=-1)

    nsf = NeuralSplineFlow(target_samples, 10,32,3)

    epochs = 1000
    batch_size = 30000
    nsf.train(epochs, batch_size)
    list_score_nll.append(torch.tensor([nsf.loss_values[-1]]))
    with torch.no_grad():
        grid = torch.cartesian_prod(torch.linspace(0, 1, lines), torch.linspace(0, 1, columns))
        density = torch.exp(nsf.model.log_prob(grid)).reshape(lines, columns).T
        figure = plt.figure()
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.imshow(torch.flip(torch.flip(density.T, [0, 1]), [0, 1]), extent=[0, columns, 0, lines])
        filename_png = 'runs_nsf/euler_nsf' + str(i) + '.png'
        figure.savefig(filename_png, bbox_inches = 'tight', pad_inches = 0 )

    filename = 'runs_nsf/euler_nsf' + str(i) + '.sav'
    torch.save(nsf.state_dict(), filename)

f = open('runs_nsf/score.txt', 'w')
f.write('mean score NLL = ' + str(torch.mean(torch.cat(list_score_nll), dim = 0).item()) +'\n')
f.write('std NLL= ' + str(torch.std(torch.cat(list_score_nll), dim = 0).item()))
