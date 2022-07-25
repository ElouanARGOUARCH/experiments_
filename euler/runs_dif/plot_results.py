import torch
import numpy
import matplotlib.pyplot as plt
lines = 256
columns = 197
number_runs = 20
for i in range(number_runs):
    filename = 'euler_dif' + str(i) + '.sav'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dif = torch.load(filename,map_location=torch.device(device))
    with torch.no_grad():
        grid = torch.cartesian_prod(torch.linspace(0, 1, lines), torch.linspace(0, 1, columns))
        density = torch.exp(dif.log_density(grid)).reshape(lines, columns).T
        figure = plt.figure()
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.imshow(torch.flip(torch.flip(density.T, [0, 1]), [0, 1]), extent=[0, columns, 0, lines])
        filename_png = 'euler_dif' + str(i) + '.png'
        figure.savefig(filename_png, bbox_inches='tight', pad_inches=0)
