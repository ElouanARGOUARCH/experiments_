import torch
list_score_nll = []
list_score_dkl = []
number_runs = 11
lines = 256
columns = 197
import numpy
for i in range(number_runs):
    filename = 'euler_rnvp' + str(i) + '.sav'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rnvp = torch.load(filename,map_location=torch.device(device))
    list_score_nll.append(torch.tensor([rnvp.loss_values[-1]]))

print('mean score NLL = ' + str(torch.mean(torch.cat(list_score_nll), dim = 0).item()))
print('std NLL= ' + str(torch.std(torch.cat(list_score_nll), dim = 0).item()))