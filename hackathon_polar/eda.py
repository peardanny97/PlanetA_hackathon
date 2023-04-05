import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import numpy as np


elev_data = loadmat("./data/CryoSat2_data", appendmat=True)
hydpot_data = loadmat("./data/hydropotential", appendmat=True)
idx_SGL_data = loadmat("./data/idx_SGL", appendmat=True)
idx_nSGL_data = loadmat("./data/idx_nSGL", appendmat=True)


elev = torch.tensor(elev_data['elev'])
hydpot = torch.tensor(hydpot_data['hydpot'])
idx_SGL = torch.tensor(idx_SGL_data['idx_SGL'])
idx_nSGL = torch.tensor(idx_nSGL_data['idx_nSGL'])

is_observed_y, is_observed_x = torch.where(idx_SGL == 1)
is_not_observed_y, is_not_observed_x = torch.where(idx_nSGL == 1)


test_y, test_x = torch.where(torch.logical_and(idx_SGL == 0, idx_nSGL == 0))

observed_indices = np.arange(len(is_observed_x))
not_observed_indices = np.arange(len(is_not_observed_x))

np.random.shuffle(observed_indices)
np.random.shuffle(not_observed_indices)

is_observed_x, is_observed_y = is_observed_x[observed_indices], is_observed_y[observed_indices]
is_not_observed_x, is_not_observed_y = is_not_observed_x[not_observed_indices], is_not_observed_y[not_observed_indices]

# check all zero / nan
delete_idx = []
for i, (y, x) in enumerate(zip(is_observed_y, is_observed_x)):
    if sum(elev[y, x]) == 0:
        delete_idx.append(i)

is_observed_x = np.delete(is_observed_x, delete_idx)
is_observed_y = np.delete(is_observed_y, delete_idx)

delete_idx = []
for i, (y, x) in enumerate(zip(is_not_observed_y, is_not_observed_x)):
    if sum(elev[y, x]) == 0:
        delete_idx.append(i)

is_not_observed_x = np.delete(is_not_observed_x, delete_idx)
is_not_observed_y = np.delete(is_not_observed_y, delete_idx)


# is_observed_idx = np.random.randint(len(is_observed_y), size=3)
# is_not_observed_idx = np.random.randint(len(is_not_observed_y), size = 3)

# for idx in is_observed_idx:
#     y, x = is_observed_y[idx], is_observed_x[idx]
#     plt.plot(elev[y, x], '--o', label='1')

# for idx in is_not_observed_idx:
#     y, x = is_not_observed_y[idx], is_not_observed_x[idx]
#     plt.plot(elev[y, x], '--o', label='0')

# plt.legend()
# plt.savefig("elev_eda_0.jpg")

grid = np.zeros((601, 501))

for y, x in zip(is_observed_y, is_observed_x):
    grid[y, x] = 1

# highlight = np.ma.masked_less(grid, 1)

print(hydpot.shape)

plt.contour(grid)
plt.contourf(hydpot)

plt.colorbar()
plt.savefig("hydpot.jpg")