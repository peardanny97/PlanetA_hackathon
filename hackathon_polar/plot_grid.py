import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat
from matplotlib import colors

idx_SGL_data = loadmat("./data/idx_SGL", appendmat=True)
idx_nSGL_data = loadmat("./data/idx_nSGL", appendmat=True)

idx_SGL = np.array(idx_SGL_data['idx_SGL'])
idx_nSGL = np.array(idx_nSGL_data['idx_nSGL'])

is_observed_y, is_observed_x = np.where(idx_SGL == 1)
is_not_observed_y, is_not_observed_x = np.where(idx_nSGL == 1)

with open("./result/grid_result_0.9.pickle", "rb") as fp:
    grid = pickle.load(fp)

for y, x in zip(is_observed_y, is_observed_x):
    grid[y, x] = 4

for y, x in zip(is_not_observed_y, is_not_observed_x):
    grid[y, x] = 5


# no, yes, no obs, sea, is observed, may not be observed
cmap = colors.ListedColormap([
    'black', 'yellow', 'white', 'white', 'skyblue', 'blue'
])

xlist = np.linspace(0, 500, 501)
ylist = np.linspace(0, 600, 601)
X, Y = np.meshgrid(xlist, ylist)

grid = grid.astype(np.uint8)
levels = np.arange(0, 5 + 2) - 0.5

plt.contourf(X, Y, grid[::-1], cmap=cmap, levels=levels)
plt.savefig("./result/result_0.9.jpg")
