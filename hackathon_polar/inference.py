import pickle
from data_handler import get_dataloader
from model import Classifier
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve

torch.set_default_dtype(torch.float64)


model = Classifier(1, 2)
model.load_state_dict(torch.load("./result/best_model.pt"))
model.eval()
model = model.cuda()


valid_dataloader = get_dataloader("valid", 706)
test_dataloader = get_dataloader("test", 1)

# get optimal threshold
val_correct = 0

with torch.no_grad():
    for data in valid_dataloader:
        elev, hydpot, label = data
        elev = elev.unsqueeze(1)
        elev, hydpot = elev.cuda(), hydpot.cuda()
        
        pred = model(elev, hydpot).squeeze()
        scores = torch.sigmoid(pred)[:, 1]

        val_correct += (torch.argmax(pred, dim =1) == label.cuda()).sum().cpu()

        fpr, tpr, thresholds = roc_curve(label, scores.detach().cpu(), pos_label = 1)

print("Validation Accuracy:", val_correct / 706)


optimal_threshold = thresholds[np.argmax(np.sqrt(tpr * (1 - fpr)))]

print("Gmean optimal threshold", optimal_threshold)

grid = np.zeros((601, 501))

no_obs_pos_arr = []
sea_pos_arr = []

for data in tqdm(test_dataloader):
    elev, hydpot, pos = data
    y, x = pos
    if torch.sum(elev) == 0: # no obs in test
        grid[y, x] = 2
        no_obs_pos_arr.append(pos)
    elif True in torch.isnan(elev): # sea
        grid[y, x] = 3
        sea_pos_arr.append(pos)
    else:
        elev = elev.unsqueeze(1)
        elev, hydpot = elev.cuda(), hydpot.cuda()
        # is_lake = torch.sigmoid(model(elev, hydpot))[0][1] > optimal_threshold
        is_lake = torch.sigmoid(model(elev, hydpot))[0][1] > 0.8

        grid[y, x] = is_lake.cpu()

with open("./result/grid_result_0.8.pickle", "wb") as fp:
    pickle.dump(grid, fp)