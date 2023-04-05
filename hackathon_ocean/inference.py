from data_handler import get_dataloader
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

from data_handler import get_dataloader
from model import MLP


model = MLP()

model.load_state_dict(torch.load("./best_model.pt"))
model.cuda()

model.eval()

train_dataloader = get_dataloader('train', batch_size=128)
valid_dataloader = get_dataloader('valid', batch_size = 128)

depth = [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500  ]

with torch.no_grad():
    for idx, data in enumerate(valid_dataloader.dataset):
        feature, true_label, n_interpolate = data 

        if n_interpolate != 0 or -1 in true_label:
            continue

        print(idx)
        feature = torch.tensor(feature).cuda()


        pred = model(feature).cpu()
        pred = torch.clamp(pred, min=0)
        plt.plot(depth, true_label, '--bo', label="true" )
        plt.plot(depth, pred, '--ro', label='pred')
        plt.legend()
        plt.savefig("./result.jpg")

        break