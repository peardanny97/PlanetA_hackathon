import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import set_seed
from data_handler import get_dataloader
from model import MLP

torch.set_default_dtype(torch.float64)

set_seed(0)

batch_size = 128
epochs = 1000

train_dataloader = get_dataloader('train', batch_size)
valid_dataloader = get_dataloader('valid', batch_size)

model = MLP()
model = model.cuda()

best_valid_loss = 100

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min = 0)

for epoch in tqdm(range(epochs)):
    running_loss = 0
    model.train()
    for data in train_dataloader:
        features, labels, n_interpolates = data
        features = torch.tensor(features).cuda()
        loss = None
        
        empty_start_arr = [] # -1이 시작되는 index를 caching
        
        # empty_start 찾는 for문
        for label in labels:

            if len(np.where(label.numpy().astype("int32") == -1)[0]) == 0:
                empty_start = 14
            else:
                empty_start = np.where(label.numpy().astype("int32") == -1)[0][0]
            
            empty_start_arr.append(empty_start)

        labels = torch.tensor(labels).cuda() # batchsize * 14

        outputs = model(features) # batchsize * 14

        optimizer.zero_grad()
        for i in range(len(outputs)):
            output = outputs[i] # shape : 14
            label = labels[i] # shape : 14
            n_interpolate = n_interpolates[i] # scalar

            # weight = torch.exp((13.0 - n_interpolate) / 91)

            empty_start = empty_start_arr[i] # ith 해당하는 empty start

            # weight = [1 1 1 1 1] if empty_start == 5
            weight = torch.ones(empty_start) # shape [empty_start]
            weight *= (13.0 - n_interpolate) / 91 # 1 ~ 13 sum 
            weight[0:5] *= 15 # weight on first point 
            weight = weight.cuda()

            with torch.no_grad():
                label = label[:empty_start] 

            if loss is None:
                # MSELoss(reduction=none) -> [empty_start] * weight
                loss = torch.mean(weight * nn.MSELoss(reduction='none')(output[:empty_start], label))
            else:
                loss += torch.mean(weight * nn.MSELoss(reduction='none')(output[:empty_start], label))



        running_loss += loss.item()

        loss /= len(outputs)

        loss.backward()

        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, train running loss", running_loss / len(train_dataloader.dataset))

    valid_loss = 0
    model.eval()
    with torch.no_grad():
        for data in valid_dataloader:
            features, labels, n_interpolates = data
            features = torch.tensor(features).cuda()
            
            loss = None
            outputs = model(features)

            empty_start_arr = []
            for label in labels:
                if len(np.where(label.numpy().astype("int32") == -1)[0]) == 0:
                    empty_start = 14
                else:
                    empty_start = np.where(label.numpy().astype("int32") == -1)[0][0]
                empty_start_arr.append(empty_start)

            labels = torch.tensor(labels).cuda()

            for i in range(len(outputs)):
                output = outputs[i]
                label = labels[i]
                n_interpolate = n_interpolates[i]

                output = torch.clamp(output, min = 0)
            
                empty_start = empty_start_arr[i]
                with torch.no_grad():
                    label = label[:empty_start]

                if loss is None:
                    loss = nn.MSELoss()(output[:empty_start], label)
                else:
                    loss += nn.MSELoss()(output[:empty_start], label)


            valid_loss += loss.item()

        if valid_loss / len(valid_dataloader.dataset) < best_valid_loss:
            best_valid_loss = valid_loss / len(valid_dataloader.dataset)
            torch.save(model.state_dict(), "./best_model.pt")
            print("Best Valid loss: ", best_valid_loss)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, valid running loss", valid_loss / len(valid_dataloader.dataset))

print(best_valid_loss)