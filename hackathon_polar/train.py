import torch
import torch.nn as nn
import torch.optim as optim
from data_handler import get_dataloader
from utils import set_seed, FocalLoss
from model import Classifier
from tqdm import tqdm

torch.set_default_dtype(torch.float64)

set_seed(0)

train_dataloader = get_dataloader("train", 64)
valid_dataloader = get_dataloader("valid", 64)
test_dataloader = get_dataloader("test", 64)

net = Classifier(1, 2).cuda()


epochs = 1000
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min = 0)

for epoch in tqdm(range(epochs)):
    best_val_acc = 0
    correct = 0
    net.train()
    for data in train_dataloader:  
        elev, hydpot, label = data
        label = label.long()
        elev = elev.unsqueeze(1)
    
        elev, hydpot, label = elev.cuda(), hydpot.cuda(), label.cuda()

        # elev.shape -> [Batch_size, 1(scaler per time), 102(Time)]
        # hydpot, label -> [Batch_size]

        pred = net(elev, hydpot)

        optimizer.zero_grad()
        loss = FocalLoss(gamma=1)(pred, label)
        
        loss.backward()
        optimizer.step()

        correct += (torch.argmax(pred, dim=1) == label).sum()
        
    
    if epoch % 10 == 0:
        print(correct / len(train_dataloader.dataset))
    
    net.eval()
    val_correct = 0
    for data in valid_dataloader:
        elev, hydpot, label = data
        label = label.long()
        elev = elev.unsqueeze(1)
        elev, hydpot, label = elev.cuda(), hydpot.cuda(), label.cuda()

        pred = net(elev, hydpot)

        val_correct += (torch.argmax(pred, dim =1) == label).sum()

    if epoch % 10 == 0:
        valid_acc = val_correct / len(valid_dataloader.dataset)
        print(valid_acc)

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            torch.save(net.state_dict(), "./best_model.pt")

    scheduler.step()
