import torch
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

class PolarDataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.elev_data = loadmat("./data/CryoSat2_data", appendmat=True)
        self.hydpot_data = loadmat("./data/hydropotential", appendmat=True)
        self.idx_SGL_data = loadmat("./data/idx_SGL", appendmat=True)
        self.idx_nSGL_data = loadmat("./data/idx_nSGL", appendmat=True)


        self.elev = torch.tensor(self.elev_data['elev'])
        self.hydpot = torch.tensor(self.hydpot_data['hydpot'])
        self.idx_SGL = torch.tensor(self.idx_SGL_data['idx_SGL'])
        self.idx_nSGL = torch.tensor(self.idx_nSGL_data['idx_nSGL'])

        self.is_observed_y, self.is_observed_x = torch.where(self.idx_SGL == 1)
        self.is_not_observed_y, self.is_not_observed_x = torch.where(self.idx_nSGL == 1)


        self.test_y, self.test_x = torch.where(torch.logical_and(self.idx_SGL == 0, self.idx_nSGL == 0))

        np.random.seed(1)
        observed_indices = np.arange(len(self.is_observed_x))
        not_observed_indices = np.arange(len(self.is_not_observed_x))

        np.random.shuffle(observed_indices)
        np.random.shuffle(not_observed_indices)

        self.is_observed_x, self.is_observed_y = self.is_observed_x[observed_indices], self.is_observed_y[observed_indices]
        self.is_not_observed_x, self.is_not_observed_y = self.is_not_observed_x[not_observed_indices], self.is_not_observed_y[not_observed_indices]

        # check all zero / nan
        delete_idx = []
        for i, (y, x) in enumerate(zip(self.is_observed_y, self.is_observed_x)):
            if sum(self.elev[y, x]) == 0:
                delete_idx.append(i)

        self.is_observed_x = np.delete(self.is_observed_x, delete_idx)
        self.is_observed_y = np.delete(self.is_observed_y, delete_idx)

        delete_idx = []
        for i, (y, x) in enumerate(zip(self.is_not_observed_y, self.is_not_observed_x)):
            if sum(self.elev[y, x]) == 0:
                delete_idx.append(i)

        self.is_not_observed_x = np.delete(self.is_not_observed_x, delete_idx)
        self.is_not_observed_y = np.delete(self.is_not_observed_y, delete_idx)
        

        pivot_observed = int(len(self.is_observed_x) * 0.8)
        pivot_not_observed = int(len(self.is_not_observed_x) * 0.8)

        self.train_x, self.train_y = torch.cat((self.is_observed_x[:pivot_observed], self.is_not_observed_x[:pivot_not_observed]), dim=0), torch.cat((self.is_observed_y[:pivot_observed], self.is_not_observed_y[:pivot_not_observed]), dim=0)
        self.train_label = torch.cat((torch.ones(pivot_observed), torch.zeros(pivot_not_observed)), dim = 0)
        
        self.valid_x, self.valid_y = torch.cat((self.is_observed_x[pivot_observed:], self.is_not_observed_x[pivot_not_observed:]), dim=0), torch.cat((self.is_observed_y[pivot_observed:], self.is_not_observed_y[pivot_not_observed:]), dim=0)
        self.valid_label = torch.cat((torch.ones(len(self.is_observed_x) - pivot_observed), torch.zeros(len(self.is_not_observed_x) - pivot_not_observed)), dim=0)


        print("train statistics: ", len(self.train_x), len(self.train_y), len(self.train_label))
        print("valid statistics: ", len(self.valid_x), len(self.valid_y), len(self.valid_label))
        print("test statistics: ", len(self.test_x), len(self.test_y))
    
    def __len__(self):
        if self.split == 'test':
            return len(self.test_x)
        elif self.split == 'train':
            return len(self.train_x)
        elif self.split == 'valid':
            return len(self.valid_x)

    def __getitem__(self, idx):
        if self.split == 'test':
            # elev[pos_y, pos_x] -> length 102 vector
            # hydpot[pos_y, pos_x] -> scalar
            # label: 1 observed label: 0 maybe not observed
            return self.elev[self.test_y[idx], self.test_x[idx]], self.hydpot[self.test_y[idx], self.test_x[idx]], (self.test_y[idx], self.test_x[idx])
        elif self.split == 'train':
            return self.elev[self.train_y[idx], self.train_x[idx]], self.hydpot[self.train_y[idx], self.train_x[idx]], self.train_label[idx]
        elif self.split == 'valid':
            return self.elev[self.valid_y[idx], self.valid_x[idx]], self.hydpot[self.valid_y[idx], self.valid_x[idx]], self.valid_label[idx]

def get_dataloader(split, batch_size):
    dataset = PolarDataset(split)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    return dataloader
