import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import pickle

class OceanDataset(Dataset):
    def __init__(self, split):
        self.split = split


        years = np.arange(0, 26, 1, dtype=np.int32)
        np.random.shuffle(years)

        years_pivot = int(26 * 0.8)

        self.train_year, self.valid_year = years[:years_pivot], years[years_pivot:]


        # Load data
        self.train_sla = []
        self.train_sst = []
        self.train_label = []
        
        for year in self.train_year:
            sla_filename, sst_filename, label_filename = f"{year}th_low_freq_sla.pickle", f"{year}th_low_freq_sst.pickle", f"{year}th_target.pickle"
            
            with open(os.path.join("./data", sla_filename), "rb") as fp:
                self.train_sla.append(pickle.load(fp))

            with open(os.path.join("./data", sst_filename), "rb") as fp:
                self.train_sst.append(pickle.load(fp))
            
            with open(os.path.join("./data", label_filename), "rb") as fp:
                self.train_label.append(np.array(pickle.load(fp)))

            
        self.train_sla, self.train_sst, self.train_label = np.array(self.train_sla), np.array(self.train_sst), np.array(self.train_label)
        
        # sla/sst: year(20) * month * lat(40) * lon(80)
        # label: year(20): data_num * 14(depth) * features(latitude', 'longitude', 'year', 'month', 'depth', 'temperature', 'interpolated_length')

        self.valid_sla = []
        self.valid_sst = []
        self.valid_label = []

        for year in self.valid_year:
            sla_filename, sst_filename, label_filename = f"{year}th_low_freq_sla.pickle", f"{year}th_low_freq_sst.pickle", f"{year}th_target.pickle"
            
            with open(os.path.join("./data", sla_filename), "rb") as fp:
                self.valid_sla.append(pickle.load(fp))

            with open(os.path.join("./data", sst_filename), "rb") as fp:
                self.valid_sst.append(pickle.load(fp))
            
            with open(os.path.join("./data", label_filename), "rb") as fp:
                self.valid_label.append(np.array(pickle.load(fp)))

            
        self.valid_sla, self.valid_sst, self.valid_label = np.array(self.valid_sla), np.array(self.valid_sst), np.array(self.valid_label)

        # print(self.train_sla.shape, self.train_sst.shape, self.train_label.shape)

        with open("./data/low_freq_lat.pickle", "rb") as fp:
            self.lat_info = pickle.load(fp)

        self.lat_info = np.array(self.lat_info)

        with open("./data/low_freq_lon.pickle", "rb") as fp:
            self.lon_info = pickle.load(fp)
        self.lon_info = np.array(self.lon_info)

        if self.split == "train":
            self.cropped_train_sla = None
            self.cropped_train_sst = None
            self.cropped_train_label = None


            for label_year in self.train_label:
                sla, sst, label = self.crop_input_wrt_label(self.split, label_year)

                if self.cropped_train_sla is None:
                    self.cropped_train_sla = np.array(sla) 
                else:
                    self.cropped_train_sla = np.concatenate((self.cropped_train_sla, sla), axis = 0)
                
                if self.cropped_train_sst is None:
                    self.cropped_train_sst = np.array(sst)
                else:
                    self.cropped_train_sst = np.concatenate((self.cropped_train_sst, sst), axis = 0)
                
                if self.cropped_train_label is None:
                    self.cropped_train_label = np.array(label)
                else:
                    self.cropped_train_label = np.concatenate((self.cropped_train_label, label), axis = 0)

            del self.train_sla, self.train_sst, self.train_label

        elif self.split == "valid":
            self.cropped_valid_sla = None
            self.cropped_valid_sst = None
            self.cropped_valid_label = None

            for label_year in self.valid_label:
                sla, sst, label = self.crop_input_wrt_label(self.split, label_year)

                if self.cropped_valid_sla is None:
                    self.cropped_valid_sla = np.array(sla) 
                else:
                    self.cropped_valid_sla = np.concatenate((self.cropped_valid_sla, sla), axis = 0)
                
                if self.cropped_valid_sst is None:
                    self.cropped_valid_sst = np.array(sst)
                else:
                    self.cropped_valid_sst = np.concatenate((self.cropped_valid_sst, sst), axis = 0)
                
                if self.cropped_valid_label is None:
                    self.cropped_valid_label = np.array(label)
                else:
                    self.cropped_valid_label = np.concatenate((self.cropped_valid_label, label), axis = 0)

            del self.valid_sla, self.valid_sst, self.valid_label

    
    def crop_input_wrt_label(self, split, label_year, input_size=2):
        if split == "train":
            year_arr, sla_arr, sst_arr = self.train_year, self.train_sla, self.train_sst 
        else:
            year_arr, sla_arr, sst_arr = self.valid_year, self.valid_sla, self.valid_sst

        n_problem = 0

        return_sla = []
        return_sst = []
        return_label = []

        for label in label_year:
            lat, lon, year, month, _, _, n_interpolate = label[0]
            year = int(year - 1993)
            month = int(month) - 1
            year_idx = np.where(year_arr == year)[0].item()
            
            sla_grid = sla_arr[year_idx, month - 1]
            sst_grid = sst_arr[year_idx, month - 1]


            try:
                lat_upper_idx = min(np.where(self.lat_info > lat)[0])
                lat_lower_idx = max(np.where(self.lat_info <= lat)[0])
            except:
                n_problem += 1
                continue

            try:
                lon_upper_idx = min(np.where(self.lon_info > lon)[0])
                lon_lower_idx = max(np.where(self.lon_info <= lon)[0])
            except:
                n_problem += 1
                continue

            if input_size == 3:
                lat_lower_idx -= 1
                lon_lower_idx -= 1    
            elif input_size == 5:
                lat_lower_idx -= 2
                lat_upper_idx += 1
                lon_lower_idx -= 2
                lon_upper_idx += 1

            height = lat_upper_idx - lat_lower_idx  + 1
            width = lon_upper_idx - lon_lower_idx + 1

            sla_input = sla_grid[lat_lower_idx:lat_upper_idx+1, lon_lower_idx:lon_upper_idx+1]
            sst_input = sst_grid[lat_lower_idx:lat_upper_idx+1, lon_lower_idx:lon_upper_idx+1]

            if True in np.isnan(sla_input) or True in np.isnan(sst_input):
                n_problem += 1
                continue
            else:
                return_sla.append(sla_input)
                return_sst.append(sst_input)
                # lat, lon, month, temp, n_interpolate
                return_label.append(np.vstack((label[:, 0], label[:, 1], label[:, 3], label[:, -2], label[:, -1])).transpose())

            assert width == input_size and height == input_size

        return np.array(return_sla), np.array(return_sst), np.array(return_label)
    def __len__(self):
        if self.split == "train":
            return len(self.cropped_train_sla)
        elif self.split == "valid":
            return len(self.cropped_valid_sla)

    def __getitem__(self, idx):
        if self.split == "train":

            
            lat, lon, month, label, n_interpolate = self.cropped_train_label[idx][0, 0], self.cropped_train_label[idx][0, 1], self.cropped_train_label[idx][0, 2], self.cropped_train_label[idx][:, 3], self.cropped_train_label[idx][0, 4]
            lat_layer = np.full((2,2), lat/90)
            lon_layer = np.full((2,2), lon/180)
            month_layer = np.full((2,2), month/12)
            feature_layer = np.stack((self.cropped_train_sla[idx],self.cropped_train_sst[idx],lat_layer, lon_layer, month_layer))

            # layer 2 * 2 * 5 (sla, sst, lat, lon, month)
            

            #flattened_sla = self.cropped_train_sla[idx].flatten()
            #flattened_sst = self.cropped_train_sst[idx].flatten()

            lat, lon, month, label, n_interpolate = self.cropped_train_label[idx][0, 0], self.cropped_train_label[idx][0, 1], self.cropped_train_label[idx][0, 2], self.cropped_train_label[idx][:, 3], self.cropped_train_label[idx][0, 4]

            # if month in (3, 4, 5):
            #     month = np.eye(4, dtype=np.float32)[0]
            # elif month in (6, 7, 8):
            #     month = np.eye(4, dtype=np.float32)[1]
            # elif month in (9, 10, 11):
            #     month = np.eye(4, dtype=np.float32)[2]
            # else:
            #     month = np.eye(4, dtype=np.float32)[3]

            # concated = np.concatenate((np.concatenate((np.concatenate((np.concatenate((flattened_sla, flattened_sst)), month)), np.array([lat]))), np.array([lon])))
            
            # feature length 4 + 4 + 4 + 1 + 1 => channel 2*2 각 채널 1. sla 2. sst 3. month (normalization /12 ) 4.lat (/90 ?)  5. lon (/180)
            # 2 * 2 * 5 (input feature) *batch size 유의미..? 
            # with zero padding(1) 4 * 4, filter size 2*2 stride  = 2, 2 * 2 channel 32 64 ... last flatten linear ftn 14 len vector MSE loss 

            return feature_layer, label, n_interpolate

        elif self.split == "valid":

            # flattened_sla = self.cropped_valid_sla[idx].flatten()
            # flattened_sst = self.cropped_valid_sst[idx].flatten()

            lat, lon, month, label, n_interpolate = self.cropped_valid_label[idx][0, 0], self.cropped_valid_label[idx][0, 1], self.cropped_valid_label[idx][0, 2], self.cropped_valid_label[idx][:, 3], self.cropped_valid_label[idx][0, 4]
            
            lat_layer = np.full((2,2), lat/90)
            lon_layer = np.full((2,2), lon/180)
            month_layer = np.full((2,2), month/12)
            feature_layer = np.stack((self.cropped_valid_sla[idx],self.cropped_valid_sst[idx],lat_layer, lon_layer, month_layer))

            # if month in (3, 4, 5):
            #     month = np.eye(4, dtype=np.float32)[0]
            # elif month in (6, 7, 8):
            #     month = np.eye(4, dtype=np.float32)[1]
            # elif month in (9, 10, 11):
            #     month = np.eye(4, dtype=np.float32)[2]
            # else:
            #     month = np.eye(4, dtype=np.float32)[3]

            # concated = np.concatenate((np.concatenate((np.concatenate((np.concatenate((flattened_sla, flattened_sst)), month)), np.array([lat]))), np.array([lon])))

            return feature_layer, label, n_interpolate

def get_dataloader(split, batch_size):
    dataloader = DataLoader(OceanDataset(split=split), batch_size=batch_size, shuffle=True, drop_last=False)

    return dataloader