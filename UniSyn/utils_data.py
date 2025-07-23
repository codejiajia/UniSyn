import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def split(data, repeat):
    data_train, data_temp = train_test_split(data, test_size=0.2, random_state=repeat, shuffle=True)
    data_val, data_test = train_test_split(data_temp, test_size=0.5, random_state=repeat, shuffle=True)
    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    print('data_train', data_train.shape)
    print('data_val', data_val.shape)
    print('data_test', data_test.shape)
    path_train = './repeat/repeat'+str(repeat)+'_train.csv'
    path_val = './repeat/repeat'+str(repeat)+'_val.csv'
    path_test = './repeat/repeat'+str(repeat)+'_test.csv'
    print(path_train)
    data_train.to_csv(path_train)
    data_val.to_csv(path_val)
    data_test.to_csv(path_test)
            
class load_data(Dataset):
    def __init__(self, csv_path, transforms=None):
        data = pd.read_csv(csv_path)
        self.ic50_row_labels=data['row_drug_sensitivity']
        self.ic50_col_labels=data['col_drug_sensitivity']
        self.loewe_labels = data['synergy_loewe']
        self.bliss_labels = data['synergy_bliss']
        self.S_labels = data['S_mean']
        selected_columns = ['drug_row', 'drug_col', 'depmap']
        data = data[selected_columns]
        self.inputs = np.array(data)
        self.transforms = transforms

    def __getitem__(self, index):
        row_labels = torch.tensor([self.ic50_row_labels[index]], dtype=torch.long)
        col_labels = torch.tensor([self.ic50_col_labels[index]], dtype=torch.long)
        labels = torch.tensor([self.S_labels[index]], dtype=torch.float)
        inputs_np = self.inputs[index]
        inputs_tensor = torch.from_numpy(inputs_np).float() 
        return (inputs_tensor, row_labels, col_labels, labels)
    
    def __len__(self):
        return len(self.S_labels.index)




