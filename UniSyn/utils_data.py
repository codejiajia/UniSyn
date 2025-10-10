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

def drugout_split(data, repeat):
    # Seed for reproducibility
    seed = 42
    np.random.seed(repeat)

    # Step 1: Extract unique drugs and shuffle
    unique_drugs = np.unique(np.concatenate([data['drug_row'].values, data['drug_col'].values]))
    np.random.shuffle(unique_drugs)

    # Step 2: Split unique drugs into train (80%) and test (20%) sets
    train_drug_count = int(len(unique_drugs) * 0.8)
    train_drugs = unique_drugs[:train_drug_count]
    test_drugs = unique_drugs[train_drug_count:]

    # Step 3: Separate training data (both drugs in a pair must be in train_drugs)
    data_train = data[(data['drug_row'].isin(train_drugs)) & (data['drug_col'].isin(train_drugs))].reset_index(drop=True)

    # Step 4: Separate test data (at least one drug in the pair must be in test_drugs)
    data_val_test = data[(data['drug_row'].isin(test_drugs)) & (data['drug_col'].isin(test_drugs))].reset_index(drop=True)

    # Step 5: Further split val_test into validation (50%) and test (50%)
    data_val, data_test = train_test_split(data_val_test, test_size=0.5, random_state=seed)

    # Step 6: Reset indices and print shapes
    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    print(f'data_train: {data_train.shape}')
    print(f'data_val: {data_val.shape}')
    print(f'data_test: {data_test.shape}')

    # Step 7: Save files
    path_train = f'./repeat/repeat{repeat}_train.csv'
    path_val = f'./repeat/repeat{repeat}_val.csv'
    path_test = f'./repeat/repeat{repeat}_test.csv'
    
    print(f'Saving to: {path_train}, {path_val}, {path_test}')
    data_train.to_csv(path_train, index=False)
    data_val.to_csv(path_val, index=False)
    data_test.to_csv(path_test, index=False)

def cellout_split(data, repeat):
    groups = data['depmap']
    
    # Step 1: Split off 20% (val + test)
    gss_20 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=repeat)
    for train_idx, valtest_idx in gss_20.split(data, groups=groups):
        data_train = data.iloc[train_idx].reset_index(drop=True)
        data_valtest = data.iloc[valtest_idx].reset_index(drop=True)
        groups_valtest = data_valtest['depmap']
        break

    # Step 2: Split valtest into 50% val, 50% test (i.e., 10% each of total)
    gss_10_10 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=repeat)
    for val_idx, test_idx in gss_10_10.split(data_valtest, groups=groups_valtest):
        data_val = data_valtest.iloc[val_idx].reset_index(drop=True)
        data_test = data_valtest.iloc[test_idx].reset_index(drop=True)
        break
    
    print('data_train', data_train.shape)
    print('data_val', data_val.shape)
    print('data_test', data_test.shape)
    
    # Save files
    path_train = './repeat/repeat'+str(repeat)+'_train.csv'
    path_val = './repeat/repeat'+str(repeat)+'_val.csv'
    path_test = './repeat/repeat'+str(repeat)+'_test.csv'
    
    print(path_train)
    data_train.to_csv(path_train)
    data_val.to_csv(path_val)
    data_test.to_csv(path_test)

def bothout_split(data, repeat):

    # Step 1: Extract unique drug pairs and cell lines
    unique_cells = pd.unique(data['depmap'].values.ravel())
    unique_drug_pairs = data[['drug_row', 'drug_col']].apply(lambda x: tuple(sorted(x)), axis=1).unique()

    # Create mapping dictionaries for quick lookup
    drug_pair_map = {pair: idx for idx, pair in enumerate(unique_drug_pairs)}
    cell_map = {cell: idx for idx, cell in enumerate(unique_cells)}

    # Shuffle both drug pairs and cell lines
    np.random.seed(repeat)
    np.random.shuffle(unique_cells)
    np.random.shuffle(unique_drug_pairs)

    # Calculate train, validation, and test sizes (80% - 10% - 10%)
    train_pair_size = int(len(unique_drug_pairs) * 0.8)
    train_cell_size = int(len(unique_cells) * 0.8)

    # Initialize lists for train, validation, and test samples
    train_samples = []
    val_test_samples = []

    # Initialize sets to track unique pairs in each set
    train_pairs = set(unique_drug_pairs[:train_pair_size])
    val_test_pairs = set(unique_drug_pairs[train_pair_size:])

    # Split data
    for index, row in data.iterrows():
        pair = tuple(sorted([row['drug_row'], row['drug_col']]))
        k = cell_map[row['depmap']]

        # Assign to the appropriate set based on cold start logic
        if pair in train_pairs and k < train_cell_size:
            train_samples.append(row)
        elif pair in val_test_pairs and k >= train_cell_size:
            val_test_samples.append(row)

    # Convert to DataFrames
    data_train = pd.DataFrame(train_samples, columns=data.columns)
    data_val_test = pd.DataFrame(val_test_samples, columns=data.columns)

    # Further split val_test into validation (50%) and test (50%)
    data_val, data_test = train_test_split(data_val_test, test_size=0.5, random_state=repeat)

    print(f'Train samples: {len(data_train)}')
    print(f'Validation samples: {len(data_val)}')
    print(f'Test samples: {len(data_test)}')

    # Save the datasets
    path_train = f'./repeat/repeat{repeat}_train.csv'
    path_val = f'./repeat/repeat{repeat}_val.csv'
    path_test = f'./repeat/repeat{repeat}_test.csv'
    
    print(f'Saving to: {path_train}, {path_val}, {path_test}')
    data_train.to_csv(path_train, index=False)
    data_val.to_csv(path_val, index=False)
    data_test.to_csv(path_test, index=False)


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




