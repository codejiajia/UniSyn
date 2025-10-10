import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import os
import config
from model import Net_View1,Net
from utils_data import split, drugout_split, cellout_split, bothout_split, load_data
from utils_model import train,  save_model, test, metric
import warnings
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings("ignore")

torch.set_num_threads(32)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def run(repeat):
    setup_seed(42)
    transform = transforms.ToTensor()
    path_train = './repeat/repeat'+str(repeat)+'_train.csv'
    path_val = './repeat/repeat'+str(repeat)+'_val.csv'
    train_data = load_data(path_train, transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                             num_workers=model_config['num_workers'], pin_memory=True,drop_last=True)
    valid_data = load_data(path_val, transform)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    log_dir = './save/repeat'+str(repeat)+'.txt'
    f = open(log_dir,'w')
    f.close()
    layers = {
        'ic50_layers': [1536,256,64],
        'synergy_layers': [3072,1024,64],
    }
    model_View1 = Net_View1(model_config, drug_fp, drug_seq,drug_gra,cell,cell_mu,cell_nv)
    model = Net(model_config, drug_fp, cell, model_View1,layers)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.95, 0.999), amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=True)
    scaler = GradScaler()
    start_epoch = 0
    best_valid_pear = 0
    save_dir = './save/repeat'+str(repeat)+'_best.pth'
    patience = 100
    min_delta = 0.001
    counter = 0 
    for epoch in range(start_epoch+1, epochs+1):
        train_loss, train_results, val_results = train(model, device, reverse_drug_map, train_loader, valid_loader, optimizer, criterion_mse, criterion_class, epoch, scaler)
        (
            train_mse, train_rmse, train_r2, train_pear, train_pcc,
            train_row_bacc, train_row_precision, train_row_recall, train_row_f1, train_row_kappa, train_row_roc_auc, train_row_prc_auc,
            train_col_bacc, train_col_precision, train_col_recall, train_col_f1, train_col_kappa, train_col_roc_auc, train_col_prc_auc
        ) = train_results
        (
            val_mse, val_rmse, val_r2, val_pear, val_pcc,
            val_row_bacc, val_row_precision, val_row_recall, val_row_f1, val_row_kappa, val_row_roc_auc, val_row_prc_auc,
            val_col_bacc, val_col_precision, val_col_recall, val_col_f1, val_col_kappa, val_col_roc_auc, val_col_prc_auc
        ) = val_results
        scheduler.step(val_pear)
        if val_pear> best_valid_pear + min_delta:
            best_valid_pear = val_pear
            save_model(epoch, model, optimizer, val_pear, save_dir)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break 
        if epoch % 5 == 0:
            train_res = 'Epoch: {:05d}, Train loss: {:.3f}, '.format(epoch, train_loss)
            train_res += 'MSE: {:.3f}, RMSE: {:.3f}, R2: {:.3f}, Pearson: {:.3f}, Spearman: {:.3f}, '.format(
                train_mse, train_rmse, train_r2, train_pear, train_pcc)
            train_res += 'row_bacc: {:.3f}, row_precision: {:.3f}, row_recall: {:.3f}, '.format(
                train_row_bacc, train_row_precision, train_row_recall)
            train_res += 'row_f1: {:.3f}, row_kappa: {:.3f}, row_ROC_AUC: {:.3f}, row_PRC_AUC: {:.3f}, '.format(
                train_row_f1, train_row_kappa, train_row_roc_auc, train_row_prc_auc)
            train_res += 'col_bacc: {:.3f}, col_precision: {:.3f}, col_recall: {:.3f}, '.format(
                train_col_bacc, train_col_precision, train_col_recall)
            train_res += 'col_f1: {:.3f}, col_kappa: {:.3f}, col_ROC_AUC: {:.3f}, col_PRC_AUC: {:.3f}'.format(
                train_col_f1, train_col_kappa, train_col_roc_auc, train_col_prc_auc)
            val_res = 'Validation - MSE: {:.3f}, RMSE: {:.3f}, R2: {:.3f}, Pearson: {:.3f}, Spearman: {:.3f}, '.format(
                val_mse, val_rmse, val_r2, val_pear, val_pcc)
            val_res += 'row_bacc: {:.3f}, row_precision: {:.3f}, row_recall: {:.3f}, '.format(
                val_row_bacc, val_row_precision, val_row_recall)
            val_res += 'row_f1: {:.3f}, row_kappa: {:.3f}, row_ROC_AUC: {:.3f}, row_PRC_AUC: {:.3f}, '.format(
                val_row_f1, val_row_kappa, val_row_roc_auc, val_row_prc_auc)
            val_res += 'col_bacc: {:.3f}, col_precision: {:.3f}, col_recall: {:.3f}, '.format(
                val_col_bacc, val_col_precision, val_col_recall)
            val_res += 'col_f1: {:.3f}, col_kappa: {:.3f}, col_ROC_AUC: {:.3f}, col_PRC_AUC: {:.3f}'.format(
                val_col_f1, val_col_kappa, val_col_roc_auc, val_col_prc_auc)
            print(train_res)
            print(val_res)
            with open(log_dir,"a") as f:
                f.write(train_res+'\n')
                f.write(val_res+'\n')
                f.close()

def get_res(repeat):
    setup_seed(42)
    print(f'Start test, repeat {repeat}')
    transform = transforms.ToTensor()
    path_test = './repeat/repeat'+str(repeat)+'_test.csv'
    test_data = load_data(path_test, transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                            num_workers=model_config['num_workers'], pin_memory=True)
    model_View1 = Net_View1(model_config,drug_fp, drug_seq,drug_gra,cell,cell_mu,cell_nv)
    layers = {
        'ic50_layers': [1536,256,64],
        'synergy_layers': [3072,1024,64],
    }
    model = Net(model_config,  drug_fp, cell, model_View1,layers)
    model.to(device)
    save_dir = './save/repeat'+str(repeat)+'_best.pth'
    if os.path.exists(save_dir):
        checkpoint = torch.load(save_dir, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print('Load saved model')
    else:
        print('No saved model')
    test_data_pd = pd.read_csv(path_test)
    compare= test(model, device, reverse_drug_map,test_loader)
    compare = compare.reset_index(drop=True)
    test_data_pd['row_true'] = compare['row_true'].astype(float)
    test_data_pd['row_pred'] = compare['row_pred'].astype(float)
    test_data_pd['col_true'] = compare['col_true'].astype(float)
    test_data_pd['col_pred'] = compare['col_pred'].astype(float)
    test_data_pd['true'] = compare['true'].astype(float)
    test_data_pd['pred'] = compare['pred'].astype(float)
    test_data_pd['row_prob'] = compare['row_prob'].astype(float)
    test_data_pd['col_prob'] = compare['col_prob'].astype(float)
    test_data_pd = test_data_pd.drop(labels=['Unnamed: 0'],axis=1)       
    pred_dir = './predict/repeat'+str(repeat)+'_predict.csv'
    test_data_pd.to_csv(pred_dir)
    results = metric(test_data_pd)
    (
            mse, rmse, r2, pear, pcc,
            row_bacc, row_precision, row_recall, row_f1, row_kappa, row_roc_auc, row_prc_auc,
            col_bacc, col_precision, col_recall, col_f1, col_kappa, col_roc_auc, col_prc_auc
        ) = results
    res = 'MSE: {:.3f}, RMSE: {:.3f}, R2: {:.3f}, Pearson: {:.3f}, Spearman: {:.3f}, '.format(mse, rmse, r2, pear, pcc)
    res += 'row_bacc: {:.3f}, row_precision: {:.3f}, row_recall: {:.3f}, '.format(row_bacc, row_precision, row_recall)
    res += 'row_f1: {:.3f}, row_kappa: {:.3f}, row_ROC_AUC: {:.3f}, row_PRC_AUC: {:.3f}, '.format(row_f1, row_kappa, row_roc_auc, row_prc_auc)
    res += 'col_bacc: {:.3f}, col_precision: {:.3f}, col_recall: {:.3f}, '.format(col_bacc, col_precision, col_recall)
    res += 'col_f1: {:.3f}, col_kappa: {:.3f}, col_ROC_AUC: {:.3f}, col_PRC_AUC: {:.3f}'.format(col_f1, col_kappa, col_roc_auc, col_prc_auc)
    print(res)
    res_dir = './predict/repeat'+str(repeat)+'_metric.txt'
    with open(res_dir, "w") as f:
        f.write(res+'\n')
        f.close()
        
if __name__ == '__main__':
    # 1 = normal, 2 = drugout, 3 = cellout, 4 = bothout
    split_mode = 1 
    split_flag = 0
    train_flag = 0
    test_flag = 0
    if split_flag == 0:
        print('Split data')
        data = pd.read_csv('./rawData/drugcomb/data_to_split.csv')
        print(data)
        for repeat in range(1,6):
            if split_mode == 1:
                print(f'Using normal split (repeat {repeat})')
                split(data, repeat=repeat)
            elif split_mode == 2:
                print(f'Using drugout split (repeat {repeat})')
                drugout_split(data, repeat=repeat)
            elif split_mode == 3:
                print(f'Using cellout split (repeat {repeat})')
                cellout_split(data, repeat=repeat)
            elif split_mode == 4:
                print(f'Using bothout split (repeat {repeat})')
                bothout_split(data, repeat=repeat)
            else:
                raise ValueError(f"Invalid split_mode: {split_mode}. Must be one of [1, 2, 3, 4].")


    print('Load config')
    model_config = config.model_config
    gpu = model_config['gpu']
    batch_size = model_config['batch_size']
    criterion_mse = nn.MSELoss()
    criterion_class= nn.CrossEntropyLoss()
    lr = model_config['lr']
    epochs = model_config['epochs']
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device = torch.device('cpu')
    cell = pd.read_csv('./rawData/drugcomb/Cell_use_zscore.csv')
    cell = cell.drop(labels=['id'],axis=1)
    cell_nv = pd.read_csv('./rawData/drugcomb/nv_zscore.csv')
    cell_mu = pd.read_csv('./rawData/drugcomb/mutation.csv')
    drug_fp = pd.read_csv('./rawData/drugcomb/Drug_use.csv')
    drug_fp = drug_fp.drop(labels=['id'],axis=1)
    drug_seq = pd.read_csv('./rawData/drugcomb/drug_sequence_em.csv',header=0)
    drug_seq = drug_seq.iloc[:,1:769]
    drug_gra = np.load('./rawData/drugcomb/drug_feature_graph.npy', allow_pickle=True).item()
    drug_map = np.load('./rawData/drugcomb/Drug_map.npy', allow_pickle=True).item()
    reverse_drug_map = {value: key for key, value in drug_map.items()}
    for repeat in range(1,6):
        if train_flag == 1:
            print(f'Start train, repeat {repeat}')
            run(repeat)
        if test_flag == 1:
            get_res(repeat)
        torch.cuda.empty_cache()