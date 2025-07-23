import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from torch.cuda.amp import autocast

def train(model, device, reverse_drug_map, train_loader, val_loader, optimizer, criterion_mse, criterion_class, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    count = 0
    compare = pd.DataFrame(columns=('row_pred','row_true','col_pred','col_true','pred','true','row_prob', 'col_prob'))
    for i, data in enumerate(train_loader, 0):
        inputs, row_labels, col_labels, labels = data
        row_labels = row_labels.to(device).squeeze().long()  
        col_labels = col_labels.to(device).squeeze().long()
        labels = labels.reshape(labels.shape[0], 1).to(device).float()
        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
                drug_row_ic50, drug_col_ic50, outputs = model(reverse_drug_map, inputs)
                T = 2.0
                soft_teacher = ((drug_row_ic50 + drug_col_ic50) / 2.0) / T
                soft_student = outputs / T
                soft_loss = F.kl_div(
                    F.log_softmax(soft_student, dim=0),
                    F.softmax(soft_teacher.detach(), dim=0),
                    reduction='batchmean'
                ) * (T * T)
                drug_row_ic50 = drug_row_ic50.reshape(-1, 2).float()  
                drug_col_ic50 = drug_col_ic50.reshape(-1, 2).float()
                outputs = outputs.float() 
                loss_row = criterion_class(drug_row_ic50, row_labels)
                loss_col = criterion_class(drug_col_ic50, col_labels)
                loss_syn = criterion_mse(outputs, labels)
                λ = 0.1
                loss = loss_row+loss_col+loss_syn+λ * soft_loss
                loss = loss.float()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            drug_row_ic50, drug_col_ic50, outputs = model(reverse_drug_map, inputs)
            T = 2.0
            soft_teacher = ((drug_row_ic50 + drug_col_ic50) / 2.0) / T
            soft_student = outputs / T
            soft_loss = F.kl_div(
                F.log_softmax(soft_student, dim=0),
                F.softmax(soft_teacher.detach(), dim=0),
                reduction='batchmean'
            ) * (T * T)
            drug_row_ic50 = drug_row_ic50.reshape(-1, 2).float()  
            drug_col_ic50 = drug_col_ic50.reshape(-1, 2).float()
            outputs = outputs.float() 
            loss_row = criterion_class(drug_row_ic50, row_labels)
            loss_col = criterion_class(drug_col_ic50, col_labels)
            loss_syn = criterion_mse(outputs, labels)
            λ = 0.1
            loss = loss_row+loss_col+loss_syn+λ * soft_loss
            loss = loss.float()
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()
        count += 1
        running_loss = 0.0
        row_labels = row_labels.cpu().detach().float()
        col_labels = col_labels.cpu().detach().float()
        labels = labels.cpu().detach().float()
        drug_row_ic50 = drug_row_ic50.cpu().detach().float()
        drug_col_ic50 = drug_col_ic50.cpu().detach().float()
        outputs = outputs.cpu().detach().float()
        row_labels_list = np.array(row_labels).tolist()
        col_labels_list = np.array(col_labels).tolist()
        labels_list = np.array(labels)[:,0].tolist()
        outputs_list = np.array(outputs)[:,0].tolist()
        row_probs = torch.softmax(drug_row_ic50, dim=1)
        col_probs = torch.softmax(drug_col_ic50, dim=1)
        row_pred_list = torch.argmax(row_probs, dim=1).cpu().numpy().tolist()
        col_pred_list = torch.argmax(col_probs, dim=1).cpu().numpy().tolist()
        row_prob_list = row_probs[:, 1].cpu().numpy().tolist()
        col_prob_list = col_probs[:, 1].cpu().numpy().tolist()
        compare_temp = pd.DataFrame({
            'row_true': row_labels_list,
            'row_pred': row_pred_list,
            'col_true': col_labels_list,
            'col_pred': col_pred_list,
            'true': labels_list,
            'pred': outputs_list,
            'row_prob': row_prob_list,
            'col_prob': col_prob_list
        })
        compare = pd.concat([compare, compare_temp], ignore_index=True)
    compare_copy = compare.copy()
    train_results = metric(compare_copy)
    val_results = validate(model, device, reverse_drug_map, val_loader)
    return total_loss/count, train_results, val_results

def metric(compare):
    row_true = compare['row_true'].astype('int64')
    row_pred = compare['row_pred'].astype('int64')
    col_true = compare['col_true'].astype('int64')
    col_pred = compare['col_pred'].astype('int64')
    y_true = compare['true'].astype('int64')
    y_pred = compare['pred'].astype('int64')
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pear = pearsonr(y_true, y_pred)[0]
    pcc = spearmanr(y_true, y_pred)[0]
    row_bacc = balanced_accuracy_score(row_true, row_pred)
    row_precision = precision_score(row_true, row_pred, average='binary', zero_division=0)
    row_recall = recall_score(row_true, row_pred, average='binary', zero_division=0)
    row_f1 = f1_score(row_true, row_pred, average='binary', zero_division=0)
    row_kappa = cohen_kappa_score(row_true, row_pred)
    if 'row_prob' in compare.columns:
        row_fpr, row_tpr, _ = roc_curve(row_true, compare['row_prob'])
        row_roc_auc = auc(row_fpr, row_tpr)
        row_prec, row_rec, _ = precision_recall_curve(row_true, compare['row_prob'])
        row_prc_auc = auc(row_rec, row_prec)
    else:
        row_roc_auc = row_prc_auc = 0
    col_bacc = balanced_accuracy_score(col_true, col_pred)
    col_precision = precision_score(col_true, col_pred, average='binary', zero_division=0)
    col_recall = recall_score(col_true, col_pred, average='binary', zero_division=0)
    col_f1 = f1_score(col_true, col_pred, average='binary', zero_division=0)
    col_kappa = cohen_kappa_score(col_true, col_pred)
    if 'col_prob' in compare.columns:
        col_fpr, col_tpr, _ = roc_curve(col_true, compare['col_prob'])
        col_roc_auc = auc(col_fpr, col_tpr)
        col_prec, col_rec, _ = precision_recall_curve(col_true, compare['col_prob'])
        col_prc_auc = auc(col_rec, col_prec)
    else:
        col_roc_auc = col_prc_auc = 0
    return (
        mse, rmse,r2, pear,pcc,
        row_bacc, row_precision, row_recall, row_f1, row_kappa, row_roc_auc, row_prc_auc,
        col_bacc, col_precision, col_recall, col_f1, col_kappa, col_roc_auc, col_prc_auc
    )

def save_model(epoch, model, optimizer, val_pear, save_dir):
    checkpoint = {
        'best_pear': val_pear,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_dir)
    print(f"Model saved to {save_dir}")

def test(model, device, reverse_drug_map, test_loader):
    model.eval()
    compare = pd.DataFrame(columns=('row_pred', 'row_true', 'col_pred', 'col_true', 'pred', 'true',  'row_prob', 'col_prob'))
    with torch.no_grad():
        for data in test_loader:
            inputs, row_labels, col_labels, labels = data
            row_labels = row_labels.to(device).squeeze().long()  
            col_labels = col_labels.to(device).squeeze().long()
            labels = labels.reshape(labels.shape[0], 1).to(device).float()
            drug_row_ic50 ,drug_col_ic50, outputs = model(reverse_drug_map, inputs)
            drug_row_ic50 = drug_row_ic50.reshape(-1, 2).float()  
            drug_col_ic50 = drug_col_ic50.reshape(-1, 2).float()
            outputs = outputs.float() 
            row_labels = row_labels.cpu().detach().float()
            col_labels = col_labels.cpu().detach().float()
            labels = labels.cpu().detach().float()
            drug_row_ic50  = drug_row_ic50.cpu().detach().float()
            drug_col_ic50 = drug_col_ic50.cpu().detach().float()
            outputs = outputs.cpu().detach().float()
            row_labels_list = np.array(row_labels).tolist()
            col_labels_list = np.array(col_labels).tolist()
            labels_list = np.array(labels)[:,0].tolist()
            outputs_list = np.array(outputs)[:,0].tolist()
            row_probs = torch.softmax(drug_row_ic50, dim=1)
            col_probs = torch.softmax(drug_col_ic50, dim=1)
            row_pred_list = torch.argmax(row_probs, dim=1).cpu().numpy().tolist()
            col_pred_list = torch.argmax(col_probs, dim=1).cpu().numpy().tolist()
            row_prob_list = row_probs[:, 1].cpu().numpy().tolist()
            col_prob_list = col_probs[:, 1].cpu().numpy().tolist()
            compare_temp = pd.DataFrame({
                'row_true': row_labels_list,
                'row_pred': row_pred_list,
                'col_true': col_labels_list,
                'col_pred': col_pred_list,
                'true': labels_list,
                'pred': outputs_list,
                'row_prob': row_prob_list,
                'col_prob': col_prob_list
            })
            compare = pd.concat([compare, compare_temp], ignore_index=True)
    return compare

def validate(model, device, reverse_drug_map, val_loader):
    model.eval()
    compare = pd.DataFrame(columns=('row_pred', 'row_true', 'col_pred', 'col_true', 'pred', 'true', 'row_prob', 'col_prob'))
    with torch.no_grad():
        for data in val_loader:
            inputs, row_labels, col_labels, labels = data
            row_labels = row_labels.to(device).squeeze().long()  
            col_labels = col_labels.to(device).squeeze().long()
            labels = labels.reshape(labels.shape[0], 1).to(device).float()
            drug_row_ic50, drug_col_ic50, outputs = model(reverse_drug_map, inputs)
            drug_row_ic50 = drug_row_ic50.reshape(-1, 2).float()  
            drug_col_ic50 = drug_col_ic50.reshape(-1, 2).float()
            outputs = outputs.float() 
            row_labels = row_labels.cpu().detach().float()
            col_labels = col_labels.cpu().detach().float()
            labels = labels.cpu().detach().float()
            drug_row_ic50 = drug_row_ic50.cpu().detach().float()
            drug_col_ic50 = drug_col_ic50.cpu().detach().float()
            outputs = outputs.cpu().detach().float()
            row_labels_list = np.array(row_labels).tolist()
            col_labels_list = np.array(col_labels).tolist()
            labels_list = np.array(labels)[:,0].tolist()
            outputs_list = np.array(outputs)[:,0].tolist()
            row_probs = torch.softmax(drug_row_ic50, dim=1)
            col_probs = torch.softmax(drug_col_ic50, dim=1)
            row_pred_list = torch.argmax(row_probs, dim=1).cpu().numpy().tolist()
            col_pred_list = torch.argmax(col_probs, dim=1).cpu().numpy().tolist()
            row_prob_list = row_probs[:, 1].cpu().numpy().tolist()
            col_prob_list = col_probs[:, 1].cpu().numpy().tolist()
            compare_temp = pd.DataFrame({
                'row_true': row_labels_list,
                'row_pred': row_pred_list,
                'col_true': col_labels_list,
                'col_pred': col_pred_list,
                'true': labels_list,
                'pred': outputs_list,
                'row_prob': row_prob_list,
                'col_prob': col_prob_list
            })
            compare = pd.concat([compare, compare_temp], ignore_index=True)
    results = metric(compare)
    return results
