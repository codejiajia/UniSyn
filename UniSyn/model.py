import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch_geometric.nn import HypergraphConv, GINConv,GATConv, max_pool
import os
from layers import GNN_drug,GNN_cell,MutanLayer,EncoderD2C
import config

class Net_View1(nn.Module):
    def __init__(self, model_config, drug_fp, drug_seq, drug_gra, cell_exp, cell_mu, cell_nv):
        super(Net_View1, self).__init__()
        hidden_size = model_config['hidden_size']
        dropout = model_config['dropout']
        gpu = model_config['gpu']
        dim_cell_exp = model_config['dim_cell_exp']
        dim_cell_mu = model_config['dim_cell_mu']
        dim_cell_nv = model_config['dim_cell_nv']
        layer_drug = model_config['layer_drug']
        dim_drug_fp = model_config['dim_drug_fp']
        dim_drug_seq = model_config['dim_drug_seq']
        dim_drug_gra = model_config['dim_drug_gra']
        dim_drug_gnn = model_config['dim_drug_gnn']
        intermediate_size = hidden_size * 2
        num_attention_heads = model_config['num_attention_heads']
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(gpu)
        else:
            self.device = torch.device('cpu')
        self.drug_fp = torch.Tensor(np.array(drug_fp)).to(self.device)
        self.drug_seq = torch.Tensor(np.array(drug_seq)).to(self.device)
        self.drug_gra = drug_gra
        self.drug_gra = {key: value.to(self.device) for key, value in self.drug_gra.items()}
        self.GNN_drug = GNN_drug(layer_drug, dim_drug_gnn)
        self.cell_exp = torch.Tensor(np.array(cell_exp)).to(self.device)
        self.cell_mu = torch.Tensor(np.array(cell_mu)).to(self.device)
        self.cell_nv = torch.Tensor(np.array(cell_nv)).to(self.device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.cell_fc_exp = nn.Linear(dim_cell_exp, hidden_size)
        nn.init.kaiming_normal_(self.cell_fc_exp.weight)
        self.cell_fc_mu = nn.Linear(dim_cell_mu, hidden_size)
        nn.init.kaiming_normal_(self.cell_fc_mu.weight)
        self.cell_fc_nv = nn.Linear(dim_cell_nv, hidden_size)
        nn.init.kaiming_normal_(self.cell_fc_nv.weight)
        self.drug_fc_fp = nn.Linear(dim_drug_fp, hidden_size)
        nn.init.kaiming_normal_(self.drug_fc_fp.weight)
        self.drug_fc_seq = nn.Linear(dim_drug_seq, hidden_size)
        nn.init.kaiming_normal_(self.drug_fc_seq.weight)
        self.drug_fc_gra = nn.Linear(dim_drug_gra, hidden_size)
        nn.init.kaiming_normal_(self.drug_fc_gra.weight)
        self.norm = nn.LayerNorm(hidden_size)
        self.drug_cell_CA = EncoderD2C(hidden_size, intermediate_size, num_attention_heads, dropout)
        self.fp_cell_fc1 = nn.Linear(hidden_size * 6, hidden_size*3)
        nn.init.kaiming_normal_(self.fp_cell_fc1.weight)
        self.fp_cell_fc2 = nn.Linear(hidden_size * 3, hidden_size)
        nn.init.kaiming_normal_(self.fp_cell_fc2.weight)
    def forward(self, reverse_drug_map,drug_cell):
        drug_cell = drug_cell.long()
        batch_size = drug_cell.shape[0]
        num_modalities = 3
        seq_len = 1
        cell_exp = self.cell_exp[drug_cell[:, 1]].view(batch_size, seq_len, -1)
        cell_mu = self.cell_mu[drug_cell[:, 1]].view(batch_size, seq_len, -1)
        cell_nv = self.cell_nv[drug_cell[:, 1]].view(batch_size, seq_len, -1)
        drug_fp = self.drug_fp[drug_cell[:, 0]].view(batch_size, seq_len, -1)
        drug_seq = self.drug_seq[drug_cell[:, 0]].view(batch_size, seq_len, -1)
        drug_names = [reverse_drug_map[int(idx)] for idx in drug_cell[:,0]]
        drug_gra_input = [self.drug_gra[name] for name in drug_names]
        drug_gra = self.GNN_drug(drug_gra_input)
        cell_exp = self.cell_fc_exp(cell_exp).squeeze(1)
        cell_mu = self.cell_fc_mu(cell_mu).squeeze(1)
        cell_nv = self.cell_fc_nv(cell_nv).squeeze(1)
        drug_fp = self.drug_fc_fp(drug_fp).squeeze(1)
        drug_seq = self.drug_fc_seq(drug_seq).squeeze(1)
        drug_gra = self.drug_fc_gra(drug_gra)
        c = torch.stack([cell_exp, cell_mu, cell_nv], dim=1)
        d = torch.stack([drug_fp, drug_seq, drug_gra], dim=1)
        c = self.norm(c)
        d = self.norm(d)
        cell_out, drug_out, drug_self_attn, cell_self_attn, drug_cross_attn, cell_cross_attn = self.drug_cell_CA(d, c, None, None)
        cell_out = cell_out.view(cell_out.size(0), -1)
        drug_out = drug_out.view(drug_out.size(0), -1)
        x_raw = torch.cat((cell_out, drug_out), dim=1)
        x = self.relu(self.fp_cell_fc1(x_raw))
        x = self.dropout(x)
        x = self.relu(self.fp_cell_fc2(x))
        x = self.dropout(x)
        return x_raw, x, drug_cross_attn, cell_cross_attn

class IC50Predictor(nn.Module):
    def __init__(self, ic50_layers, drop=0.1):
        super(IC50Predictor, self).__init__()
        layers = []
        for i in range(len(ic50_layers) - 1):
            linear_layer=nn.Linear(ic50_layers[i], ic50_layers[i + 1])
            nn.init.kaiming_normal_(linear_layer.weight)
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop))
        self.output_layer = nn.Linear(ic50_layers[-1], 2)
        nn.init.kaiming_normal_(self.output_layer.weight)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return  self.output_layer(x)

class SynergyPredictor(nn.Module):
    def __init__(self, synergy_layers, drop=0.1):
        super(SynergyPredictor, self).__init__()
        layers = []
        for i in range(len(synergy_layers) - 1):
            linear_layer=nn.Linear(synergy_layers[i], synergy_layers[i + 1])
            nn.init.kaiming_normal_(linear_layer.weight)
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop))
        self.output_layer = nn.Linear(synergy_layers[-1], 1)
        nn.init.kaiming_normal_(self.output_layer.weight)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return self.output_layer(x)

class Net(nn.Module):
    def __init__(self, model_config, drug_fp, cell, model_View1,layers):
        super(Net, self).__init__()
        gpu = model_config['gpu']
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(gpu)
        else:
            self.device = torch.device('cpu')
        self.model_View1 = model_View1
        self.model_View1_A = model_View1
        self.model_View1_B = model_View1
        self.ic50_predictor1 = IC50Predictor( layers['ic50_layers'])
        self.ic50_predictor2 = IC50Predictor( layers['ic50_layers'])
        self.synergy_predictor = SynergyPredictor( layers['synergy_layers'])
        self.frozen = False
        
    def forward(self, reverse_drug_map,triplets):
        triplets = triplets.long()
        x1_raw,x1A,drug_cross_attn_A, cell_cross_attn_A  = self.model_View1_A(reverse_drug_map,triplets[:,[0,2]])
        x2_raw,x1B,drug_cross_attn_B, cell_cross_attn_B = self.model_View1_A(reverse_drug_map,triplets[:,[1,2]])
        if self.frozen:
            x1_raw_detached = x1_raw.detach()
            x2_raw_detached = x2_raw.detach()
            x1A_detached = x1A.detach()
            x1B_detached = x1B.detach()
            drug_row_ic50 = self.ic50_predictor1(x1A_detached)
            drug_col_ic50 = self.ic50_predictor1(x1B_detached)
        else:
            drug_row_ic50 = self.ic50_predictor1(x1_raw)
            drug_col_ic50 = self.ic50_predictor1(x2_raw)
        x_view1 = torch.cat((x1_raw, x2_raw), 1)
        x = x_view1
        synergy_loewe = self.synergy_predictor(x)
        return drug_row_ic50 ,drug_col_ic50 ,synergy_loewe

   