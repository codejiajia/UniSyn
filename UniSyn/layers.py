import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, max_pool,GINConv, JumpingKnowledge, global_max_pool
from torch_geometric.data import Batch
import math

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        Q, K, V = self.qkv_proj(x).chunk(3, dim=-1)
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.norm(self.out_proj(attn_output) + x), attn_weights

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, query, key, value, mask=None):
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], -1)
        return self.norm(self.out_proj(attn_output) + query), attn_weights

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return self.norm(x + self.dropout(F.gelu(self.fc2(self.fc1(x)))))

class EncoderD2C(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout=0.1):
        super(EncoderD2C, self).__init__()
        self.self_attn_drug = SelfAttention(hidden_size, num_heads, dropout)
        self.self_attn_cell = SelfAttention(hidden_size, num_heads, dropout)
        self.cross_attn = CrossAttention(hidden_size, num_heads, dropout)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)

    def forward(self, drug_modalities, cell_modalities, mask_drug=None, mask_cell=None):
        drug_fused, drug_self_attn = self.self_attn_drug(drug_modalities)
        cell_fused, cell_self_attn = self.self_attn_cell(cell_modalities)
        cell_out, cell_cross_attn = self.cross_attn(cell_fused, drug_fused, drug_fused, mask_drug)
        drug_out, drug_cross_attn = self.cross_attn(drug_fused, cell_fused, cell_fused, mask_cell)
        cell_out = self.ffn(cell_out)
        drug_out = self.ffn(drug_out)
        return cell_out, drug_out, drug_self_attn, cell_self_attn, drug_cross_attn, cell_cross_attn

class GNN_cell(torch.nn.Module):
    def __init__(self, num_feature, layer_cell, dim_cell, cluster_predefine):
        super().__init__()
        self.num_feature = num_feature
        self.layer_cell = layer_cell
        self.dim_cell = dim_cell
        self.cluster_predefine = cluster_predefine
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        self.device = torch.device(0)
        for i in range(self.layer_cell):
            if i:
                conv = GATConv(self.dim_cell, self.dim_cell,heads=1)
            else:
                conv = GATConv(self.num_feature, self.dim_cell,heads=1)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False)
            self.convs_cell.append(conv)
            self.bns_cell.append(bn)

    def forward(self, cell):
        cell_batch= Batch.from_data_list(cell)
        print(cell_batch)
        for i in range(self.layer_cell):
            cell_batch.edge_index = cell_batch.edge_index.to(self.device)
            print(f"Before GATConv: cell_batch.x.shape={cell_batch.x.shape}, edge_index.shape={cell_batch.edge_index.shape}")
            if torch.isnan(cell_batch.x).any() or torch.isinf(cell_batch.x).any():
                raise ValueError("cell_batch.x contains NaN or Inf before GATConv!")
            if torch.isnan(cell_batch.edge_index).any() or torch.isinf(cell_batch.edge_index).any():
                raise ValueError("cell_batch.edge_index contains NaN or Inf before GATConv!")
            cell_batch.x = F.relu(self.convs_cell[i](cell_batch.x, cell_batch.edge_index))
            num_node = int(cell_batch.size(0) / cell_batch.num_graphs)
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell_batch.num_graphs)])
            cell_batch = max_pool(cluster.cpu(), cell_batch.cpu(), transform=None).to(self.device)
            cell_batch.x = self.bns_cell[i](cell_batch.x)
        node_representation = cell_batch.x.reshape(-1, self.final_node * self.dim_cell)
        return node_representation
    
    def grad_cam(self, cell):
        cell_batch= Batch.from_data_list(cell)
        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell_batch.x, cell_batch.edge_index))
            if i == 0:
                cell_node = cell.x
                cell_node.retain_grad()
            num_node = int(cell.x.size(0) / cell.num_graphs)
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            cell = max_pool(cluster, cell, transform=None)
            cell.x = self.bns_cell[i](cell.x)
        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell)
        return cell_node, node_representation

class GNN_drug(torch.nn.Module):
    def __init__(self, layer_drug, dim_drug):
        super().__init__()
        self.layer_drug = layer_drug
        self.dim_drug = dim_drug
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()
        for i in range(self.layer_drug):
            if i:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(77, self.dim_drug), nn.ReLU(), nn.Linear(self.dim_drug, self.dim_drug))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)
            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

    def forward(self, drug):
        batched_data = Batch.from_data_list(drug)
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        x_drug_list = []
        for i in range(self.layer_drug):
            x = F.relu(self.convs_drug[i](x, edge_index))
            x = self.bns_drug[i](x)
            x_drug_list.append(x)
        node_representation = self.JK(x_drug_list)
        x_drug = global_max_pool(node_representation, batch)
        return x_drug 

class MutanLayer(nn.Module):
    def __init__(self, dim, multi,modal1_dim, modal2_dim, modal3_dim):
        super(MutanLayer, self).__init__()
        self.dim = dim
        self.multi = multi
        self.modal1_proj = nn.Linear(modal1_dim, dim)
        self.modal2_proj = nn.Linear(modal2_dim, dim)
        self.modal3_proj = nn.Linear(modal3_dim, dim)
        modal1 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)
        modal2 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)
        modal3 = []
        for i in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(dim, dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        bs = modal1_emb.size(0)
        modal1_emb = self.modal1_proj(modal1_emb)
        modal2_emb = self.modal2_proj(modal2_emb)
        modal3_emb = self.modal3_proj(modal3_emb)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_mm.append(x_modal1 + x_modal2 + x_modal3)
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(bs, self.dim)
        x_mm = torch.relu(x_mm)
        return x_mm



