import os
import torch
from torch_geometric.data import DataLoader
import torch.nn.functional as F

from gflownet.models.GCN import mol2graph
from gflownet.models.GCN.callbacks import EarlyStopping
from gflownet.models.GCN.network import MolecularGCN
from gflownet.models.GCN.utils import get_data

import datetime

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# date = datetime.datetime.now().strftime('%Y%m%d')

# def train(model, optimizer, loader):
#     model.train()
#     loss_all = 0
#     for data in loader:
#         optimizer.zero_grad()
#         data = data.to(device)
#         output = model.forward(data.x, data.edge_index, data.batch).squeeze(1)
#         loss =  F.cross_entropy(output, data.y)
#         loss.backward()
#         loss_all += loss.item() * data.num_graphs
#         optimizer.step()
        
#     return loss_all / len(loader)

# def eval(model, loader, ks=None):
#     model.eval()
#     score_list = []
#     with torch.no_grad():
#         loss_all = 0
#         for data in loader:
#             data = data.to(device)
#             output = model.forward(data.x, data.edge_index, data.batch) # output.shape = (batch_size, vocab_size)
#             loss = F.cross_entropy(output, data.y)
#             loss_all += loss.item() * data.num_graphs
#             if ks is not None:
#                 for k in ks:
#                     score_list.append(topk_accuracy(data, output, k))
#     return  loss_all/len(loader), score_list

def topk_accuracy(data, output, k: int):
    _, pred = output.topk(k, 1, True, True) # (k, dim=1, largest=True, sorted=True)
    pred = pred.t() # (batch, maxk) -> (maxk, batch)
    correct = pred.eq(data.y.unsqueeze(0).expand_as(pred)) # target:(batch,) -> (1, batch) -> (maxk, batch)
    # Tensor.eq: compute element-wise equality, correct: bool matrix
    score = correct.float().sum() / len(data)
    score = score.detach().item()
    return score


def predict_GCN(cfg, batch, GCN_model, device):
    batch_size = cfg.model.batch_size
    lr = cfg.model.lr_GCN
    mols, labels = get_data(batch)
    X = [mol2graph.mol2vec(m) for m in mols.tolist()]
    for i, data in enumerate(X):
        data.y = torch.LongTensor([labels[i]]).to(device)
    loader = DataLoader(X, batch_size=batch_size, shuffle=True, drop_last=True)
    # inference
    optimizer = torch.optim.Adam(GCN_model.parameters(), lr=lr)
    for data in loader:
        data = data.to(device)
        output = GCN_model.forward(data.x, data.edge_index, data.batch).squeeze(1)
        