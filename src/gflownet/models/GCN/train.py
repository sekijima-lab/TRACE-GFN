import os
import torch
from torch_geometric.data import DataLoader
import torch.nn.functional as F

from gflownet.models.GCN import mol2graph
from gflownet.models.GCN.callbacks import EarlyStopping
from gflownet.models.GCN.network import MolecularGCN
from gflownet.models.config import TrainConfig
from gflownet.data.config import DataConfig
import datetime
from tqdm.auto import tqdm
import wandb
import argparse
from pathlib import Path
import numpy as np
from rdkit import Chem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
date = datetime.datetime.now().strftime('%Y%m%d')

def train(model, optimizer, loader):
    model.train()
    loss_all = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        output = model.forward(data.x, data.edge_index, data.batch).squeeze(1)
        loss =  F.cross_entropy(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
        
    return loss_all / len(loader)


def eval(model, loader, ks=None):
    model.eval()
    score_list = []
    with torch.no_grad():
        loss_all = 0
        for data in loader:
            data = data.to(device)
            output = model.forward(data.x, data.edge_index, data.batch) # output.shape = (batch_size, vocab_size)
            loss = F.cross_entropy(output, data.y)
            loss_all += loss.item() * data.num_graphs
            if ks is not None:
                for k in ks:
                    score_list.append(topk_accuracy(data, output, k))
    return  loss_all/len(loader), score_list

def topk_accuracy(data, output, k: int):
    _, pred = output.topk(k, 1, True, True) # (k, dim=1, largest=True, sorted=True)
    pred = pred.t() # (batch, maxk) -> (maxk, batch)
    correct = pred.eq(data.y.unsqueeze(0).expand_as(pred)) # target:(batch,) -> (1, batch) -> (maxk, batch)
    # Tensor.eq: compute element-wise equality, correct: bool matrix
    score = correct.float().sum() / len(data)
    score = score.detach().item()
    return score

def get_data(data_path):
    mols, labels = [], []
    with open(data_path, 'r') as f:
        smis = f.read().splitlines()
    for smi in smis:
        smi = smi.split(' ')
        labels.append(int(smi[0].strip('[]')))
        smi = smi[1:]
        smi = ''.join(smi)
        mols.append(Chem.MolFromSmiles(smi)) 
    return np.array(mols), np.array(labels)

def main():
    cfg = TrainConfig()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--backward', '-b', action='store_true', help='backward policy')
    args = parser.parse_args()
    backward = args.backward
    if backward:
        print('training backward policy')
    else:
        print('training forward policy')
    cfg = TrainConfig()
    data_cfg = DataConfig()
    
    wandb.init(project='GCN_py3.11',
               config={
                   'backward': backward,
                   'd_Transformer': cfg.d_Transformer,
                   'num_encoder_layers': cfg.num_encoder_layers,
                   'num_decoder_layers': cfg.num_decoder_layers,
                   'nhead': cfg.nhead,
                   'dropout_Transformer': cfg.dropout_Transformer,
                   'dim_ff': cfg.dim_ff,
                   'batch_size_Transformer': cfg.batch_size_Transformer,
               }
               )
    
    print('Loading data...')
    if not backward:
            train_path=data_cfg.USPTO_path / 'src_train.txt'
            valid_path=data_cfg.USPTO_path / 'src_valid.txt'
    else:
            train_path=data_cfg.USPTO_path / 'src_train_product.txt'
            valid_path=data_cfg.USPTO_path / 'src_valid_product.txt'    
    batch_size = cfg.batch_size_GCN
    dim = cfg.dim_GCN
    n_conv_hidden = cfg.n_conv_hidden
    n_mlp_hidden = cfg.n_mlp_hidden
    dropout = cfg.dropout_GCN
    lr = cfg.lr_GCN        
    epochs = cfg.epochs_GCN
    patience = cfg.patience_GCN
    ks = [1, 3, 5, 10]
    ckpt_dir = Path(__file__).parent / 'ckpts' / f'checkpoints_{date}'
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f'{ckpt_dir} was created.')
    
    mols_train, y_train = get_data(train_path)
    mols_valid, y_valid = get_data(valid_path)
    
    print('-'*100)
    print('Training: ', mols_train.shape)
    print('Validation: ', mols_valid.shape)
    print('-'*100)

    labels = y_train.tolist() + y_valid.tolist()
    
    # Mol to Graph
    print('Converting mol to graph...')
    X_train = [mol2graph.mol2vec(m) for m in tqdm(mols_train.tolist())]
    for i, data in enumerate(X_train):
        data.y = torch.LongTensor([y_train[i]]).to(device)
    X_valid = [mol2graph.mol2vec(m) for m in tqdm(mols_valid.tolist())]
    for i, data in enumerate(X_valid):
        data.y = torch.LongTensor([y_valid[i]]).to(device)
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(X_valid, batch_size=batch_size, shuffle=True, drop_last=True)
    print('completed.')
    print('-'*100)

    # Model instance construction
    print('Model instance construction')
    model = MolecularGCN(
        dim = dim,
        n_conv_hidden = n_conv_hidden,
        n_mlp_hidden = n_mlp_hidden,
        dropout = dropout
        ).to(device)
    print(model)
    print('-'*100)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    earlystopping = EarlyStopping(patience=patience, path=ckpt_dir)
    for epoch in range(1, epochs+1):
        # training
        train_loss = train(model, optimizer, train_loader)

        # performance evaluation
        loss_train, _ = eval(model, train_loader)
        loss_valid, score_list = eval(model, valid_loader, ks=ks)
        top1acc = score_list[0]
        top3acc = score_list[1]
        top5acc = score_list[2]
        top10acc = score_list[3]
        
        print(f'Epoch: {epoch}/{epochs}, loss_train: {loss_train:.5}, loss_valid: {loss_valid:.5}')
        print(f'top k accuracy: top1={top1acc:.2}, top3={top3acc:.2}, top5={top5acc:.2}, top10={top10acc:.2}')
        # early stopping detection
        earlystopping(loss_valid, model, epoch)
        if earlystopping.early_stop:
            print('Early Stopping!')
            print('-'*100)
            break

if __name__ == '__main__':
    main()