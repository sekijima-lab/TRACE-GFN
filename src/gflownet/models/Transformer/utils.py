import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.init import xavier_uniform_
import torchtext.vocab.vocab as Vocab
from gflownet.models.Transformer.model import Transformer
from gflownet.utils.utils import smi_tokenizer


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
import torchtext.transforms as T
import torch.optim as optim
import torch.nn.functional as F


class smi_Dataset(Dataset):
    def __init__(self, src, tgt):
        super().__init__()
        self.src = src
        self.tgt = tgt
        
    def __getitem__(self, i):
        src = self.src[i]
        tgt = self.tgt[i]
        return src, tgt
    
    def __len__(self):
        return len(self.src)

def make_dataloader(src_train, tgt_train, src_transforms, tgt_transforms, batch_size):
    '''
    datasets: output of make_counter()
    transforms: output of make_vocab()
    '''
    src_train = src_transforms(src_train)
    tgt_train = tgt_transforms(tgt_train)
    train_dataset = smi_Dataset(src=src_train, tgt=tgt_train)

    dataloader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            drop_last=False,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True
                            )
    return dataloader

def predict_Transformer(cfg, model, batch, src_transforms, tgt_transforms, data_dict, vocab, dev):
    '''
    samplingされたtrajectoryの文字列について、予測確率のlogの和を返す
    return preds: list[float], len(preds) = len(batch)
    '''
    
    modelcfg = cfg.model
    batch_size = modelcfg.glb_batch_size    
    Transformer_src = batch.smis_Transformer_src
    Transformer_tgt = batch.smis_Transformer_tgt
    templates = batch.templates
    # 'src, tgt, templates must have the same length'
    assert len(Transformer_src) == len(Transformer_tgt)
    assert len(Transformer_src) == len(templates)
    model.to(dev)
    src_data = []
    tgt_data = []

    for smi, template in zip(Transformer_src, templates):
        src_str = f'[{template}]' + smi
        src_list = smi_tokenizer(src_str).split(' ')
        src_data.append(src_list)
    
    for tgt_str in Transformer_tgt:
        tgt_list = smi_tokenizer(tgt_str).split(' ')
        tgt_data.append(tgt_list)
    
    train_dataloader = make_dataloader(src_train=src_data, tgt_train=tgt_data, 
                                       src_transforms=src_transforms, tgt_transforms=tgt_transforms, 
                                       batch_size=batch_size
                                       )
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(data_dict['tgt_max_len']-1).to(dev)
    # print('start training...')
    preds = []
    for i, data in enumerate(train_dataloader):
        src, tgt = data[0].to(dev).permute(1, 0), data[1].to(dev).permute(1, 0)
        tgt_input = tgt[:-1, :] # (seq, batch)
        tgt_output = tgt[1:, :] # shifted right
        outputs = model(src=src, tgt=tgt_input, tgt_mask=tgt_mask,
                        src_pad_mask=True, tgt_pad_mask=True, memory_pad_mask=True) # out: (seq_length, batch_size, vocab_size)
        outputs = F.log_softmax(outputs, dim=-1) # (seq_length, batch_size, vocab_size)
        outputs = outputs.permute(1, 0, 2) # (batch_size, seq_length, vocab_size)
        tgt_output = tgt_output.permute(1, 0) # (batch_size, seq_length)
        for i in range(outputs.size(0)):
            batch_preds = 0
            ans = tgt_output[i].squeeze().tolist() # (seq_length, )
            tensor = outputs[i].squeeze() # (seq_length, vocab_size)
            for j in range(tensor.size(0)):
                pred = tensor[j][ans[j]]
                batch_preds += pred
                if ans[j] == vocab.get_stoi()['<eos>']:
                    break
            preds.append(batch_preds)
    return torch.stack(preds).to(dev) 


'''legacy code'''
# def predict_Transformer(cfg, model, batch, src_transforms, tgt_transforms, data_dict, vocab, dev):
#     '''
#     samplingされたtrajectoryの文字列について、予測確率のlogの和を返す
#     return preds: list[float], len(preds) = len(batch)
#     '''
    
#     modelcfg = cfg.model
#     root_smi = batch.root_smi
#     smis = batch.smis_Transformer # root_smiを除去したものなのでtgtとして使う
#     templates = batch.templates
#     model.to(dev)
#     # lr = modelcfg.lr_Transformer
#     # betas = modelcfg.betas_Transformer
#     batch_size = modelcfg.glb_batch_size
#     src_strs = []
#     src_data, tgt_data = [], []

#     # raw: a1, smi1, a2, smi2, ...
#     # smis: smi1, smi2, ...
#     # templates: a1, a2, ...
    
#     src_strs.append(root_smi)
#     src_strs.extend(smis[:-1])
#     for smi, template in zip(src_strs, templates):
#         src_str = f'[{template}]' + smi
#         src_list = smi_tokenizer(src_str).split(' ')
#         src_data.append(src_list)
    
#     for tgt_str in smis:
#         tgt_list = smi_tokenizer(tgt_str).split(' ')
#         tgt_data.append(tgt_list)
    
#     train_dataloader = make_dataloader(src_train=src_data, tgt_train=tgt_data, 
#                                        src_transforms=src_transforms, tgt_transforms=tgt_transforms, 
#                                        batch_size=batch_size
#                                        )
#     tgt_mask = nn.Transformer.generate_square_subsequent_mask(data_dict['tgt_max_len']-1).to(dev)
#     # print('start training...')
#     preds = []
#     for i, data in enumerate(train_dataloader):
#         src, tgt = data[0].to(dev).permute(1, 0), data[1].to(dev).permute(1, 0)
#         tgt_input = tgt[:-1, :] # (seq, batch)
#         tgt_output = tgt[1:, :] # shifted right
#         outputs = model(src=src, tgt=tgt_input, tgt_mask=tgt_mask,
#                         src_pad_mask=True, tgt_pad_mask=True, memory_pad_mask=True) # out: (seq_length, batch_size, vocab_size)
#         outputs = F.log_softmax(outputs, dim=-1) # (seq_length, batch_size, vocab_size)
#         outputs = outputs.permute(1, 0, 2) # (batch_size, seq_length, vocab_size)
#         tgt_output = tgt_output.permute(1, 0) # (batch_size, seq_length)
#         for i in range(outputs.size(0)):
#             batch_preds = 0
#             ans = tgt_output[i].squeeze().tolist() # (seq_length, )
#             tensor = outputs[i].squeeze() # (seq_length, vocab_size)
#             for j in range(tensor.size(0)):
#                 pred = tensor[j][ans[j]]
#                 batch_preds += pred
#                 if ans[j] == vocab.get_stoi()['<eos>']:
#                     break
#             preds.append(batch_preds)
#     return torch.stack(preds).to(dev) 


class TransformerLR(_LRScheduler):
    """TransformerLR class for adjustment of learning rate.

    The scheduling is based on the method proposed in 'Attention is All You Need'.
    """

    def __init__(self, optimizer, warmup_epochs=8000, last_epoch=-1, verbose=False):
        """Initialize class."""
        self.warmup_epochs = warmup_epochs
        self.normalize = self.warmup_epochs**0.5
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Return adjusted learning rate."""
        step = self.last_epoch + 1
        scale = self.normalize * min(step**-0.5, step * self.warmup_epochs**-1.5)
        return [base_lr * scale for base_lr in self.base_lrs]    

def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec


class EarlyStopping:
    def __init__(self, patience=10, ckpt_dir=None):
        '''引数: 最小値の非更新数カウンタ、表示設定、モデル格納path'''

        self.patience = patience    #設定ストップカウンタ
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = ckpt_dir         #ベストモデル格納path

    def __call__(self, val_loss, step, optimizer, cur_loss, model):
        '''
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        '''
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, step, optimizer, cur_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            print(f'Validation loss increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.checkpoint(val_loss, step, optimizer, cur_loss, model)
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            print(f'Validation loss decreased! ({self.val_loss_min:.6f} --> {val_loss:.6f}) Saving model ...')
            self.checkpoint(val_loss, step, optimizer, cur_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, step, optimizer, cur_loss, model):
        torch.save({'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': cur_loss,}, f'{self.path}/ckpt_{step+1}.pth')
        self.val_loss_min = val_loss  #その時のlossを記録する

class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0 # latest value
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, batch_size, v=None):
    '''
    Computes the accuracy of top1 prediction
    
    output: (seq_length*batch_size, num_tokens)
    target: (seq_length*batch_size)
    '''
    
    pad_mask = (target != v['<pad>']) # padはFalse, それ以外はTrue
    true_pos = torch.nonzero(pad_mask).squeeze().tolist()
    out_extracted = output[true_pos]
    t_extracted = target[true_pos]
    _, pred = out_extracted.topk(1, 1, True, True) # arg of topk: (k, dim=1, largest=True, sorted=True)
    pred = pred.t() # (seq*batch, maxk) -> (maxk, seq*batch)
    correct = pred.eq(t_extracted.reshape(1, -1).expand_as(pred)) # target:(seq*batch, 1) -> (1, seq*batch) -> (maxk, seq*batch)
    # Tensor.eq: compute element-wise equality, correct: bool matrix
    correct_rate = (correct[0].float().sum(0, keepdim=True)) / len(t_extracted)
    
    # compute accuracy per whole molecule
    target = target.reshape(-1, batch_size)
    output = output.reshape(-1, batch_size, v.__len__())
    _, pred = output.topk(1, 2, True, True)
    pred = pred.squeeze() # (seq, batch) -> (batch, seq)
    correct_cum = 0
    EOS_token = v['<eos>']
    for i in range(batch_size):
        t = target[:, i].tolist()
        eos_idx = t.index(EOS_token)
        t = t[0:eos_idx]
        p = pred[:, i].tolist()
        p = p[0:len(t)]
        if t == p:
            correct_cum += 1
    perfect_acc = correct_cum / batch_size
    return correct_rate.item(), perfect_acc

def calc_topk_perfect_acc(x, target, batch_size, EOS):
    '''
    x: predicted tensor of shape (seq, batch, k)
    target: (seq, batch)
    '''
    correct_cum = 0
    if x.dim() < 3:
        x = x.unsqueeze(-1)
    for i in range(batch_size):
        t = target[:, i].tolist()
        eos_idx = t.index(EOS)
        t = t[0:eos_idx]
        for j in range(x.size(2)):
            p = x[:, i, j].tolist()
            p = p[0:len(t)]
            if t == p:
                correct_cum += 1
                break
    return correct_cum / batch_size

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
