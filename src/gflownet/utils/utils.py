import numpy as np
import math
from copy import deepcopy
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter

class batch_Dataset(Dataset):
    def __init__(self, batch):
        super().__init__()
        self.trajs = batch['trajs']
        self.cond_beta = batch['cond_info']['beta']
        self.cond_enc = batch['cond_info']['encoding']
        self.log_rewards = batch['log_rewards']
        self.flat_rewards = batch['flat_rewards']
        self.is_valid = batch['is_valid']
        
        self.t_len = len(self.trajs)
        self.cb_len = len(self.cond_beta)
        self.ce_len = len(self.cond_enc)
        self.lr_len = len(self.log_rewards)
        self.fr_len = len(self.flat_rewards)
        self.iv_len = len(self.is_valid)
        
        assert self.t_len == self.cb_len == self.ce_len == self.lr_len == self.fr_len == self.iv_len, "All lengths must be the same"
        
        
    def __getitem__(self, i):
        return self.trajs[i], self.cond_beta[i], self.cond_enc[i], self.log_rewards[i], self.flat_rewards[i], self.is_valid[i]

    def __len__(self):
        return self.t_len
    
def make_batch_dl(batch: dict, batch_size: int):
    '''
    keys: trajs, cond_info, log_rewards, flat_rewards, is_valid
    '''
    train_dataset = batch_Dataset(batch)
    dataloader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            drop_last=False,
                            shuffle=True,
                            pin_memory=True
                            )
    return dataloader

def calc_dup_dict(smis):
    '''
    Calculate the dictionary of duplicated SMILES
    '''
    smis_counts = Counter(smis)
    duplicated_smiles_dict = {}
    for smiles, count in smis_counts.items():
        if count >= 2:
            duplicated_smiles_dict[smiles] = count
    return duplicated_smiles_dict

def calc_p_B(batch, dup_dict):
    '''
    Calculate the probability of the batch
    '''
    src_smis = batch.smis_Transformer_src
    log_p_B = []
    for smi in src_smis:
        if smi in dup_dict.keys():
            log_p_B.append(math.log(1 / int(dup_dict[smi])))
        else:
            log_p_B.append(0)
    return torch.tensor(log_p_B, dtype=torch.float32)

def smi_tokenizer(smi):
    '''
    Tokenize a SMILES molecule or reaction
    '''
    import re
    pattern =  '(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

# caluculate the number of parameters
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
    

def MW_checker(mol, threshold:int = 750):
    MW = Descriptors.ExactMolWt(mol)
    if MW > threshold:
        return False
    else:
        return True

def is_empty(li):
    return all(not sublist for sublist in li)


def extract_retained_substructures(seed_smi, template_idx, r_dict):
    """
    反応テンプレートの反応が実行された場合に、保持されるべき原料（seed_smi）の部分構造を抽出する

    Args:
        seed_smi (str): 部分構造を抽出するためのSMILES文字列
        template_idx (int): 使用するテンプレートのインデックス
        r_dict (dict): テンプレートの辞書

    Returns:
        list: 抽出された部分構造のリスト
    """
    
    seed_mol = Chem.MolFromSmiles(seed_smi)
    template_str = r_dict[str(template_idx)]
    template_rxn = AllChem.ReactionFromSmarts(template_str)
    reactants = template_rxn.GetReactants()
    filters = []
    for query_mol in reactants:
        if seed_mol.HasSubstructMatch(query_mol):
            # 部分構造検索
            matches = seed_mol.GetSubstructMatches(query_mol)
            if not matches:
                continue
            for match in matches:
                atoms_to_remove_in_query = [
                    atom.GetIdx() for atom in query_mol.GetAtoms() if atom.GetAtomMapNum() == 0
                ]

                # target_mol で対応する原子を削除
                atoms_to_remove_in_target = [match[i] for i in atoms_to_remove_in_query]


                editable_mol = Chem.EditableMol(seed_mol)
                for atom_idx in sorted(atoms_to_remove_in_target, reverse=True):
                    editable_mol.RemoveAtom(atom_idx)
                    
                filters.append(editable_mol.GetMol())
    return filters


def filtering_mol(mol, filters):
    if mol is None:
        return None
    for filter in filters:  
        if mol.HasSubstructMatch(filter):
            return Chem.MolToSmiles(mol)
    return None

def sub_matching(mol, query):
    for q in query:  
        if mol.HasSubstructMatch(q):
            return True
    return False

def rxn_filter(parent_smi, template, output, r_dict):
    """
    反応テンプレートの反応が実行された場合に、原料に対して新たに追加された骨格が反応テンプレートの原料側に存在するかどうかをチェック
    """
    reactant_smi = Chem.MolToSmiles(Chem.MolFromSmiles(parent_smi))
    r_template_smarts = r_dict[str(template)]
    product = output
    product_mol = Chem.MolFromSmiles(product)
    
    # reverse reaction
    reactants_part, products_part = r_template_smarts.split('>>')
    reverse_template_smarts = products_part + '>>' + reactants_part
    reverse_rxn = AllChem.ReactionFromSmarts(reverse_template_smarts)

    # 逆反応を実行して可能な反応物群を取得
    possible_reactants = reverse_rxn.RunReactants((product_mol,))
    num = len(possible_reactants)
    if num > 0 :
        return True
    else:
        return False
    
def extract_vina(path):
    with open(path, "r") as file:
        for line in file:
            if line.startswith("REMARK VINA RESULT:"):
                vina_result = float(line.split()[3])
                return vina_result
            
def atom_exchange(flag=False, smis=None):
    '''
    smis: list of SMILES
    '''
    if flag:
        new_smis = []
        for smi in smis:
            if '[Si]' in smi or '[B]' in smi:
                smi = smi.replace('[Si]', 'C').replace('[B]', 'C')
                smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            new_smis.append(smi)
        return new_smis
    else:
        return smis
    

def clean_nan(matched_indices, matched_probs_n):
    """
    NaNを含む要素を削除する関数
    """
    # NaNを含むインデックスを取得
    nan_indices = np.isnan(matched_probs_n)
    
    # NaNを含む要素を削除
    matched_indices = np.delete(matched_indices, nan_indices)
    matched_probs_n = np.delete(matched_probs_n, nan_indices)
    
    return matched_indices, matched_probs_n

def fill_None(score_list, fill_num):
    """
    If the score_list contains NoneType, replace it with fill_num
    """
    for i in range(len(score_list)):
        if score_list[i] is None:
            score_list[i] = fill_num
    return score_list