from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
import torch.nn.functional as F
from gflownet.models.GCN import mol2graph
from torch_geometric.data import DataLoader
import itertools


def get_data(batch):
    labels = batch.templates
    smis = batch.smis_GCN   
    return np.array(smis), np.array(labels)


def template_prediction(GCN_model, input_smi, num_sampling, GCN_device=None):
    try:
        mol = Chem.MolFromSmiles(input_smi)
        data = mol2graph.mol2vec(mol).to(GCN_device)
        with torch.no_grad():
            output = GCN_model.forward(data.x, data.edge_index, data.batch).squeeze() # shape(1, 1000) -> (1000,)
            probs, indices = torch.topk(output, num_sampling)
    except:
        probs, indices = None, None
    return probs, indices

def batch_template_prediction(GCN_model, input_smi, num_sampling=5, GCN_device=None):
    mol = Chem.MolFromSmiles(input_smi)
    data = mol2graph.mol2vec(mol).to(GCN_device)
    output = GCN_model.forward(data.x, data.edge_index, data.batch).squeeze() # shape(1, 1000) -> (1000,)
    _, indices = torch.topk(output, num_sampling)
    return indices
    
def check_templates(probs, indices, input_smi, r_dict) -> list[int]:
    matched_probs = []
    matched_indices = []
    molecule = Chem.MolFromSmiles(input_smi)
    for i, p  in zip(indices, probs):
        idx = str(i.item())
        rsmi = r_dict[idx]
        rxn = AllChem.ReactionFromSmarts(rsmi)
        reactants = rxn.GetReactants()
        flag = False
        for reactant in reactants:
            if molecule.HasSubstructMatch(reactant):
                flag = True
                break
        if flag == True:
            matched_probs.append(p.item())
            matched_indices.append(i.item())
    return matched_probs, matched_indices # list of int, ex) [0, 123, ..., 742]


def predict_GCN(cfg, batch, GCN_model, device):
    '''
    GCN_modelがbatch.smisからbatch.templatesを予測する確率の計算
    return probs: list[float], len(probs) = len(batch.smis)
    '''
    batch_size = cfg.model.glb_batch_size
    smis, labels = get_data(batch)
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    X = [mol2graph.mol2vec(m) for m in mols]
    for i, data in enumerate(X):
        data.y = torch.LongTensor([labels[i]]).to(device)
    loader = DataLoader(X, batch_size=batch_size, shuffle=False, drop_last=False)
    # inference
    probs = []
    for data in loader:
        data = data.to(device)
        if data.y.shape[0] == 1:
            data.x_repeat = data.x.repeat(2, 1)
            data.edge_index_repeat = data.edge_index.repeat(1, 2)
            data.batch_dup = torch.ones(data.batch.size(0)).to(device).long()
            data.batch_repeat = torch.cat([data.batch, data.batch_dup], dim=0).long()
            # GCN_modelはforwardのoutputがlog_softmax
            output = GCN_model.forward(data.x_repeat, data.edge_index_repeat, data.batch_repeat).squeeze() # shape(batch_size, 1000)
            out = output[0][data.y[0]]
            probs.append(out)
        else:
            output = GCN_model.forward(data.x, data.edge_index, data.batch).squeeze() # shape(batch_size, 1000)
            for i in range(output.size(0)):
                out = output[i]
                correct_idx = data.y[i]
                out_i = out[correct_idx]
                probs.append(out_i)
    probs = torch.stack(probs).to(device)
    return probs