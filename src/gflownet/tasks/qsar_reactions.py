import os
import sys
import shutil
import socket
from typing import Callable, Dict, List, Tuple, Union
import pickle
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import profilers
from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor

from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from gflownet.config import Config
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.utils import extract_vina, atom_exchange, fill_None

import datetime
import time
import wandb


class QSAR_ReactionTask(GFNTask):
    """
    Sets up a task where the reward is computed using a proxy for the binding energy of a molecule to
    Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.

    This setup essentially reproduces the results of the Trajectory Balance paper when using the TB
    objective, or of the original paper when using Flow Matching.
    """

    def __init__(   
        self,
        cfg: Config,
        rng: np.random.Generator = None, # random number generator
        wrap_model: Callable[[nn.Module], nn.Module] = None, # モデルをラップする関数, multiprocessing用, synflownetでは使用しないのでNoneのまま
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.cfg = cfg
        self.models = self._load_task_models()
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size() # default: 32, num_cond_dimと一致している必要がある？
        if self.cfg.task.add_sa_score:
            print('Adding SA score to reward with coefficient', self.cfg.task.sa_score_coeff)
        
    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        # return FlatRewards(torch.as_tensor(y) / 8)
        return FlatRewards(torch.as_tensor(y))

    # FlatRewards型を報酬値に変換
    def inverse_flat_reward_transform(self, rp: FlatRewards) -> float:
        # return rp * 8
        return rp

    def _load_task_models(self):
        with open(Path(__file__).parent.parent / 'models' / f'qsar_{self.cfg.protein_name}_optimized.pkl', 'rb') as f:
            qsar_model = pickle.load(f)
        qsar_model, self.device = self._wrap_model(qsar_model, send_to_device=False) 
        return {"qsar": qsar_model}
    
    def sample_conditional_information(self, n: int,
                                       train_it: int,
                                       final: bool = False) -> Dict[str, Tensor]:
        if final:
            cfg = self.cfg
            cfg.cond.temperature.sample_dist = "constant"
            cfg.cond.temperature.dist_params = [1.0]
            self.temperature_conditional = TemperatureConditional(cfg, self.rng)
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, flat_reward))
    
    def scores_to_flat_rewards(self, scores: List) -> FlatRewards:
        scores = torch.tensor(scores)
        scores = self.flat_reward_transform_docking(scores).clip(1e-4, 1).reshape((-1, 1))
        return scores
    
    def list_to_flat_rewards(self, scores: List) -> FlatRewards:
        scores = torch.tensor(scores)
        scores = FlatRewards(torch.as_tensor(scores)).reshape((-1, 1))
        return scores
    
    def scale_sa_score(self, sa_scores: Tensor) -> Tensor:
        """
        Scales the SA (synthetic accessibility) scores to a range between 0 and 1.
        """
        sa_scores = (self.cfg.task.sa_max - sa_scores) / (self.cfg.task.sa_max - self.cfg.task.sa_min)
        sa_scores = sa_scores.clip(0, 1)
        return sa_scores
    
    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        is_valid = torch.tensor([m is not None for m in mols]).bool() 

        if not is_valid.any(): 
            return FlatRewards(torch.zeros((0, 1))), is_valid

        ecfps = []
        None_indices = []
        for i, mol in enumerate(mols):
            if mol is not None:
                ecfps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))
            else:
                None_indices.append(i)
                ecfps.append([0]*2048)
        ecfp6_array = np.array(ecfps)
        X = pd.DataFrame(ecfp6_array, columns=[f'bit_{i}' for i in range(2048)])
        y_pred = self.models['qsar'].predict_proba(X)[:, 1]
        for None_idx in None_indices:
            y_pred[None_idx] = 0
        y_pred = torch.tensor(y_pred)
        preds = self.flat_reward_transform(y_pred).clip(1e-4, 1).reshape((-1, 1))
        
        if self.cfg.task.add_sa_score:
            sa_scores = []
            for mol in mols:
                if mol is not None:
                    sa_score = sascorer.calculateScore(mol)
                sa_scores.append(sa_score)
            sa_scores = torch.tensor(sa_scores).reshape((-1, 1))
            sa_scores = self.scale_sa_score(sa_scores)
            preds += self.cfg.task.sa_score_coeff * self.flat_reward_transform(sa_scores)
        return FlatRewards(preds), is_valid

class QSAR_ReactionTrainer(StandardOnlineTrainer):
    task: QSAR_ReactionTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 0
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = 'norm'
        cfg.opt.clip_grad_param = 10
        cfg.algo.global_batch_size = 64
        cfg.algo.offline_ratio = 0
        cfg.algo.method = 'TB'
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.9
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_sample_p_b = False

    def setup_task(self):
        self.task = QSAR_ReactionTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--protein_name', type=str, required=True)
    argparser.add_argument('--init_compound_idx', type=int, required=True)
    argparser.add_argument('--condition', type=float, default=16.0)
    argparser.add_argument('--max_depth', type=int, default=5)
    args = argparser.parse_args()
    protein_name = args.protein_name
    init_compound_idx = args.init_compound_idx
    condition = [args.condition]
    max_depth = args.max_depth
    
    """Example of how this trainer can be run"""
    with open(Path(__file__).parent.parent / 'data' / protein_name / f'init_compound_{str(init_compound_idx)}.smi', 'r') as f:
        init_compound = f.read().rstrip()
    print(f'starting material: idx{init_compound_idx}_{init_compound}')
    # hyperparameters
    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hps = {
        'protein_name': protein_name,
        "log_dir": f"./logs/{protein_name}_reactions_{run_time}",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 100,
        "num_final_gen_steps": 10,
        "num_workers": 0,
        'checkpoint_every': 20,
        'data': {
            'init_compound': init_compound,
        },
        "opt": {
            "lr_decay": 50000,
            "learning_rate": 1e-5,
            'lr_GCN': 1e-5,
            'lr_Transformer': 1e-5,
        },
        "algo": {"sampling_tau": 0.0,
                 "offline_ratio": 0.0, 
                 "max_depth": max_depth,
                 "num_sample_iter": 5,
                 'num_final_sample_iter': 5,
                 'sampling_method': 'beam_search', # 'beam_search' (recommended) or 'prob_sampling'
                 'prob_sampling_size': 64,
                 'temperature': 1.0,
                 'beam_width': 10,
                 'beam_nbest': 10,
                 'tb': {'do_length_normalize': True,
                        'do_parameterize_p_b': True,
                        },
                 },
        'model': {
            'glb_batch_size': 32,
        },
        'replay': {
            'use': True,
            'capacity': 5000,
            'warmup': 0,
            'hindsight_ratio': 0.0,
        },
        "cond": {
            "temperature": {
                "sample_dist": "constant",
                "dist_params": condition,

            }
        },
    }

    
    # setup log directory
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    if hps["algo"]["sampling_method"] == "prob_sampling":
        algo_name = f'prob_{hps["algo"]["prob_sampling_size"]}'
    elif hps["algo"]["sampling_method"] == "beam_search":
        algo_name = f'beam_{hps["algo"]["beam_width"]}'
        
    wandb.init(project=f'{protein_name}_TRACER-GFN',
               name=f'DGX_{run_time}_mol{init_compound_idx}_condition{hps["cond"]["temperature"]["dist_params"][0]}',
               config=hps)
    
    profiler = profilers.SimpleProfiler()

    trial = QSAR_ReactionTrainer(hps, profiler=profiler)
    trial.print_every = 1
    time_start = time.time()
    
    trial.run()
    
    time_end = time.time()
    # print total time in h/m/s
    total_time = time_end - time_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f'total time: {hours}h {minutes}m {seconds}s')
    wandb.log({'total_time': total_time})

    profiler.describe()


if __name__ == "__main__":
    main()

