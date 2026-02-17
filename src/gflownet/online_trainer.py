import copy
import os
import pathlib
import pickle
import sys
import types

import git
import torch
from omegaconf import OmegaConf
from torch import Tensor

from gflownet.algo.trajectory_balance_synthesis import TrajectoryBalance
from gflownet.data.replay_buffer import ReplayBuffer

from .trainer import GFNTrainer

from gflownet.models.Transformer.model import Transformer
from gflownet.models.GCN import network
from gflownet.models.mlp import MLP

from gflownet.models.Transformer.preprocess import make_counter, make_transforms

# Collect all config classes from the current codebase for checkpoint compatibility
import gflownet.config as _cfg_main
import gflownet.models.config as _cfg_models
import gflownet.data.config as _cfg_data
import gflownet.algo.config as _cfg_algo
import gflownet.tasks.config as _cfg_tasks
import gflownet.utils.config as _cfg_utils

_CONFIG_MODULES = [_cfg_main, _cfg_models, _cfg_data, _cfg_algo, _cfg_tasks, _cfg_utils]


class _CompatPickle:
    """Pickle module wrapper that handles checkpoints saved with old module paths."""

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'config.config':
                for src in _CONFIG_MODULES:
                    if hasattr(src, name):
                        return getattr(src, name)
                # Return a dummy class for anything not found
                return type(name, (), {})
            return super().find_class(module, name)

    # Forward everything else to standard pickle
    UnpicklingError = pickle.UnpicklingError


class StandardOnlineTrainer(GFNTrainer):
    def setup_model(self):
        self.model_cfg = self.cfg.model
        dim_GCN = self.model_cfg.dim_GCN
        n_conv_hidden = self.model_cfg.n_conv_hidden
        n_mlp_hidden = self.model_cfg.n_mlp_hidden
        dropout_GCN = self.model_cfg.dropout_GCN
        ckpt_GCN = self.model_cfg.ckpt_GCN
        
        self.GCN_model = network.MolecularGCN(dim = dim_GCN,
                                n_conv_hidden = n_conv_hidden,
                                n_mlp_hidden = n_mlp_hidden,
                                dropout = dropout_GCN
                                ).to(self.device)
        if self.model_cfg.gfn_ckpt_GCN is not None:
            path = self.model_cfg.gfn_ckpt_GCN
            ckpt_load_GCN = torch.load(path, map_location=self.device)
            self.GCN_model.load_state_dict(ckpt_load_GCN['models_state_dict'][0])
            print("GCN ckpt is specified: ", self.model_cfg.gfn_ckpt_GCN)
        else:
            ckpt_load_GCN = torch.load(ckpt_GCN, map_location=self.device)
            self.GCN_model.load_state_dict(ckpt_load_GCN)
        
        self.src_transforms, self.tgt_transforms, self.vocab = self._preprocess()
        self.cfg.algo.tgt_max_len = self.data_dict['tgt_max_len']
        d_Transformer = self.model_cfg.d_Transformer
        num_encoder_layers = self.model_cfg.num_encoder_layers
        num_decoder_layers = self.model_cfg.num_decoder_layers
        nhead = self.model_cfg.nhead
        dropout_Transformer = self.model_cfg.dropout_Transformer
        dim_ff = self.model_cfg.dim_ff
        ckpt_Transformer = self.model_cfg.ckpt_Transformer
        self.Transformer_model = Transformer(d_model=d_Transformer, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                             num_decoder_layers=num_decoder_layers, dim_feedforward=dim_ff,
                                             vocab=self.vocab, dropout=dropout_Transformer, device=self.device).to(self.device)
        if self.model_cfg.gfn_ckpt_Transformer is not None:
            ckpt_load_Transformer = torch.load(self.model_cfg.gfn_ckpt_Transformer, map_location=self.device)
            self.Transformer_model.load_state_dict(ckpt_load_Transformer['models_state_dict'][0])
            print("Transformer ckpt is specified: ", self.model_cfg.gfn_ckpt_Transformer)
        else:
            ckpt_load_Transformer = torch.load(ckpt_Transformer, map_location=self.device, pickle_module=_CompatPickle)
            self.Transformer_model.load_state_dict(ckpt_load_Transformer['model_state_dict'])
        
        self.MLP_model = MLP(n_in = self.cfg.cond.temperature.num_thermometer_dim,
                        n_hid = self.cfg.model.num_mlp_hidden,
                        n_out = 1,
                        n_layer = 2
                        )
        if self.model_cfg.gfn_ckpt_MLP is not None:
            ckpt_load_MLP = torch.load(self.model_cfg.gfn_ckpt_MLP, map_location=self.device)
            self.MLP_model.load_state_dict(ckpt_load_MLP['models_state_dict'][0])
            print("MLP ckpt is specified: ", self.model_cfg.gfn_ckpt_MLP)
        else:
            print("No MLP checkpoint specified, using default initialization.")
 
    def _preprocess(self):
        src_train_path = f'{self.cfg.data.USPTO_path}/src_train.txt'
        tgt_train_path = f'{self.cfg.data.USPTO_path}/tgt_train.txt'
        src_valid_path = f'{self.cfg.data.USPTO_path}/src_valid.txt'
        tgt_valid_path = f'{self.cfg.data.USPTO_path}/tgt_valid.txt'
        data_dict = make_counter(src_train_path=src_train_path,
                                tgt_train_path=tgt_train_path,
                                src_valid_path=src_valid_path,
                                tgt_valid_path=tgt_valid_path
                                )
        self.data_dict = data_dict
        return make_transforms(data_dict=data_dict, make_vocab=True)
        
    def setup_algo(self):
        algo = self.cfg.algo.method
        if algo == "TB":
            algo = TrajectoryBalance
        else:
            raise ValueError(algo)
        self.algo = algo(self.cfg)

    def setup_data(self):
        self.training_data = []
        self.test_data = []

    def setup(self):
        super().setup() 
        self.offline_ratio = 0.5 # replay buffer で使うデータの割合？
        self.replay_buffer = ReplayBuffer(self.cfg, self.rng) if self.cfg.replay.use else None
        Z_params = list(self.MLP_model.parameters())
            
        self.opt_GCN = torch.optim.Adam(
            self.GCN_model.parameters(),
            self.cfg.opt.lr_GCN,
            (self.cfg.opt.momentum, 0.999),
            weight_decay=self.cfg.opt.weight_decay,
            eps=self.cfg.opt.adam_eps,
        )
        self.opt_Transformer = torch.optim.Adam(
            self.Transformer_model.parameters(),
            self.cfg.opt.lr_Transformer,
            (self.cfg.opt.momentum, 0.999),
            weight_decay=self.cfg.opt.weight_decay,
            eps=self.cfg.opt.adam_eps,
        )
        self.opt_Z = torch.optim.Adam(Z_params, 
                                      self.cfg.algo.tb.Z_learning_rate, 
                                      (0.9, 0.999)) # adam(params, lr, (momentum, alpha))
        self.sampling_tau = self.cfg.algo.sampling_tau
        if self.sampling_tau > 0:
            self.sampling_Transformer = copy.deepcopy(self.Transformer_model)
            self.sampling_GCN = copy.deepcopy(self.GCN_model)
        else:
            self.sampling_Transformer = self.Transformer_model
            self.sampling_GCN = self.GCN_model

        self.mb_size = self.cfg.algo.global_batch_size # mb: mini-batch
        self.clip_grad_callback = {
            "value": lambda params: torch.nn.utils.clip_grad_value_(params, self.cfg.opt.clip_grad_param),
            "norm": lambda params: torch.nn.utils.clip_grad_norm_(params, self.cfg.opt.clip_grad_param),
            "none": lambda x: None, 
        }[self.cfg.opt.clip_grad_type]

        # saving hyperparameters
        git_hash = git.Repo(__file__, search_parent_directories=True).head.object.hexsha[:7]
        self.cfg.git_hash = git_hash

        os.makedirs(self.cfg.log_dir, exist_ok=True)
        print("\n\nHyperparameters:\n")
        yaml = OmegaConf.to_yaml(self.cfg)
        print(yaml)
        with open(pathlib.Path(self.cfg.log_dir) / "hps.yaml", "w") as f:
            f.write(yaml)

    def step(self, loss: Tensor):
        loss.backward()
        self.opt_GCN.step()
        self.opt_GCN.zero_grad()
        self.opt_Transformer.step()
        self.opt_Transformer.zero_grad()
        self.opt_Z.step()
        self.opt_Z.zero_grad()
        # self.lr_sched.step()
        # self.lr_sched_Z.step()
        if self.sampling_tau > 0:
            for a, b in zip(self.GCN_model.parameters(), self.sampling_GCN.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))
            for a, b in zip(self.Transformer_model.parameters(), self.sampling_Transformer.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))
