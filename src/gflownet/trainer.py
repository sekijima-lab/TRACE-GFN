import os
import pathlib
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple

import wandb
import numpy as np
import torch
import torch.nn as nn
# import torch.utils.tensorboard
import time
from omegaconf import OmegaConf
from rdkit import RDLogger
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from gflownet.data.replay_buffer import ReplayBuffer
from gflownet.data.sampling_iterator import SamplingIterator
from gflownet.utils.utils import make_batch_dl
from gflownet.utils.misc import create_logger
from gflownet.utils.multiprocessing_proxy import mp_object_wrapper

from .config import Config

# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType("FlatRewards", Tensor)  # type: ignore

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
RewardScalar = NewType("RewardScalar", Tensor)  # type: ignore


class GFNAlgorithm:
    def compute_batch_losses(
        self, model: nn.Module, batch: List, num_bootstrap: Optional[int] = 0
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Computes the loss for a batch of data, and proves logging informations

        Parameters
        ----------
        model: nn.Module
            The model being trained or evaluated
        batch: gd.Batch
            A batch of graphs
        num_bootstrap: Optional[int]
            The number of trajectories with reward targets in the batch (if applicable).

        Returns
        -------
        loss: Tensor
            The loss for that batch
        info: Dict[str, Tensor]
            Logged information about model predictions.
        """
        raise NotImplementedError()


class GFNTask:
    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        """Combines a minibatch of reward signal vectors and conditional information into a scalar reward.

        Parameters
        ----------
        cond_info: Dict[str, Tensor]
            A dictionary with various conditional informations (e.g. temperature)
        flat_reward: FlatRewards
            A 2d tensor where each row represents a series of flat rewards.

        Returns
        -------
        reward: RewardScalar
            A 1d tensor, a scalar log-reward for each minibatch entry.
        """
        raise NotImplementedError()

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        """Compute the flat rewards of mols according the the tasks' proxies

        Parameters
        ----------
        mols: List[RDMol]
            A list of RDKit molecules.
        Returns
        -------
        reward: FlatRewards
            A 2d tensor, a vector of scalar reward for valid each molecule.
        is_valid: Tensor
            A 1d tensor, a boolean indicating whether the molecule is valid.
        """
        raise NotImplementedError()


class GFNTrainer:
    def __init__(self, hps: Dict[str, Any], profiler=None):
        """A GFlowNet trainer. Contains the main training loop in `run` and should be subclassed.

        Parameters
        ----------
        hps: Dict[str, Any]
            A dictionary of hyperparameters. These override default values obtained by the `set_default_hps` method.
        device: torch.device
            The torch device of the main worker.
        """
        self.training_data: Dataset
        self.test_data: Dataset
        self.GCN_model: nn.Module
        self.Transformer_model: nn.Module
        self.MLP_model: nn.Module
        self.sampling_Transformer: nn.Module
        self.sampling_GCN: nn.Module
        self.replay_buffer: Optional[ReplayBuffer]
        self.mb_size: int        
        self.task: GFNTask
        self.algo: GFNAlgorithm
        self.profiler = profiler

        # There are three sources of config values
        #   - The default values specified in individual config classes
        #   - The default values specified in the `default_hps` method, typically what is defined by a task
        #   - The values passed in the constructor, typically what is called by the user
        # The final config is obtained by merging the three sources
        self.hps = hps
        self.cfg: Config = OmegaConf.structured(Config())
        self.set_default_hps(self.cfg)
        # OmegaConf returns a fancy object but we can still pretend it's a Config instance
        self.cfg = OmegaConf.merge(self.cfg, hps)  # type: ignore

        self.device = torch.device(self.cfg.device)
        # Print the loss every `self.print_every` iterations
        self.print_every = self.cfg.print_every
        # These hooks allow us to compute extra quantities when sampling data
        self.sampling_hooks: List[Callable] = []
        self.valid_sampling_hooks: List[Callable] = []
        # Will check if parameters are finite at every iteration (can be costly)
        self._validate_parameters = False

        self.root_smi = self.cfg.data.init_compound
        self.setup()

    def set_default_hps(self, base: Config):
        raise NotImplementedError()

    def setup_task(self):
        raise NotImplementedError()

    def setup_model(self):
        raise NotImplementedError()

    def setup_algo(self):
        raise NotImplementedError()

    def setup_data(self):
        pass

    def step(self, loss: Tensor):
        raise NotImplementedError()

    def setup(self):
        RDLogger.DisableLog("rdApp.*")
        self.rng = np.random.default_rng(142857)
        self.setup_data()
        self.setup_task()
        self.setup_model()
        self.setup_algo()


    def _wrap_for_mp(self, obj, send_to_device=False):
        """Wraps an object in a placeholder whose reference can be sent to a
        data worker process (only if the number of workers is non-zero)."""
        if send_to_device:
            obj.to(self.device)
        if self.cfg.num_workers > 0 and obj is not None:
            placeholder = mp_object_wrapper(
                obj,
                self.cfg.num_workers,
                pickle_messages=self.cfg.pickle_mp_messages,
            )
            return placeholder, torch.device("cpu")
            
        else:
            return obj, self.device

    def build_callbacks(self):
        return {}

    def build_training_data_loader(self) -> DataLoader:
        # model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True) # dev: deviceの略
        sampling_Transformer, dev = self._wrap_for_mp(self.sampling_Transformer, send_to_device=True)
        sampling_GCN, dev = self._wrap_for_mp(self.sampling_GCN, send_to_device=True)
        replay_buffer, _ = self._wrap_for_mp(self.replay_buffer, send_to_device=False)

        iterator = SamplingIterator(
            self.training_data,
            sampling_GCN,
            sampling_Transformer,
            self.algo,
            self.task,
            dev,
            batch_size=self.cfg.algo.global_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            replay_buffer=replay_buffer,
            ratio=self.cfg.algo.offline_ratio,
            num_sample_iter=self.cfg.algo.num_sample_iter,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "train"),
            random_action_prob=self.cfg.algo.train_random_action_prob,
            hindsight_ratio=self.cfg.replay.hindsight_ratio,
            src_transforms=self.src_transforms,
            vocab=self.vocab,
            cfg=self.cfg
        )
        
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else None,
        )

    def build_validation_data_loader(self) -> DataLoader:    
        sampling_Transformer, dev = self._wrap_for_mp(self.sampling_Transformer, send_to_device=True)
        sampling_GCN, dev = self._wrap_for_mp(self.sampling_GCN, send_to_device=True)
        iterator = SamplingIterator(
            self.test_data,
            sampling_GCN,
            sampling_Transformer,
            self.algo,
            self.task,
            dev,
            batch_size=self.cfg.algo.global_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            replay_buffer=None,
            ratio=self.cfg.algo.valid_offline_ratio,
            sample_cond_info=self.cfg.algo.valid_sample_cond_info,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "valid"),
            random_action_prob=self.cfg.algo.valid_random_action_prob,
            stream=False,
            hindsight_ratio=self.cfg.replay.hindsight_ratio,
            src_transforms=self.src_transforms,
            vocab=self.vocab,
            cfg=self.cfg
        )
        for hook in self.valid_sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else None,
        )

    def build_final_data_loader(self) -> DataLoader:
        sampling_Transformer, dev = self._wrap_for_mp(self.sampling_Transformer, send_to_device=True)
        sampling_GCN, dev = self._wrap_for_mp(self.sampling_GCN, send_to_device=True)
        iterator = SamplingIterator(
            self.training_data,
            sampling_GCN,
            sampling_Transformer,
            self.algo,
            self.task,
            dev,
            batch_size=self.cfg.algo.global_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            replay_buffer=None,
            ratio=0.0,
            num_sample_iter=self.cfg.algo.num_final_sample_iter,
            log_dir=os.path.join(self.cfg.log_dir, "final"),
            random_action_prob=0.0,
            hindsight_ratio=0.0,
            src_transforms=self.src_transforms,
            vocab=self.vocab,
            init_train_iter=self.cfg.num_training_steps,
            final=True,
            cfg=self.cfg
        )
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else None,
        )

    def train_batch(self, batch_dict, batch_properties, epoch_idx: int, batch_idx: int, train_it: int) -> Dict[str, Any]:
        batch_size = self.cfg.model.glb_batch_size
        while True:
            try:
                batch_dl = make_batch_dl(batch_dict, batch_size)
                # print('Start training...')
                for data in batch_dl:
                    trajs, cond_beta, cond_enc, log_rewards, flat_rewards, is_valid = data
                    batch = self.algo.construct_batch(self.root_smi, trajs, cond_beta, cond_enc, log_rewards)
                    batch.num_offline = batch_properties['num_offline']
                    batch.num_online = batch_properties['num_online']
                    batch.preferences = batch_properties['preferences']
                    batch.focus_dir = batch_properties['focus_dir']
                    batch.extra_info = batch_properties['extra_info']
                    loss, info = self.algo.compute_batch_losses(GCN_model=self.GCN_model, Transformer_model=self.Transformer_model,
                                                                MLP_model=self.MLP_model, src_transforms=self.src_transforms,
                                                                tgt_transforms=self.tgt_transforms, data_dict=self.data_dict,
                                                                vocab=self.vocab, dev=self.device, batch=batch, num_bootstrap=0,
                                                                )
                    if not torch.isfinite(loss):
                        raise ValueError("loss is not finite")

                    step_info = self.step(loss)
                    if self._validate_parameters and not all([torch.isfinite(i).all() for i in self.GCN_model.parameters()]):
                        raise ValueError("parameters are not finite")
                    if self._validate_parameters and not all([torch.isfinite(i).all() for i in self.Transformer_model.parameters()]):
                        raise ValueError("parameters are not finite")

                    if step_info is not None:
                        info.update(step_info)

                    if hasattr(batch, "extra_info"):
                        info.update(batch.extra_info)
                break
            except Exception as e:
                if "out of memory" in str(e):
                    print(e)
                    if batch_size < 8:
                        raise e
                    torch.cuda.empty_cache()
                    print(f'decrease batch size {batch_size} to {batch_size // 2}')
                    batch_size = batch_size // 2
                    continue
                else:
                    raise e
                    
        if self.cfg.algo.sampling_method == 'prob_sampling':
            print(f'{self.algo.synthesis_sampler.calc_unique_rate():.2f}% SMILES are unique in probabilistic sampling.')
                
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def evaluate_batch(self, batch: List, epoch_idx: int = 0, batch_idx: int = 0) -> Dict[str, Any]:
        loss, info = self.algo.compute_batch_losses(GCN_model=self.GCN_model, Transformer_model=self.Transformer_model,
                                                    MLP_model=self.MLP_model, src_transform=self.src_transform,
                                                    tgt_transform=self.tgt_transform, data_dict=self.data_dict,
                                                    vocab=self.vocab, batch=batch, num_bootstrap=0
                                                    )
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def run(self, logger=None):
        """Trains the GFN for `num_training_steps` minibatches, performing
        validation every `validate_every` minibatches.
        """
        if logger is None:
            logger = create_logger(logfile=self.cfg.log_dir + "/train.log")
        self.Transformer_model.to(self.device)
        self.GCN_model.to(self.device)
        self.sampling_Transformer.to(self.device)
        self.sampling_GCN.to(self.device)
        epoch_length = max(len(self.training_data), 1)
        valid_freq = self.cfg.validate_every
        ckpt_freq = self.cfg.checkpoint_every if self.cfg.checkpoint_every is not None else valid_freq
        self.algo.tgt_max_len = self.data_dict['tgt_max_len']

        while True:
            try:
                with self.profiler.profile("data_loading"):
                    train_dl = self.build_training_data_loader()
                    valid_dl = self.build_validation_data_loader()

                if self.cfg.num_final_gen_steps:
                    final_dl = self.build_final_data_loader()
                callbacks = self.build_callbacks()
                start = self.cfg.start_at_step + 1
                num_training_steps = self.cfg.num_training_steps
                logger.info("Start running")
                for it, (batch, batch_properties) in zip(range(start, 1 + num_training_steps), cycle(train_dl)): # it: iteration, batch: dictionary
                    time_start = time.time()

                    if self.cfg.reward == 'DOCK':
                        dock_succes_rate = (self.task.dock_total - self.task.dock_failed) / self.task.dock_total
                        wandb.log({'num_training_steps': it+1,
                                   'dock_succes_rate': dock_succes_rate})
                    else:
                        dock_succes_rate = 1

                    epoch_idx = it // epoch_length
                    batch_idx = it % epoch_length

                    if self.replay_buffer is not None and len(self.replay_buffer) < self.replay_buffer.warmup:
                        logger.info(
                            f"iteration {it} : warming up replay buffer {len(self.replay_buffer)}/{self.replay_buffer.warmup}"
                        )
                        continue
                    with self.profiler.profile("training_it"):
                        info = self.train_batch(batch, batch_properties, epoch_idx, batch_idx, it)

                    if it % self.print_every == 0:
                        logger.info(f"iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))

                    if valid_freq > 0 and it % valid_freq == 0:
                        with self.profiler.profile("validation"):
                            for batch in valid_dl:
                                info = self.evaluate_batch(batch.to(self.device), epoch_idx, batch_idx)
                                self.log(info, it, "valid")
                                logger.info(
                                    f"validation - iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items())
                                )
                            end_metrics = {}
                            for c in callbacks.values():
                                if hasattr(c, "on_validation_end"):
                                    c.on_validation_end(end_metrics)
                            self.log(end_metrics, it, "valid_end")
                    time_end = time.time()
                    print(f'training time: {time_end - time_start}')
                    if ckpt_freq > 0 and it % ckpt_freq == 0:
                        self._save_state(it)
                self._save_state(num_training_steps)
                break
            except Exception as e:
                if "out of memory" in str(e):
                    print(e)
                    batch_size = batch_size // 2
                    torch.cuda.empty_cache()
                    if batch_size < 8:
                        raise e
                    torch.cuda.empty_cache()
                else:
                    raise e
            
        num_final_gen_steps = self.cfg.num_final_gen_steps
        if num_final_gen_steps:
            logger.info(f"Generating final {num_final_gen_steps} batches ...")
            for final_it, (batch, batch_properties) in zip(range(num_final_gen_steps), cycle(final_dl)):
                print(f'final generation: iteration {final_it+1} done')
                pass
            logger.info("Final generation steps completed.")

    def _save_state(self, it):
        torch.save(
            {
                "models_state_dict": [self.GCN_model.state_dict()],
                "cfg": self.cfg,
                "step": it,
            },
            open(pathlib.Path(self.cfg.log_dir) / f"model_state_GCN_{it}.pt", "wb"),
        )
        torch.save(
            {
                "models_state_dict": [self.Transformer_model.state_dict()],
                "cfg": self.cfg,
                "step": it,
            },
            open(pathlib.Path(self.cfg.log_dir) / f"model_state_Transformer_{it}.pt", "wb"),
        )
        torch.save(
            {
                "models_state_dict": [self.MLP_model.state_dict()],
                "cfg": self.cfg,
                "step": it,
            },
            open(pathlib.Path(self.cfg.log_dir) / f"model_state_MLP_{it}.pt", "wb"),
        )

    def log(self, info, index, key):
        if not hasattr(self, "_summary_writer"):
            self._summary_writer = torch.utils.tensorboard.SummaryWriter(self.cfg.log_dir)
        for k, v in info.items():
            self._summary_writer.add_scalar(f"{key}_{k}", v, index)


def cycle(it):
    while True:
        for i in it:
            yield i
