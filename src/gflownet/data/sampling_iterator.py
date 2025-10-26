import os
import sqlite3
from collections.abc import Iterable
from copy import deepcopy
from typing import Callable, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import RDLogger
from torch.utils.data import Dataset, IterableDataset
from rdkit import Chem
from gflownet.data.replay_buffer import ReplayBuffer
from gflownet.config import Config
import wandb


class SamplingIterator(IterableDataset):
    """This class allows us to parallelise and train faster.

    By separating sampling data/the model and building torch geometric
    graphs from training the model, we can do the former in different
    processes, which is much faster since much of graph construction
    is CPU-bound.

    """

    def __init__(
        self,
        dataset: Dataset,
        GCN_model: nn.Module,
        Transformer_model: nn.Module,
        algo,
        task,
        device,
        batch_size: int = 1,
        illegal_action_logreward: float = -50,
        ratio: float = 0.5,
        num_sample_iter: int = 100,
        stream: bool = True,
        replay_buffer: ReplayBuffer = None,
        log_dir: str = None,
        sample_cond_info: bool = True,
        random_action_prob: float = 0.0,
        hindsight_ratio: float = 0.0,
        init_train_iter: int = 0,
        final: bool = False,
        src_transforms = None,
        vocab = None,
        cfg = None
    ):
        """Parameters
        ----------
        dataset: Dataset
            A dataset instance
        model: nn.Module
            The model we sample from (must be on CUDA already or share_memory() must be called so that
            parameters are synchronized between each worker)
        ctx:
            The context for the environment, e.g. a MolBuildingEnvContext instance
        algo:
            The training algorithm, e.g. a TrajectoryBalance instance
        task: GFNTask
            A Task instance, e.g. a MakeRingsTask instance
        device: torch.device
            The device the model is on
        replay_buffer: ReplayBuffer
            The replay buffer for training on past data
        batch_size: int
            The number of trajectories, each trajectory will be comprised of many graphs, so this is
            _not_ the batch size in terms of the number of graphs (that will depend on the task)
        illegal_action_logreward: float
            The logreward for invalid trajectories
        ratio: float
            The ratio of offline trajectories in the batch.
        stream: bool
            If True, data is sampled iid for every batch. Otherwise, this is a normal in-order
            dataset iterator.
        log_dir: str
            If not None, logs each SamplingIterator worker's generated molecules to that file.
        sample_cond_info: bool
            If True (default), then the dataset is a dataset of points used in offline training.
            If False, then the dataset is a dataset of preferences (e.g. used to validate the model)
        random_action_prob: float
            The probability of taking a random action, passed to the graph sampler
        init_train_iter: int
            The initial training iteration, incremented and passed to task.sample_conditional_information
        final: bool
            Whether these are the final generation steps. Useful because beta is set to 64.0 for these
        """
        self.data = dataset
        self.GCN_model = GCN_model
        self.Transformer_model = Transformer_model
        # self.MLP_model = MLP_model
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.illegal_action_logreward = illegal_action_logreward
        self.offline_batch_size = int(np.ceil(self.batch_size * ratio))
        self.online_batch_size = int(np.floor(self.batch_size * (1 - ratio)))
        self.ratio = ratio
        self.num_sample_iter = num_sample_iter
        # self.ctx = ctx
        self.algo = algo
        self.task = task
        self.device = device
        self.stream = stream
        self.sample_online_once = True
        self.sample_cond_info = sample_cond_info
        self.random_action_prob = random_action_prob
        self.hindsight_ratio = hindsight_ratio
        self.train_it = init_train_iter
        self.do_validate_batch = False  # Turn this on for debugging
        self.final = final
        self.cfg = cfg
        self.datacfg = self.cfg.data
        self.root_smi = self.datacfg.init_compound
        self.src_transforms = src_transforms
        self.vocab = vocab
        self.train_iter = 0
        self.buffer = None
        self.it = 0

        # Slightly weird semantics, but if we're sampling x given some fixed cond info (data)
        # then "offline" now refers to cond info and online to x, so no duplication and we don't end
        # up with 2*batch_size accidentally
        if not sample_cond_info:
            self.offline_batch_size = self.online_batch_size = self.batch_size

        # This SamplingIterator instance will be copied by torch DataLoaders for each worker, so we
        # don't want to initialize per-worker things just yet, such as where the log the worker writes
        # to. This must be done in __iter__, which is called by the DataLoader once this instance
        # has been copied into a new python process.
        self.log_dir = log_dir
        self.log_hooks: List[Callable] = []
    
    def add_log_hook(self, hook: Callable):
        self.log_hooks.append(hook)

    def _idx_iterator(self):
        RDLogger.DisableLog("rdApp.*")
        if self.stream:
            # If we're streaming data, just sample `offline_batch_size` indices
            while True:
                yield self.rng.integers(0, len(self.data), self.offline_batch_size)
        else:
            # Otherwise, figure out which indices correspond to this worker
            worker_info = torch.utils.data.get_worker_info()
            n = len(self.data)
            if n == 0:
                yield np.arange(0, 0)
                return
            assert (
                self.offline_batch_size > 0
            ), "offline_batch_size must be > 0 if not streaming and len(data) > 0 (have you set ratio=0?)"
            if worker_info is None:  # no multi-processing
                start, end, wid = 0, n, -1
            else:  # split the data into chunks (per-worker)
                nw = worker_info.num_workers
                wid = worker_info.id
                start, end = int(np.round(n / nw * wid)), int(np.round(n / nw * (wid + 1)))
            bs = self.offline_batch_size
            if end - start <= bs:
                yield np.arange(start, end)
                return
            for i in range(start, end - bs, bs):
                yield np.arange(i, i + bs)
            if i + bs < end:
                yield np.arange(i + bs, end)

    def __len__(self):
        if self.stream:
            return int(1e6)
        if len(self.data) == 0 and self.sample_online_once:
            return 1
        return len(self.data)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        # Now that we know we are in a worker instance, we can initialize per-worker things
        self.rng = self.algo.rng = self.task.rng = np.random.default_rng(142857 + self._wid)
        # self.ctx.device = self.device
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = f"{self.log_dir}/generated_mols_{self._wid}.db"
            # self.log.connect(self.log_path)

        for idcs in self._idx_iterator():
            self.train_iter += 1
            num_offline = idcs.shape[0]  # This is in [0, self.offline_batch_size]
            # Sample conditional info such as temperature, trade-off weights, etc.

            if self.sample_cond_info:
                num_online = self.online_batch_size
                cond_info = self.task.sample_conditional_information(
                    num_offline + self.online_batch_size, self.train_it, final=self.final
                )
                print(f'conditional information, beta: {cond_info["beta"][0]}')
                pass

            else:  # If we're not sampling the conditionals, then the idcs refer to listed preferences
                num_online = num_offline
                num_offline = 0
                cond_info = self.task.encode_conditional_information(
                    steer_info=torch.stack([self.data[i] for i in idcs])
                )
                trajs, flat_rewards = [], []

            # # Sample some on-policy data
            if num_online > 0:
                with torch.no_grad():
                    src_transforms = self.src_transforms
                    vocab = self.vocab
                    num_online = 0
                    while num_online == 0:
                        data, FlatRewards, validity, total = self.algo.create_training_data_from_own_samples(
                            GCN_model = self.GCN_model,
                            Transformer_model = self.Transformer_model,
                            n = self.num_sample_iter,
                            dev = self.device,
                            transforms = src_transforms,
                            vocab = vocab,
                            task = self.task
                        )
                        num_online = len(data)
                        # self.buffer = (data, FlatRewards, validity, total)
                        print(f'{num_online} trajectories are generated.')
                        
                        if num_online == 0:
                            if len(self.replay_buffer) > 0:
                                num_replay = min(100, len(self.replay_buffer))
                                trajs, log_rewards, flat_rewards, cond_info, is_valid = self.replay_buffer.sample(
                                num_replay
                                )
                                print(f'{len(replay_trajs)} samples are sampled from replay buffer.')
                                cond_info = {k: torch.stack([d[k] for d in cond_info]) for k in cond_info[0]}
                                batch_properties = {'num_offline': num_offline, 'num_online': num_online, 'preferences': cond_info.get("preferences", None),
                                                    'focus_dir': cond_info.get("focus_dir", None), 'extra_info': extra_info}
                                batch = {'trajs': trajs, 'cond_info': cond_info, 'log_rewards': log_rewards, 'flat_rewards': flat_rewards, 'is_valid': is_valid}
                                self.train_it += worker_info.num_workers if worker_info is not None else 1
                                yield batch, batch_properties                      
                    sampling_num = num_offline + num_online
                    cond_info = self.task.sample_conditional_information(sampling_num,
                                                                         self.train_it,
                                                                         final=self.final
                                                                         )
                    is_valid = torch.ones(num_offline + num_online).bool()
                                        
                    # Sample some dataset data
                    graphs, flat_rewards = map(list, zip(*[self.data[i] for i in idcs])) if len(idcs) else ([], [])
                    flat_rewards = (
                        list(self.task.flat_reward_transform(torch.stack(flat_rewards))) if len(flat_rewards) else []
                    )
                    trajs = data
                    # trajs:List[str], [SMILES.template.SMILES.template.SMILES.template..., SMILES.template.SMILES.template.SMILES.template..., ...]のリスト
                    
                if self.algo.bootstrap_own_reward:
                    # The model can be trained to predict its own reward,
                    # i.e. predict the output of cond_info_to_logreward
                    pred_reward = [i["reward_pred"].cpu().item() for i in trajs[num_offline:]]
                    flat_rewards += pred_reward
                else:
                    # Otherwise, query the task for flat rewards
                    valid_idcs = torch.arange(num_online)
                    # fetch the valid trajectories endpoints
                    mols = [Chem.MolFromSmiles(trajs[i].split('.')[-1]) for i in valid_idcs] # それぞれのtrajで末端のSMILESをmolに変換
                    m_is_valid = torch.tensor([m is not None for m in mols]).bool()
                    online_flat_rew = self.task.list_to_flat_rewards(FlatRewards)
                    assert (
                        online_flat_rew.ndim == 2
                    ), "FlatRewards should be (mbsize, n_objectives), even if n_objectives is 1"
                    # The task may decide some of the mols are invalid, we have to again filter those
                    valid_idcs = valid_idcs[m_is_valid]
                    pred_reward = torch.zeros((num_online, online_flat_rew.shape[1]))
                    pred_reward[valid_idcs - num_offline] = online_flat_rew.float()
                    is_valid[num_offline:] = False
                    is_valid[valid_idcs] = True # is_valid: torch.Size([64]) valid_idcs: torch.Size([101])
                    flat_rewards += list(pred_reward)


            # Compute scalar rewards from conditional information & flat rewards
            flat_rewards = torch.stack(flat_rewards) 
            # -inf < logR(x) < 0  --> 0 < log(1+R(x))
            mod_rewards = flat_rewards # + torch.ones_like(flat_rewards)
            log_rewards = self.task.cond_info_to_logreward(cond_info, mod_rewards)

            # Computes some metrics
            extra_info = {}
            if not self.sample_cond_info:
                # If we're using a dataset of preferences, the user may want to know the id of the preference
                for i, j in zip(trajs, idcs):
                    i["data_idx"] = j
            #  note: we convert back into natural rewards for logging purposes
            #  (allows to take averages and plot in objective space)
            #  TODO: implement that per-task (in case they don't apply the same beta and log transformations)
            rewards = torch.exp(log_rewards / cond_info["beta"])
            if num_online > 0 and self.log_dir is not None:
                self.save_data(trajs=trajs, flat_rewards=flat_rewards, cond_info=cond_info, rewards=rewards, train_iter=self.train_iter)
                
            if num_online > 0:
                for hook in self.log_hooks:
                    extra_info.update(
                        hook(
                            deepcopy(trajs[num_offline:]),
                            deepcopy(rewards[num_offline:]),
                            deepcopy(flat_rewards[num_offline:]),
                            {k: v[num_offline:] for k, v in deepcopy(cond_info).items()},
                        )
                    )           
            self.it += 1
            wandb.log({'num_training_steps': self.it,
                        "validity": validity,
                        "mean_reward": flat_rewards.mean().item(),
                        'max_reward': flat_rewards.max().item(),
                        'num_online': num_online})            

            if self.replay_buffer is not None:
                # If we have a replay buffer, we push the online trajectories in it
                # and resample immediately such that the "online" data in the batch
                # comes from a more stable distribution (try to avoid forgetting)
                # cond_info: {'beta': tensor([64.0, 64.0, ...]), 'encoding': tensor([[1, 1, 1,], [...], ...])}
                # cond_info is a dict, so we need to convert it to a list of dicts
                cond_info = [{k: v[i] for k, v in cond_info.items()} for i in range(num_offline + num_online)]

                # push the online trajectories in the replay buffer and sample a new 'online' batch
                for i in range(num_offline, len(trajs)):
                    # if flat_rewards[i] > 0.5:
                    self.replay_buffer.push(
                        deepcopy(trajs[i]),
                        deepcopy(log_rewards[i]),
                        deepcopy(flat_rewards[i]),
                        deepcopy(cond_info[i]),
                        deepcopy(is_valid[i]),
                    )
                if len(self.replay_buffer) > 0:
                    if num_online < 10:
                        num_replay = min(100, len(self.replay_buffer))
                    else:
                        num_replay = num_online
                    replay_trajs, replay_logr, replay_fr, replay_condinfo, replay_valid = self.replay_buffer.sample(
                    num_replay
                    )
                    print(f'{len(replay_trajs)} samples are sampled from replay buffer.')
                    trajs += replay_trajs # list of str
                    log_rewards = torch.cat([log_rewards, replay_logr], dim=0)
                    flat_rewards = torch.cat([flat_rewards, replay_fr], dim=0)
                    is_valid = torch.cat([is_valid, replay_valid], dim=0) # torch.tensor
                    cond_info += replay_condinfo            
                cond_info = {k: torch.stack([d[k] for d in cond_info]) for k in cond_info[0]}

            batch_properties = {'num_offline': num_offline, 'num_online': num_online, 'preferences': cond_info.get("preferences", None),
                                'focus_dir': cond_info.get("focus_dir", None), 'extra_info': extra_info}
            batch = {'trajs': trajs, 'cond_info': cond_info, 'log_rewards': log_rewards, 'flat_rewards': flat_rewards, 'is_valid': is_valid}
            self.train_it += worker_info.num_workers if worker_info is not None else 1
            yield batch, batch_properties

    def log_generated(self, trajs, rewards, flat_rewards, cond_info):
        mols = [""] * len(trajs)
        flat_rewards = flat_rewards.reshape((len(flat_rewards), -1)).data.numpy().tolist()
        rewards = rewards.data.numpy().tolist()
        logged_keys = [k for k in sorted(cond_info.keys()) if k not in ["encoding", "preferences", "focus_dir"]]

        data = [
            [mols[i], rewards[i]]
            + flat_rewards[i]
            + [cond_info[k][i].item() for k in logged_keys]
            for i in range(len(trajs))
        ]

        data_labels = (
            ["smi", "r", "traj"]
            + [f"fr_{i}" for i in range(len(flat_rewards[0]))]
            + [f"ci_{k}" for k in logged_keys]
        )

        self.log.insert_many(data, data_labels)
    
    def save_data(self, trajs, flat_rewards, cond_info, rewards, train_iter):
        columns = ['Trajectory', 'SMILES', 'FlatReward', 'CondInfo', 'Reward', 'TrainIter']
        df = pd.DataFrame(columns=columns)
        flat_rewards = flat_rewards.squeeze().data.numpy().tolist()
        cond_beta = cond_info['beta'].squeeze().data.numpy().tolist()
        rewards = rewards.squeeze().data.numpy().tolist()
        if len(trajs) == 1:
            flat_rewards = [flat_rewards]
            cond_beta = [cond_beta]
            rewards = [rewards]
        for i, traj in enumerate(trajs):
            result = traj.split('.')[-1]
            row = {'Trajectory': traj, 'SMILES': result, 'FlatReward': flat_rewards[i], 'CondInfo': cond_beta[i], 'Reward': rewards[i], 'TrainIter': train_iter}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(f'{self.log_dir}/results_round{self.train_iter}_wid{self._wid}.csv', index=False)


class SQLiteLog:
    def __init__(self, timeout=300):
        """Creates a log instance, but does not connect it to any db."""
        self.is_connected = False
        self.db = None
        self.timeout = timeout

    def connect(self, db_path: str):
        """Connects to db_path

        Parameters
        ----------
        db_path: str
            The sqlite3 database path. If it does not exist, it will be created.
        """
        self.db = sqlite3.connect(db_path, timeout=self.timeout)
        cur = self.db.cursor()
        self._has_results_table = len(
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'").fetchall()
        )
        cur.close()

    def _make_results_table(self, types, names):
        type_map = {str: "text", float: "real", int: "real"}
        col_str = ", ".join(f"{name} {type_map[t]}" for t, name in zip(types, names))
        cur = self.db.cursor()
        cur.execute(f"create table results ({col_str})")
        self._has_results_table = True
        cur.close()

    def insert_many(self, rows, column_names):
        assert all(
            [isinstance(x, str) or not isinstance(x, Iterable) for x in rows[0]]
        ), "rows must only contain scalars"
        if not self._has_results_table:
            self._make_results_table([type(i) for i in rows[0]], column_names)
        cur = self.db.cursor()
        cur.executemany(f'insert into results values ({",".join("?"*len(rows[0]))})', rows)  # nosec
        cur.close()
        self.db.commit()
