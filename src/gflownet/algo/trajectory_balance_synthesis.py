import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor
from torch_scatter import scatter, scatter_sum

from gflownet.config import Config
from gflownet.data.sampling_iterator import SQLiteLog
# from gflownet.envs.synthesis_building_env import (
#     ActionCategorical,
#     ReactionTemplateEnv,
#     ReactionTemplateEnvContext,
#     generate_forward_trajectory,
# )
from gflownet.models.GCN.utils import predict_GCN
from gflownet.models.Transformer.utils import predict_Transformer
from gflownet.trainer import GFNAlgorithm
from gflownet.algo.reaction_sampling import SynthesisSampler
import itertools
import wandb
from gflownet.utils.utils import calc_dup_dict, calc_p_B



def shift_right(x: torch.Tensor, z=0):
    "Shift x right by 1, and put z in the first position"
    x = torch.roll(x, 1, dims=0)
    x[0] = z
    return x


def cross(x: torch.Tensor):
    """
    Calculate $y_{ij} = \sum_{t=i}^j x_t$.
    The lower triangular portion is the inverse of the upper triangular one.
    """
    assert x.ndim == 1
    y = torch.cumsum(x, 0)
    return y[None] - shift_right(y)[:, None]


def subTB(v: torch.tensor, x: torch.Tensor):
    r"""
    Compute the SubTB(1):
    $\forall i \leq j: D[i,j] =
        \log \frac{F(s_i) \prod_{k=i}^{j} P_F(s_{k+1}|s_k)}
        {F(s_{j + 1}) \prod_{k=i}^{j} P_B(s_k|s_{k+1})}$
      for a single trajectory.
    Note that x_k should be P_F(s_{k+1}|s_k) - P_B(s_k|s_{k+1}).
    """
    assert v.ndim == x.ndim == 1
    # D[i,j] = V[i] - V[j + 1]
    D = v[:-1, None] - v[None, 1:]
    # cross(x)[i, j] = sum(x[i:j+1])
    D = D + cross(x)
    return torch.triu(D)

class Batch:
    '''
    self.raw_traj_lens: Tensor that contains the length of each trajectory
    self.root_mol: List that contains the root compound of each trajectory
    self.smis: List that contains the smiles of each trajectory, root_mol has been removed
    self.templates: List that contains the template of each trajectory
    self.log_rewards: List that contains the log_reward of each trajectory
    '''
    def __init__(self, root_smi, trajs, cond_beta, cond_enc, log_rewards):
        self.root_smi = root_smi
        self.smis = []
        self.templates = []
        for traj in trajs:
            smi_list = []
            template_list = []
            traj_list = traj.split('.') 
            for i in range(len(traj_list)):
                try:
                    value = traj_list[i]
                    if i % 2 == 0:
                        smi_list.append(value)
                    else:
                        template_list.append(int(value))
                except:
                    print('Trajectory parsing error')
                    pass
            self.smis.append(smi_list)
            self.templates.append(template_list)
        self.smis_GCN = [smis[:-1] for smis in self.smis]
        self.smis_Transformer_src = [smis[:-1] for smis in self.smis]
        self.smis_Transformer_tgt = [smis[1:] for smis in self.smis]
        self.traj_lens = torch.tensor([len(i.split('.')) for i in trajs])
        self.cond_beta = cond_beta
        self.cond_enc = cond_enc
        self.log_rewards = log_rewards


class TrajectoryBalanceModel(nn.Module):
    def forward(self, batch: gd.Batch):
        raise NotImplementedError()

    def logZ(self, cond_info: Tensor) -> Tensor:
        raise NotImplementedError()


# TB loss
class TrajectoryBalance(GFNAlgorithm):
    """Trajectory-based GFN loss implementations. Implements
    - TB: Trajectory Balance: Improved Credit Assignment in GFlowNets Nikolay Malkin, Moksh Jain,
    Emmanuel Bengio, Chen Sun, Yoshua Bengio
    https://arxiv.org/abs/2201.13259

    - SubTB(1): Learning GFlowNets from partial episodes for improved convergence and stability, Kanika Madan, Jarrid
    Rector-Brooks, Maksym Korablyov, Emmanuel Bengio, Moksh Jain, Andrei Cristian Nica, Tom Bosc, Yoshua Bengio,
    Nikolay Malkin
    https://arxiv.org/abs/2209.12782
    Note: We implement the lambda=1 version of SubTB here (this choice is based on empirical results from the paper)

    - DB: GFlowNet Foundations, Yoshua Bengio, Salem Lahlou, Tristan Deleu, Edward J. Hu, Mo Tiwari, Emmanuel Bengio
    https://arxiv.org/abs/2111.09266
    Note: This is the trajectory version of Detailed Balance (i.e. transitions are not iid, but trajectories are).
    Empirical results in subsequent papers suggest that DB may be improved by training on iid transitions (sampled from
    a replay buffer) instead of trajectories.
    """

    def __init__(
        self,
        cfg: Config,
    ):
        """Instanciate a TB algorithm.

        Parameters
        ----------
        env: ReactionTemplateEnv
            A synthesis environment.
        ctx: ReactionTemplateEnvContext
            A context.
        cfg: Config
            Hyperparameters
        """
        # self.ctx = ctx
        # self.env = env
        self.global_cfg = cfg
        self.cfg = cfg.algo.tb
        self.max_depth = cfg.algo.max_depth
        self.max_nodes = cfg.algo.max_nodes
        self.length_normalize_losses = cfg.algo.tb.do_length_normalize
        
        # Experimental flags
        self.reward_loss_is_mae = True
        self.tb_loss_is_mae = False
        self.tb_loss_is_huber = False
        self.mask_invalid_rewards = False
        self.reward_normalize_losses = False
        self.sample_temp = 1
        self.bootstrap_own_reward = self.cfg.bootstrap_own_reward
        
        # When the model is autoregressive, we can avoid giving it ["A", "AB", "ABC", ...] as a sequence of inputs, and
        # instead give "ABC...Z" as a single input, but grab the logits at every timestep. Only works if using something
        # like a transformer with causal self-attention.
        self.model_is_autoregressive = False
        self.tgt_max_len = self.global_cfg.algo.tgt_max_len
        self.synthesis_sampler = SynthesisSampler(
            max_depth=cfg.algo.max_depth,
            pad_with_terminal_state=self.cfg.do_parameterize_p_b,
            tgt_max_len=self.tgt_max_len,
            cfg=cfg
        )
        self.dup_dict = None
        self.log_dir = cfg.log_dir
        self.log = SQLiteLog()


    def create_training_data_from_own_samples(
        self,
        GCN_model: nn.Module,
        Transformer_model: nn.Module,
        n: int,
        dev: torch.device,
        transforms: nn.Module,
        vocab,
        task
        ):
        """Generate trajectories by sampling a model

        Parameters
        ----------
        model: TrajectoryBalanceModel
           The model being sampled
        n: int
            Number of trajectories to sample
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        random_action_prob: float
            Probability of taking a random action
        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, int]]
           - reward_pred: float, -100 if an illegal action is taken, predicted R(x) if bootstrapping, None otherwise
           - fwd_logprob: log Z + sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - logZ: predicted log Z
           - loss: predicted loss (if bootstrapping)
           - is_valid: is the generated graph valid according to the env & ctx
        """
        trajs, FlatRewards, validity, total = self.synthesis_sampler.sample_from_model(GCN_model=GCN_model, Transformer_model=Transformer_model,
                                                                                       n=n, dev=dev, transforms=transforms, vocab=vocab, task=task
                                                                                       )
        if self.cfg.do_parameterize_p_b:
            self.dup_dict = calc_dup_dict(self.synthesis_sampler.smis)
        
        return trajs, FlatRewards, validity, total
        

    def add_logZ_pred_to_data(self, MLP_model, data, cond_info):
        logZ_pred = MLP_model.logZ(cond_info)
        for i in range(len(data)):
            data[i]["logZ"] = logZ_pred[i]
        return data

    def create_training_data_from_graphs(
        self,
        graphs,
        model: Optional[TrajectoryBalanceModel] = None,
        cond_info: Optional[Tensor] = None,
    ):
        """Generate synthesis trajectories from known endpoints

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints
        model: TrajectoryBalanceModel
           The model being sampled
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)

        Returns
        -------
        trajs: List[Dict{'traj': List[tuple[Graph, int]]}]
           A list of trajectories.
        """
        if self.cfg.do_sample_p_b:
            assert model is not None and cond_info is not None
            dev = self.ctx.device
            cond_info = cond_info.to(dev)
            return self.synthesis_sampler.sample_backward_from_graphs(
                graphs, model if self.cfg.do_parameterize_p_b else None, cond_info, dev
            )
        trajs = None
        for traj in trajs:
            n_back = [self.env.count_backward_transitions(gp) for gp, _ in traj["traj"][1:]] + [1]
            traj["bck_logprobs"] = (1 / torch.tensor(n_back).float()).log().to(self.ctx.device)
            traj["result"] = traj["traj"][-1][0]
        return trajs


    def construct_batch(self, root_smi, trajs, cond_beta, cond_enc, log_rewards):
        """Construct a batch from a list of trajectories and their information

        Parameters
        ----------
        trajs: List[List[tuple[Graph, int]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        log_rewards: Tensor
            The transformed log-reward (e.g. torch.log(R(x) ** beta) ) for each trajectory. Shape (N,)
        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        # Graphのリストをひとまとめにしてbatchというclassにする
        batch = Batch(root_smi, trajs, cond_beta, cond_enc, log_rewards)
        '''
        batch.smis: List[List[str]]
        len(batch.smis) == len(trajs)
        batch.templates: List[List[int]]
        '''
        batch.log_rewards = log_rewards
        return batch

    def compute_batch_losses(self, GCN_model: nn.Module, Transformer_model: nn.Module, MLP_model: nn.Module, 
                             src_transforms, tgt_transforms, data_dict, vocab, dev,
                             batch: Batch, num_bootstrap: int = 0  # type: ignore[override]
                             ):
        '''Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: TrajectoryBalanceModel
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: gd.Batch
          batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap: int
          the number of trajectories for which the reward loss is computed. Ignored if 0.
          
        batch.smis: List[List[str]]
        len(batch.smis) == len(trajs)
        batch.templates: List[List[int]]
        '''
        
        num_trajs = int(batch.traj_lens.shape[0])
        wandb.log({'num_molecules_per_minibatch': batch.traj_lens.sum()})
        log_rewards = batch.log_rewards
        # Clip rewards
        assert log_rewards.ndim == 1, "log_rewards must be a 1D tensor"
        
        clip_log_R = torch.maximum(
            log_rewards.to(dev), torch.tensor(self.global_cfg.algo.illegal_action_logreward, device=dev)
        ).float()
        cond_beta = batch.cond_beta # torch.Size([32])
        cond_enc = batch.cond_enc # (batch_size, n_enc), ex. (32, 32)
        cond_info = {'beta': cond_beta, 'encoding': cond_enc}
        # invalid_mask = 1 - batch.is_valid

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        # batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        smis_batch_idx = torch.arange(len(batch.smis)).repeat_interleave((batch.traj_lens-1) // 2)
        templates_batch_idx = torch.arange(len(batch.templates)).repeat_interleave((batch.traj_lens-1) // 2)
        assert (smis_batch_idx == templates_batch_idx).all(), 'smis_batch_idx and templates_batch_idx must be the same tensor'        # # The traj_len at which each graph is in its trajectory
        # traj_idcs = torch.cat([torch.arange(length) for length in batch.traj_lens])
        # # The position of the last graph of each trajectory
        # final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1
        
        GCN_model.train()
        Transformer_model.train()
        MLP_model.train()
        
        batch.smis = list(itertools.chain.from_iterable(batch.smis)) # [[smi1, smi2], [smi3, smi4, smi5]] -> [smi1, smi2, smi3, smi4, smi5]
        batch.templates = list(itertools.chain.from_iterable(batch.templates))
        batch.smis_GCN = list(itertools.chain.from_iterable(batch.smis_GCN))
        batch.smis_Transformer_src = list(itertools.chain.from_iterable(batch.smis_Transformer_src))
        batch.smis_Transformer_tgt = list(itertools.chain.from_iterable(batch.smis_Transformer_tgt))
        
        # Calculation of the probability that GCN_model predicts batch.templates from batch.smis
        probs_GCN = predict_GCN(self.global_cfg, batch, GCN_model, dev)

        # Calculation of the probability that Transformer_model predicts batch.products from batch.templates
        probs_Transformer = predict_Transformer(self.global_cfg, Transformer_model, batch, src_transforms,
                                                tgt_transforms, data_dict, vocab, dev)

        # Compute trajectory balance objective
        log_Z = MLP_model(cond_info['encoding']).to(dev).squeeze()
        
        if self.cfg.do_parameterize_p_b:
            log_p_B = calc_p_B(batch, self.dup_dict)
            traj_log_p_B = scatter(log_p_B.to(dev), smis_batch_idx.to(dev), dim=0, dim_size=num_trajs, reduce="sum")
        else:
            traj_log_p_B = torch.zeros_like(traj_log_p_F).to(dev)
        
        traj_log_p_F_GCN = scatter(probs_GCN, smis_batch_idx.to(dev), dim=0, dim_size=num_trajs, reduce="sum")
        traj_log_p_F_Transformer = scatter(probs_Transformer, templates_batch_idx.to(dev), dim=0, dim_size=num_trajs, reduce="sum")
        traj_log_p_F = traj_log_p_F_GCN + traj_log_p_F_Transformer

        numerator = log_Z + traj_log_p_F
        denominator = clip_log_R + traj_log_p_B
        if self.cfg.epsilon is not None:
            # Numerical stability epsilon
            epsilon = torch.tensor([self.cfg.epsilon], device=dev).float()
            numerator = torch.logaddexp(numerator, epsilon)
            denominator = torch.logaddexp(denominator, epsilon)
        traj_losses = (numerator - denominator).pow(2)
        wandb.log({
            'traj_losses': traj_losses.mean(),
            'numerator': numerator.mean(),
            'log_Z': log_Z.mean(),
            'p_F_GCN': traj_log_p_F_GCN.mean(), 
            'p_F_Transformer': traj_log_p_F_Transformer.mean(),
            'denominator': denominator.mean(),
            })
        # Normalize losses by trajectory length
        if self.length_normalize_losses:
            length = batch.traj_lens-1
            traj_losses = traj_losses / length.to(dev)
        if self.reward_normalize_losses:
            factor = -clip_log_R.min() + clip_log_R + 1
            factor = factor / factor.sum()
            assert factor.shape == traj_losses.shape
            traj_losses = factor * traj_losses * num_trajs
        loss = traj_losses.mean()
        info = {
            "offline_loss": traj_losses[: batch.num_offline].mean() if batch.num_offline > 0 else 0,
            "online_loss": traj_losses[batch.num_offline :].mean() if batch.num_online > 0 else 0,
            "logZ": log_Z.mean(),
            "loss": loss.item(),
            "logR": clip_log_R.mean(),
        }
        os.makedirs(self.log_dir, exist_ok=True)
        return loss, info


    def _init_subtb(self, dev):
        r"""Precompute all possible subtrajectory indices that we will use for computing the loss:
        \sum_{m=1}^{T-1} \sum_{n=m+1}^T
            \log( \frac{F(s_m) \prod_{i=m}^{n-1} P_F(s_{i+1}|s_i)}
                       {F(s_n) \prod_{i=m}^{n-1} P_B(s_i|s_{i+1})} )^2
        """
        ar = torch.arange(self._subtb_max_len, device=dev)
        # This will contain a sequence of repeated ranges, e.g.
        # tidx[4] == tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
        tidx = [torch.tril_indices(i, i, device=dev)[1] for i in range(self._subtb_max_len)]
        # We need two sets of indices, the first are the source indices, the second the destination
        # indices. We precompute such indices for every possible trajectory length.

        # The source indices indicate where we index P_F and P_B, e.g. for m=3 and n=6 we'd need the
        # sequence [3,4,5]. We'll simply concatenate all sequences, for every m and n (because we're
        # computing \sum_{m=1}^{T-1} \sum_{n=m+1}^T), and get [0, 0,1, 0,1,2, ..., 3,4,5, ...].

        # The destination indices indicate the index of the subsequence the source indices correspond to.
        # This is used in the scatter sum to compute \log\prod_{i=m}^{n-1}. For the above example, we'd get
        # [0, 1,1, 2,2,2, ..., 17,17,17, ...]

        # And so with these indices, for example for m=0, n=3, the forward probability
        # of that subtrajectory gets computed as result[2] = P_F[0] + P_F[1] + P_F[2].

        self._precomp = [
            (
                torch.cat([i + tidx[T - i] for i in range(T)]),
                torch.cat(
                    [ar[: T - i].repeat_interleave(ar[: T - i] + 1) + ar[T - i + 1 : T + 1].sum() for i in range(T)]
                ),
            )
            for T in range(1, self._subtb_max_len)
        ]

    def subtb_loss_fast(self, P_F, P_B, F, R, traj_lengths):
        r"""Computes the full SubTB(1) loss (all arguments on log-scale).

        Computes:
            \sum_{m=1}^{T-1} \sum_{n=m+1}^T
                \log( \frac{F(s_m) \prod_{i=m}^{n-1} P_F(s_{i+1}|s_i)}
                           {F(s_n) \prod_{i=m}^{n-1} P_B(s_i|s_{i+1})} )^2
            where T is the length of the trajectory, for every trajectory.

        The shape of P_F, P_B, and F should be (total num steps,), i.e. sum(traj_lengths). The shape
        of R and traj_lengths should be (num trajs,).

        Parameters
        ----------
        P_F: Tensor
            Forward policy log-probabilities
        P_B: Tensor
            Backward policy log-probabilities
        F: Tensor
            Log-scale flow predictions
        R: Tensor
            The log-reward of each trajectory
        traj_lengths: Tensor
            The length of each trajectory

        Returns
        -------
        losses: Tensor
            The SubTB(1) loss of each trajectory.
        """
        num_trajs = int(traj_lengths.shape[0])
        max_len = int(traj_lengths.max() + 1)
        dev = traj_lengths.device
        cumul_lens = torch.cumsum(torch.cat([torch.zeros(1, device=dev), traj_lengths]), 0).long()
        total_loss = torch.zeros(num_trajs, device=dev)
        ar = torch.arange(max_len, device=dev)
        car = torch.cumsum(ar, 0)
        F_and_R = torch.cat([F, R])
        R_start = F.shape[0]
        for ep in range(traj_lengths.shape[0]):
            offset = cumul_lens[ep]
            T = int(traj_lengths[ep])
            if self.cfg.do_parameterize_p_b:
                # The length of the trajectory is the padded length, reduce by 1
                T -= 1
            idces, dests = self._precomp[T - 1]
            fidces = torch.cat(
                [torch.cat([ar[i + 1 : T] + offset, torch.tensor([R_start + ep], device=dev)]) for i in range(T)]
            )
            P_F_sums = scatter_sum(P_F[idces + offset], dests)
            P_B_sums = scatter_sum(P_B[idces + offset], dests)
            F_start = F[offset : offset + T].repeat_interleave(T - ar[:T])
            F_end = F_and_R[fidces]
            total_loss[ep] = (F_start - F_end + P_F_sums - P_B_sums).pow(2).sum() / car[T]
        return total_loss

    def subtb_cum(self, P_F, P_B, F, R, traj_lengths):
        """
        Calcualte the subTB(1) loss (all arguments on log-scale) using dynamic programming.

        See also `subTB`
        """
        dev = traj_lengths.device
        num_trajs = len(traj_lengths)
        total_loss = torch.zeros(num_trajs, device=dev)
        x = torch.cumsum(traj_lengths, 0)
        # P_B is already shifted
        pdiff = P_F - P_B
        for ep, (s_idx, e_idx) in enumerate(zip(shift_right(x), x)):
            if self.cfg.do_parameterize_p_b:
                e_idx -= 1
            n = e_idx - s_idx
            fr = torch.cat([F[s_idx:e_idx], torch.tensor([R[ep]], device=F.device)])
            p = pdiff[s_idx:e_idx]
            total_loss[ep] = subTB(fr, p).pow(2).sum() / (n * n + n) * 2
        return total_loss