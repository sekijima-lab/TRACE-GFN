from typing import List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from gflownet.models.GCN.utils import template_prediction, check_templates
from gflownet.utils.utils import smi_tokenizer, MW_checker, extract_retained_substructures, sub_matching, rxn_filter, clean_nan
from gflownet.models.Transformer.beam_search import beam_decode, prob_sampling
import random
import numpy as np
import json
import rdkit.Chem as Chem
import time

class Node:
    def __init__(self):
        self.parent = None
        self.children = []
        self.path = []
        # self.visit = 1
        # self.imm_score = 0
        # self.cum_score = 0
        # self.c = 1
        # self.id = -1
        # self.rollout_result = ('None', -1000)

    def add_Node(self, c):
        raise NotImplementedError()

    def select_children(self):
        try:
            child_probs = []
            for child_node in self.children:
                child_probs.append(child_node.prob)
            child_probs = np.array(child_probs) / np.sum(child_probs)
            # 確率によって子ノードを選択
            ind = np.random.choice(np.arange(len(child_probs)), p=child_probs)
            return self.children[ind]
        except ValueError:
            print('ValueError: child_probs is empty.')
            return None

    
class MolecularNode(Node):
    def __init__(self, smi, prob, depth):
        super().__init__()
        self.smi = smi
        self.prob = prob
        self.depth = depth
        self.flat_reward = 0
        self.no_template_flag = False
        self.stop_flag = False
        self.patience = 0
    def add_Node(self, c: 'ReactionTemplateNode'):
        assert isinstance(c, ReactionTemplateNode)
        c.parent = self
        self.children.append(c)
    def remove_Node(self):
        self.parent.children.remove(self)


class ReactionTemplateNode(Node):
    def __init__(self, template, prob):
        super().__init__()
        self.template = template
        self.prob = prob
        self.no_mol_flag = False
    def add_Node(self, c: MolecularNode):
        assert isinstance(c, MolecularNode)
        c.parent = self
        # c.depth = self.depth + 1
        self.children.append(c)
    def remove_Node(self):
        self.parent.children.remove(self)
    

class SynthesisSampler:

    def __init__(self, max_depth, pad_with_terminal_state, tgt_max_len, cfg):
        """
        Parameters
        ----------
        env: ReactionTemplateEnv
            A reaction template environment.
        ctx: ReactionTemplateEnvContext
            A context.
        max_depth: int
            If not None, ends trajectories of more than max_depth steps.
        pad_with_terminal_state: bool
        """
        self.max_depth = max_depth if max_depth is not None else 10
        self.pad_with_terminal_state = pad_with_terminal_state
        self.global_cfg = cfg
        self.data_cfg = self.global_cfg.data
        self.init_compound = self.data_cfg.init_compound
        self.r_dict_path = self.data_cfg.label_template_path
        self.num_template_sampling = self.global_cfg.algo.num_template_sampling
        self.no_template = False
        self.n_invalid = 0
        self.tgt_max_len = tgt_max_len
        self.inf_max_len = tgt_max_len - 2
        self.beam_width = self.global_cfg.algo.beam_width
        self.nbest = self.global_cfg.algo.beam_nbest
        self.beam_templates_path = self.global_cfg.data.beam_template_path
        self.gen_nbest = self.global_cfg.algo.gen_nbest
        self.gen_patience = self.global_cfg.algo.gen_patience
        with open(self.r_dict_path, 'r') as f:
            self.r_dict = json.load(f)
        with open(self.beam_templates_path, 'r') as f:
            self.beam_templates = f.read().splitlines()
        self.num_gen = 0
        self.uniques = 0
        self.smis = []
        self.templates = []
        self.invalid_templates = []
    
    
    def sample_from_model(self, GCN_model: nn.Module, Transformer_model: nn.Module, n: int,
                            dev: torch.device, transforms: nn.Module, vocab, task
                            ):
        # 反応テンプレートを1つだけサンプリングすることでtrajectoryを作成
        # MCTSの時と同じようにサンプリングされた全てのテンプレートに対して生成物を生成する方法も考えられる
        with torch.no_grad():
            GCN_model.eval()
            Transformer_model.eval()
            
            self.smis = []
            init_node = MolecularNode(smi=self.init_compound, prob=1, depth=0)
            self.smis.append(self.init_compound)
            trajectories = []
            FlatRewards = []
            time_start = time.time()
            valid = 0
            total = 0
            for i in tqdm(range(n), desc='Sampling trajectories...'):
            # for i in range(n):
            #     print(f'iteration: {i+1} / {n}')
                trajs = []
                gen_mol_iter = 0
                patience = 0
                while len(trajs) == 0:
                    patience += 1
                    if patience > 10:
                        print('cannot find valid trajectory.')
                        break
                    node = init_node
                    flag = False
                    # select leaf MolecularNode
                    count = 0
                    while len(node.children) != 0:
                        count += 1
                        node = node.select_children()
                        if count > 100:
                            flag = True
                            break
                        elif isinstance(node, MolecularNode) and node.stop_flag == True:
                            continue
                        elif isinstance(node, MolecularNode) and node.no_template_flag == True:
                            continue
                        elif isinstance(node, ReactionTemplateNode) and node.no_mol_flag == True:
                            continue

                    if flag == True:
                        print('cannot find valid leaf node.')
                        continue
                    
                    # 選ばれたノードがMolecularNodeの場合
                    if isinstance(node, MolecularNode):
                        queue = [node]
                        trajs, flatrewards, v, t = self.queue_to_trajectory(queue, GCN_model, Transformer_model, n, dev, transforms, vocab, task)
                        trajectories.extend(trajs)
                        FlatRewards.extend(flatrewards)
                        valid += v
                        total += t
                        gen_mol_iter += len(trajs)
                    # 選ばれたノードがReactionTemplateNodeの場合
                    elif isinstance(node, ReactionTemplateNode):
                        above_thresholds, trajs, flatrewards, v, t = self.conditional_generation(node, node.parent, task, dev, vocab, Transformer_model, transforms)
                        trajectories.extend(trajs)
                        FlatRewards.extend(flatrewards)
                        valid += v
                        total += t
                        gen_mol_iter += len(trajs)
                        if len(above_thresholds) != 0:
                            num = min(self.gen_nbest, len(above_thresholds))
                            queue = random.sample(above_thresholds, num)
                        else:
                            continue
                        trajs, flatrewards, v, t = self.queue_to_trajectory(queue, GCN_model, Transformer_model, n, dev, transforms, vocab, task)
                        trajectories.extend(trajs)
                        FlatRewards.extend(flatrewards)
                        valid += v
                        total += t
                        gen_mol_iter += len(trajs)
                        
        time_end = time.time()
        validity = valid/total if total != 0 else 0
        print(f'Sampling trajectories finished. Time: {time_end - time_start:.2f} seconds')
        print(f'Validity: {validity:.2f}')
        return trajectories, FlatRewards, validity, total
                        
    def queue_to_trajectory(self, queue: List, GCN_model: nn.Module, Transformer_model: nn.Module,
                            n: int, dev: torch.device, transforms: nn.Module, vocab, task=None):
        trajectories = []
        FlatRewards = []
        valid = 0
        total = 0
        queue_count = 0
        while len(queue) != 0:
            queue_count += 1
            if queue_count > self.global_cfg.algo.max_depth:
                print('Exceed max depth.')
                break
            candidates = []
            q_copy = queue.copy()
            for q_node in q_copy:
                input_smi = q_node.smi
                depth = q_node.depth
                if depth == self.max_depth:
                    q_node.stop_flag = True
                    continue

                num_sampling = self.num_template_sampling
                for _ in range(10):
                    probs, indices = template_prediction(GCN_model=GCN_model, input_smi=input_smi,
                                                        num_sampling=num_sampling, GCN_device=dev)
                    # convert from log softmax to softmax
                    probs_exp = torch.exp(probs)
                    matched_probs, matched_indices = check_templates(probs_exp, indices, input_smi, self.r_dict)
                    if len(matched_indices) < 3 or (None in matched_probs):
                        num_sampling += 5
                        continue
                    else:
                        break
                if len(matched_indices) == 0:
                    print(f'Cannot find valid reaction template for {input_smi}.')
                    q_node.no_template_flag = True
                    continue
                sum_prob = sum(matched_probs)
                if sum_prob == 0:
                    q_node.no_template_flag = True
                    continue
                matched_probs_n = [p / sum_prob for p in matched_probs]
                matched_probs_n_np = np.array(matched_probs_n)
                contains_nan = np.isnan(matched_probs_n_np).any()
                if contains_nan:
                    matched_indices, matched_probs_n = clean_nan(matched_indices, matched_probs_n)
                    
                for matched_template, matched_prob in zip(matched_indices, matched_probs_n):
                    template_node = ReactionTemplateNode(matched_template, matched_prob)
                    q_node.add_Node(template_node)
                    
                # sample template node
                while True:
                    if len(q_node.children) == 0:
                        q_node.stop_flag = True
                        q_node.no_template_flag = True
                        break
                    template_node = q_node.select_children()
                    if template_node == None:
                        break
                    assert isinstance(template_node, ReactionTemplateNode)
                    above_thresholds, trajs, flatrewards, v, t = self.conditional_generation(template_node, q_node,
                                                                                             task, dev, vocab,
                                                                                             Transformer_model, transforms)
                    if len(trajs) != 0:
                        valid += v
                        total += t
                        trajectories.extend(trajs)
                        FlatRewards.extend(flatrewards)
                        candidates.extend(above_thresholds)
                        break
                    else:
                        template_node.remove_Node()
                        continue
                    
            if len(candidates) != 0:
                num = min(self.gen_nbest, len(candidates))
                queue = random.sample(candidates, num)
            else:
                break
        return trajectories, FlatRewards, valid, total

    def conditional_generation(self, template_node: ReactionTemplateNode, parent_node: MolecularNode, task, dev, vocab, Transformer_model, transforms):
        valid, total = 0, 0
        template = template_node.template
        parent_smi = parent_node.smi
        reward_threshold = parent_node.flat_reward
        p_depth = parent_node.depth
        idx_str = f'[{template}]'
        input_conditional = smi_tokenizer(idx_str + parent_smi).split(' ')
        input_tokens = transforms(input_conditional).to(dev)
        temperature = self.global_cfg.algo.temperature
        if self.global_cfg.algo.sampling_method == 'prob_sampling':
            sampling_size = self.global_cfg.algo.prob_sampling_size
            input_tokens = input_tokens.repeat(sampling_size, 1)
            outputs, out_scores = prob_sampling(v=vocab, model=Transformer_model, input_tokens=input_tokens,
                                                device=dev, inf_max_len=self.inf_max_len, temperature=temperature)
            self.num_gen += self.global_cfg.algo.prob_sampling_size
            self.uniques += len(outputs)
        elif self.global_cfg.algo.sampling_method == 'beam_search':
            outputs, out_scores = beam_decode(v=vocab, model=Transformer_model, input_tokens=input_tokens, template_idx=idx_str,
                                        device=dev, inf_max_len=self.inf_max_len, beam_width=self.beam_width,
                                        nbest=self.nbest, Temp=temperature, beam_templates=self.beam_templates)
        else:
            raise ValueError(f'Invalid sampling method: {self.global_cfg.algo.sampling_method}')
        
        total += len(outputs)
        query_substructures = extract_retained_substructures(seed_smi=parent_smi, template_idx=template, r_dict=self.r_dict)
        smis, mols, scores = [], [], []
        for output, score in zip(outputs, out_scores):
            if len(output) > (self.tgt_max_len - 2):
                continue
            mol = Chem.MolFromSmiles(output)
            if mol is None:
                continue
            if (output != '') and (len(output) > 1): # remove '' and single atom molecules to avoid mol2vec error of GCN
                valid += 1
                if sub_matching(mol, query_substructures):
                    if rxn_filter(parent_smi, template, output, self.r_dict):
                        smis.append(output)
                        mols.append(mol)
                        scores.append(score)
                    
        if len(mols) == 0:
            template_node.no_mol_flag = True
            return [], [], [], 0, 0
        flat_rewards, _ = task.compute_flat_rewards(mols)
        sum_score = sum(scores)
        out_tup = list(zip(smis, scores, flat_rewards))
        out_tup.sort(key=lambda x: x[2], reverse=True)
        above_thresholds = []
        for smi, score, flat_reward in out_tup:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if not MW_checker(mol, threshold=750):
                continue
            flat_reward = flat_reward.item()
            prob = score / sum_score
            if prob is None:
                continue
            molecular_node = MolecularNode(smi=smi, prob=prob, depth=p_depth+1)
            molecular_node.flat_reward = flat_reward
            self.smis.append(smi)
            template_node.add_Node(molecular_node)
            if reward_threshold <= flat_reward:
                if molecular_node.depth < self.max_depth:
                    above_thresholds.append(molecular_node)
            elif reward_threshold > flat_reward:
                molecular_node.patience += 1
                if molecular_node.patience > self.gen_patience:
                    molecular_node.stop_flag = True
        trajectories, flatrewards = [], []
        for cn in template_node.children:
            trajectory = []
            flat_rew = cn.flat_reward
            count = 0
            flag = False
            trajectory.append(cn.smi)
            current_mol_node = MolecularNode(smi='dummy', prob=1, depth=p_depth)
            while cn.parent != None:
                count += 1
                cn = cn.parent
                if isinstance(cn, MolecularNode):
                    if current_mol_node.smi == cn.smi:
                        current_template_node.no_mol_flag = True
                        trajectory.pop(-1)
                        break
                    trajectory.append(cn.smi)
                    current_mol_node = cn
                elif isinstance(cn, ReactionTemplateNode):
                    trajectory.append(str(cn.template))
                    current_template_node = cn
                if count > 100:
                    flag = True
                    break
            if flag == True:
                continue
            trajectory.reverse()
            trajectory = '.'.join(trajectory)
            trajectories.append(trajectory)
            flatrewards.append(flat_rew)
        return above_thresholds, trajectories, flatrewards, valid, total
    
    def calc_unique_rate(self):
        if self.num_gen == 0:
            return 0
        else: 
            return self.uniques / self.num_gen