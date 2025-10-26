from typing import List

import numpy as np
import torch

from gflownet.config import Config


class ReplayBuffer(object):
    def __init__(self, cfg: Config, rng: np.random.Generator = None):
        self.capacity = cfg.replay.capacity
        self.warmup = cfg.replay.warmup
        assert self.warmup <= self.capacity, "ReplayBuffer warmup must be smaller than capacity"

        # bufferにはstate, action, rewardなど複数の情報がtupleで入ることが想定されている。
        self.buffer: List[tuple] = []
        self.position = 0
        self.rng = rng

    def push(self, *args):
        if len(self.buffer) == 0:
            self._input_size = len(args)
        else:
            assert self._input_size == len(args), "ReplayBuffer input size must be constant"
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = args
        
        # position + 1 % capacityで、positionがcapacityを超えたら0に戻るので上書きされる
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        num = min(len(self.buffer), batch_size)
        idxs = self.rng.choice(len(self.buffer), num)
        # zip(*[self.buffer[idx] for idx in idxs])で、self.bufferのidx番目の要素を取り出して、それをzipでまとめている。
        out = list(zip(*[self.buffer[idx] for idx in idxs]))
        for i in range(len(out)):
            # stack if all elements are numpy arrays or torch tensors
            # (this is much more efficient to send arrays through multiprocessing queues)
            if all([isinstance(x, np.ndarray) for x in out[i]]):
                out[i] = np.stack(out[i], axis=0)
            elif all([isinstance(x, torch.Tensor) for x in out[i]]):
                out[i] = torch.stack(out[i], dim=0)
        return tuple(out)

    def __len__(self):
        return len(self.buffer)
