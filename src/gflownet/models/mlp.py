import torch.nn as nn
from gflownet.config import Config

def mlp(n_in, n_hid, n_out, n_layer, act=nn.LeakyReLU):
    """Creates a fully-connected network with no activation after the last layer.
    If `n_layer` is 0 then this corresponds to `nn.Linear(n_in, n_out)`.
    """
    n = [n_in] + [n_hid] * n_layer + [n_out] # in10, hidden20*3, out10だと [10, 20, 20, 20, 10] みたいなリストになる
    # Linear->activation->Linear->activation->...->Linearのnn.Sequentialを作る
    # [:-1]で最後のactivation(ReLU)を取り除く
    return nn.Sequential(*sum([[nn.Linear(n[i], n[i + 1]), act()] for i in range(n_layer + 1)], [])[:-1])

class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_layer, act=nn.LeakyReLU):
        super().__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layer = n_layer
        self.mlp = mlp(self.n_in, self.n_hid, self.n_out, self.n_layer, act)

    def forward(self, x):
        logZ_pred = self.mlp(x)
        return logZ_pred


