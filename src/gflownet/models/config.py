from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


# @dataclass
# class GraphTransformerConfig:
#     num_heads: int = 2
#     ln_type: str = "pre"
#     num_mlp_layers: int = 0


# class SeqPosEnc(Enum):
#     Pos = 0
#     Rotary = 1


# @dataclass
# class SeqTransformerConfig:
#     num_heads: int = 2
#     posenc: SeqPosEnc = SeqPosEnc.Rotary


@dataclass
class ModelConfig:
    """Generic configuration for models

    Attributes
    ----------
    num_layers : int
        The number of layers in the model
    num_emb : int
        The number of dimensions of the embedding
    """

    num_layers: int = 3
    num_emb: int = 128
    dropout: float = 0
    dim_GCN: int = 256
    n_conv_hidden: int = 1
    n_mlp_hidden: int = 3
    dropout_GCN: float = 0.1
    lr_GCN: float =  0.0004
    ckpt_GCN: str = Path(__file__).parent / 'ckpts' / 'GCN' / 'GCN.pth'
    d_Transformer: int = 512
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    nhead: int = 8
    dropout_Transformer: float = 0.1
    dim_ff: int = 2048
    ckpt_Transformer: str = Path(__file__).parent / 'ckpts' / 'Transformer' / 'Transformer.pth'
    lr_Transformer: float = 0.0004
    betas_Transformer: tuple = (0.9, 0.998)
    num_mlp_hidden: int = 256
    glb_batch_size:int = 16
    gfn_ckpt_GCN: Optional[str] = None
    gfn_ckpt_Transformer: Optional[str] = None
    gfn_ckpt_MLP: Optional[str] = None
    
    
@dataclass
class TrainConfig:
    """Generic configuration for models

    Attributes
    ----------
    num_layers : int
        The number of layers in the model
    num_emb : int
        The number of dimensions of the embedding
    """

    dim_GCN: int = 256
    n_conv_hidden: int = 1
    n_mlp_hidden: int = 3
    dropout_GCN: float = 0.1
    lr_GCN: float =  0.0004
    batch_size_GCN:int = 128
    epochs_GCN:int = 100
    patience_GCN:int = 5
    
    
    d_Transformer: int = 512
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    nhead: int = 8
    dropout_Transformer: float = 0.1
    dim_ff: int = 2048
    lr_Transformer: float = 0.0004
    betas_Transformer: tuple = (0.9, 0.998)
    num_mlp_hidden: int = 256
    batch_size_Transformer:int = 128
    
