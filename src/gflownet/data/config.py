from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Replay buffer configuration

    Attributes
    ----------
    use : bool
        Whether to use a replay buffer
    capacity : int
        The capacity of the replay buffer
    warmup : int
        The number of samples to collect before starting to sample from the replay buffer
    hindsight_ratio : float
        The ratio of hindsight samples within a batch
    """

    use: bool = False
    capacity: Optional[int] = None
    warmup: Optional[int] = None
    hindsight_ratio: float = 0
    init_compound:str = 'c1ccc(C2CCCNC2)cc1'
    label_template_path:str = Path(__file__).parent / 'label_template.json'
    USPTO_path:str = Path(__file__).parent / 'USPTO'
    beam_template_path:str = Path(__file__).parent / 'beamsearch_template_list.txt'
    

@dataclass
class ReplayConfig:
    """Replay buffer configuration

    Attributes
    ----------
    use : bool
        Whether to use a replay buffer
    capacity : int
        The capacity of the replay buffer
    warmup : int
        The number of samples to collect before starting to sample from the replay buffer
    hindsight_ratio : float
        The ratio of hindsight samples within a batch
    """

    use: bool = True
    capacity: Optional[int] = 10000
    warmup: Optional[int] = None
    hindsight_ratio: float = 0
