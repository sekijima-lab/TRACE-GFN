from dataclasses import dataclass


@dataclass
class TaskConfig:
    reward_type: str = "qsar"
    add_sa_score: bool = True
    sa_score_coeff: float = 1.0
    sa_max: float = 5.0
    sa_min: float = 1.0
    