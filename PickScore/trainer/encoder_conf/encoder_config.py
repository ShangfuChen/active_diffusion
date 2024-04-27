from dataclasses import dataclass
from typing import Optional

@dataclass
class AIEncoderConfig:
    batch_size : int = 32
    shuffle : bool = True
    lr : float = 1e-6
    n_epochs : int = 50
    triplet_margin : float = 1.0
    save_dir : Optional[str] = None
    save_every : int = 50
    input_dim : int = 16384
    n_hidden_layers : int = 5

@dataclass
class HumanEncoderConfig:
    feedback_agent : str = "ai"
    query_type : str = "random"
    n_feedbacks_per_query : int = 20
