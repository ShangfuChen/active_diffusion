from dataclasses import dataclass, field
from typing import Optional

@dataclass
class HumanEncoderConfig:
    batch_size : int = 32
    shuffle : bool = True
    lr : float = 1e-6
    n_epochs : int = 150
    save_dir : Optional[str] = None
    save_every : int = 150
    agent1_triplet_margin : float = 1.0
    agent2_triplet_margin : float = 1.0
    agent1_loss_weight : float = 1.0
    agent2_loss_weight : float = 0.25
    input_dim : int = 16384
    n_hidden_layers : int = 3
    hidden_dims : list[int] = field(default_factory=lambda: [16384, 8192, 2048, 512])
    output_dim : int = 512
    save_dir : str = "/data/hayano/full_pipeline_test6/human_encoder"
    n_data_needed_for_training : int = 32
    name : str = "human_encoder"