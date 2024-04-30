from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MulticlassClassifierConfig:
    batch_size : int = 32
    shuffle : bool = True
    lr : float = 1e-6
    n_epochs : int = 150
    save_dir : Optional[str] = None
    save_every : int = 900
    input_dim : int = 1024
    n_hidden_layers : int = 3
    hidden_dims : list[int] = field(default_factory=lambda: [16384, 8192, 2048, 512])
    output_dim : int = 10
    model_type : str = "multiclass_classifier"
    loss_type : str = "MSE"
    save_dir : str = "/data/hayano/perplex_warmup5_minquery20_thresh4/reward_classifier"
    n_data_needed_for_training : int = 20
    name : str = "human_reward_predictor"