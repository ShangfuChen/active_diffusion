from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ErrorPredictorConfig:
    batch_size : int =32
    shuffle : bool = True
    lr : float = 1e-6
    n_epochs : int = 500
    save_dir : Optional[str] = None
    save_every : int = 3000
    input_dim : int = 1024
    n_hidden_layers : int = 3
    hidden_dims : list[int] = field(default_factory=lambda: [512, 512, 512, 512])
    output_dim : int = 1
    model_type : str = "error_predictor"
    loss_type : str = "MSE"
    save_dir : str = "/data/shangfu/full_pipeline_debug2/error_predictor"
    name : str = "error_predictor"