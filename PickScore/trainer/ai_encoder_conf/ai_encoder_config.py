from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AIEncoderConfig:
    batch_size : int = 32
    shuffle : bool = True
    lr : float = 1e-6
    n_epochs : int = 150
    # n_epochs : int = 10

    triplet_margin : float = 1.0
    save_dir : Optional[str] = None
    save_every : int = 1500
    input_dim : int = 16384
    n_hidden_layers : int = 5
    hidden_dims : list[int] = field(default_factory=lambda: [16384, 8192, 2048, 512])
    output_dim : int = 512
    # save_dir : str = "/data/hayano/ensemble_warmup5_minquery20_dynamicthresh_0_6_minq10always_unnormalizedpickscore2/ai_encoder"
    name : str = "ai_encoder"

