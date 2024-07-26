from dataclasses import dataclass, field
from typing import Optional


######### NEW ENCODER ##############
@dataclass
class HumanEncoderConfig:
    batch_size : int = 2048 # try >1000
    # batch_size : int = 50
    shuffle : bool = True
    lr : float = 1e-5
    n_epochs : int = 5000
    # n_epochs : int = 10
    save_dir : Optional[str] = None
    save_every : int = 1
    agent1_triplet_margin : float = 1.0
    agent2_triplet_margin : float = 1.0
    agent1_loss_weight : float = 1.0
    agent2_loss_weight : float = 0.25
    input_dim : int = 4096
    n_hidden_layers : int = 1
    hidden_dims : list[int] = field(default_factory=lambda: [2048, 1024])
    output_dim : int = 512
    # save_dir : str = "/data/hayano/ensemble_warmup5_minquery20_dynamicthresh_0_6_minq10always_unnormalizedpickscore2/human_encoder"
    n_data_needed_for_training : int = 128
    # n_data_needed_for_training : int = 100
    n_warmup_epochs: int = 0
    name : str = "human_encoder"


########### OLD TRIPLET ENCODER ############
# @dataclass
# class HumanEncoderConfig:
#     batch_size : int = 32
#     # batch_size : int = 50
#     shuffle : bool = True
#     lr : float = 1e-6
#     n_epochs : int = 400
#     # n_epochs : int = 10
#     save_dir : Optional[str] = None
#     save_every : int = 1
#     agent1_triplet_margin : float = 1.0
#     agent2_triplet_margin : float = 1.0
#     agent1_loss_weight : float = 1.0
#     agent2_loss_weight : float = 0.25
#     input_dim : int = 16384
#     n_hidden_layers : int = 3
#     hidden_dims : list[int] = field(default_factory=lambda: [16384, 8192, 2048, 512])
#     output_dim : int = 512
#     # save_dir : str = "/data/hayano/ensemble_warmup5_minquery20_dynamicthresh_0_6_minq10always_unnormalizedpickscore2/human_encoder"
#     n_data_needed_for_training : int = 64
#     # n_data_needed_for_training : int = 100
#     n_warmup_epochs: int = 0
#     name : str = "human_encoder"