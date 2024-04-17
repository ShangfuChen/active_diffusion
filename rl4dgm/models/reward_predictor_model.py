import torch
from torch import nn

class RewardPredictorModel(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            n_hidden_layers=5,
            device=None,
        ):

        super(RewardPredictorModel, self).__init__()

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        layers = [nn.Linear(input_dim, hidden_dim)]
        layers += [nn.Linear(hidden_dim, hidden_dim)] * n_hidden_layers
        layers += [nn.Linear(hidden_dim, 1)]
        self.model = nn.Sequential(*layers).to(device)
        self.float()

    def forward(self, x):
        latents = self.model(x)
        return latents
    
