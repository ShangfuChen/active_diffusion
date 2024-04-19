import torch
from torch import nn

# class RewardPredictorModel(nn.Module):
#     def __init__(
#             self,
#             input_dim,
#             hidden_dim,
#             n_hidden_layers=5,
#             device=None,
#         ):

#         super(RewardPredictorModel, self).__init__()

#         if device is None:
#             if torch.cuda.is_available():
#                 device = "cuda"
#             else:
#                 device = "cpu"

#         layers = [nn.Linear(input_dim, hidden_dim)]
#         layers += [nn.Linear(hidden_dim, hidden_dim)] * n_hidden_layers
#         layers += [nn.Linear(hidden_dim, 1)]
#         self.model = nn.Sequential(*layers).to(device)
#         self.float()

#     def forward(self, x):
#         latents = self.model(x)
#         return latents

class RewardPredictorModel(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims=[16384]*6,
            n_hidden_layers=5,
            device=None,
        ):

        super(RewardPredictorModel, self).__init__()

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
        ]
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.model = nn.Sequential(*layers).to(device)
        self.float()

    def forward(self, x):
        latents = self.model(x)
        return latents

class RewardPredictorModel2(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims=[16384, 8192, 4096, 2048, 1024, 512],
            n_hidden_layers=5,
            device=None,
        ):

        super(RewardPredictorModel2, self).__init__()

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
        ] 
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.model = nn.Sequential(*layers).to(device)
        self.float()

    def forward(self, x):
        latents = self.model(x)
        return latents
    

# class RewardPredictorCNN(nn.Module):
#     def __init__(
#             self,
#             input_dim,
#             hidden_dims=[16384, 8192, 4096, 2048, 1024, 512],
#             n_hidden_layers=5,
#             device=None,
#         ):

#         super(RewardPredictorCNN, self).__init__()

#         if device is None:
#             if torch.cuda.is_available():
#                 device = "cuda"
#             else:
#                 device = "cpu"

#         layers = [nn.Linear(input_dim, hidden_dims[0])] # input layer
#         layers  += [nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(n_hidden_layers)]
#         layers += [nn.Linear(hidden_dims[-1], 1)]
#         self.model = nn.Sequential(*layers).to(device)
#         self.float()

#     def forward(self, x):
#         latents = self.model(x)
#         return latents
    

