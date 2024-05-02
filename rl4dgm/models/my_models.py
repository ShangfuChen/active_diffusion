import torch
from torch import nn
import numpy as np

class LinearModel(nn.Module):
    def __init__(
            self,
            input_dim,
            seed,
            hidden_dims=[16384]*6,
            output_dim=2048,
            device=None,
        ):
        """
        Series of linear layers with ReLU activations        
        """
        super(LinearModel, self).__init__()

        # set seed
        self.seed = seed
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        # input layer
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
        ] 
        
        # hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        # output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        # layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers).to(device)
        self.float()

    def forward(self, x):
        latents = self.model(x)
        return latents
    
class MultiClassClassifierModel(nn.Module):
    def __init__(
        self,
        input_dim,
        seed,
        hidden_dims=[512]*4,
        output_dim=10,
        device=None,      
    ):
        """
        Series of linear layers with ReLU activations, followed by a softmax activation         
        """
        super(MultiClassClassifierModel, self).__init__()

        # set seed
        self.seed = seed
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        # input layer
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
        ] 
        
        # hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        # output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        # layers.append(nn.Softmax(dim=0))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers).to(device)
        self.float()

    def forward(self, x):
        class_confidence = self.model(x)
        return class_confidence
    
class EnsembleLinearModels(nn.Module):
    def __init__(
        self,
        input_dim,
        seed,
        hidden_dims=[512, 512, 512, 512],
        output_dim=1,
        n_models=3,
        device=None,      
    ):
        """
        Ensemble of LinearModels       
        """
        super(EnsembleLinearModels, self).__init__()

        # set seed
        self.seed = seed
        # self.model_seeds = np.arange(n_models)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.models = [
            LinearModel(
                input_dim=input_dim,
                seed=self.seed,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                device=device,
            ) for i in range(n_models)
        ]

        for i, model in enumerate(self.models):
            self.add_module(f"model{i}", model)
        
        self.float()

    def forward(self, x):
        outputs = torch.cat([model(x) for model in self.models], dim=1)
        final_outputs = outputs.mean(dim=1)
        return final_outputs
    
    def get_all_model_outputs_mean_std(self, x):
        """
        Return all model outputs stacked, mean, and std
        """
        outputs = torch.cat([model(x) for model in self.models], dim=1)
        output_mean = outputs.mean(dim=1)
        output_std = outputs.std(dim=1)

        return outputs, output_mean, output_std
