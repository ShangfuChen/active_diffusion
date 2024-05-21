import copy

import torch
from torch import nn, vmap
from torch.func import stack_module_state, functional_call

import numpy as np

# from torchensemble import VotingRegressor

class LinearModel(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims=[16384]*6,
            output_dim=2048,
            device=None,
            model_initialization_weight:float=None,
            model_initialization_seed=None,
        ):
        """
        Series of linear layers with ReLU activations        
        """
        super(LinearModel, self).__init__()
        
        # # set seed
        # self.seed = seed
        # torch.manual_seed(seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        # torch.cuda.manual_seed(seed)

        if model_initialization_seed is not None:
            torch.manual_seed(model_initialization_seed)
        
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

        # initialize model weights
        if model_initialization_weight is not None:
            weight_scale = model_initialization_weight
            for layer in self.model.children():
                if isinstance(layer, nn.Linear):
                    layer.weight.data = torch.clip(weight_scale * torch.ones_like(layer.weight), -0.3, 0.3)
                    weight_scale *= -1.1

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
        # seed,
        hidden_dims=[512, 512, 512, 512],
        output_dim=1,
        n_models=3,
        device=None,      
    ):
        """
        Ensemble of LinearModels       
        """
        super(EnsembleLinearModels, self).__init__()

        # # set seed
        # self.seed = seed
        # # self.model_seeds = np.arange(n_models)
        # torch.manual_seed(seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        # torch.cuda.manual_seed(seed)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        model_initialization_weights = np.random.uniform(-0.15, 0.15, n_models)
        model_initialization_seeds = np.arange(n_models)
        self.models = [
            LinearModel(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                device=device,
                # model_initialization_weight=model_initialization_weights[i],
                model_initialization_seed=model_initialization_seeds[i],
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


class LinearModelforEnsemble(nn.Module):
    def __init__(self):
        
        super(LinearModelforEnsemble, self).__init__()
        
        # model parameters
        input_dim = 1024
        hidden_dims = [512, 512, 512, 512]
        output_dim = 1

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

        self.model = nn.Sequential(*layers)

        self.float()

    def forward(self, x):
        return self.model(x)
