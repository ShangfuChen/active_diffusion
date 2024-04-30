import torch
from torch import nn

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
    