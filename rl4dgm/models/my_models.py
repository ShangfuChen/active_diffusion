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


class UncertaintyModel(nn.Module):
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
        super(UncertaintyModel, self).__init__()

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
        
        self.base_model = nn.Sequential(*layers).to(device)
        # output layer
        self.mean_head = nn.Linear(hidden_dims[-1], output_dim).to(device)
        self.log_std_head = nn.Linear(hidden_dims[-1], output_dim).to(device)
        self.log_std_low = -5
        self.log_std_high = 2

        self.float()

    def forward(self, x):
        feats = self.base_model(x)
        mean = self.mean_head(feats)
        log_std = torch.tanh(self.log_std_head(feats))
        log_std = self.log_std_low + 0.5 * (
                self.log_std_high - self.log_std_low
        ) * (log_std + 1)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.rsample()
    
    def calculate_std(self, x):
        feats = self.base_model(x)
        log_std = torch.tanh(self.log_std_head(feats))
        log_std = self.log_std_low + 0.5 * (
                self.log_std_high - self.log_std_low
        ) * (log_std + 1)
        std = log_std.exp()
        return std



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
        layers.append(nn.Softmax(dim=0))

        self.model = nn.Sequential(*layers).to(device)
        self.float()

    def forward(self, x):
        class_confidence = self.model(x)
        return class_confidence
    