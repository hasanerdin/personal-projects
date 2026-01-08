from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F
import numpy as np
import math


class HeadLayer(torch.nn.Module):
    """ Multi-function head layer. Structure of the layer changes depending on
    the "extensions" dictionary. If "noisy" is active, Linear layers become
    Noisy Linear. If "dueling" is active, the dueling architecture must be employed,
    and lastly, if "distributional" is active, output shape should change
    accordingly.

    Args:
        in_size (int): Input size of the head layer
        act_size (int): Action size
        extensions (Dict[str, Any]): A dictionary that keeps extension information
        hidden_size (Optional[int], optional): Size of the hidden layer in Dueling 
        architecture. Defaults to None.

    Raises:
        ValueError: if hidden_size is not given while dueling is active
    """

    def __init__(self, in_size: int, act_size: int, extensions: Dict[str, Any],
                 hidden_size: Optional[int] = None):
        super().__init__()

        assert not(extensions["dueling"] and hidden_size is None), \
            "The hidden size should be defined if dueling is wanted to use."

        self.extensions = extensions
        self.n_atoms = extensions["distributional"]["natoms"] if extensions["distributional"] else 1

        # Defining default linear layer lambda function which takes two parameter
        if extensions["noisy"]:
            linear_layer = lambda in_feature, out_feature: NoisyLinear(in_feature, 
                                                                       out_feature, 
                                                                       extensions["noisy"]["init_std"])
        else:
            linear_layer = lambda in_feature, out_feature: nn.Linear(in_feature, out_feature)

        # Defining fc layers according to dualing, noisy and distributional extensions
        if extensions["dueling"]:
            self.fc_advantage = nn.Sequential(
                    linear_layer(in_size, hidden_size),
                    nn.ReLU(),
                    linear_layer(hidden_size, act_size * self.n_atoms)
                )

            self.fc_value = nn.Sequential(
                    linear_layer(in_size, hidden_size),
                    nn.ReLU(),
                    linear_layer(hidden_size, self.n_atoms)
                )
        else:
            self.fc = linear_layer(in_size, act_size * self.n_atoms)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """ Run last layer with the given features 

        Args:
            features (torch.Tensor): Input to the layer

        Returns:
            torch.Tensor: Q values or distributions
        """
        batch_size = features.size()[0]

        # Use fully connected layers according to the extensions
        if self.extensions["dueling"]:
            value_out = self.fc_value(features).view(batch_size, 1, self.n_atoms)

            adv_out = self.fc_advantage(features).view(batch_size, -1, self.n_atoms)
            adv_mean = adv_out.mean(dim=1, keepdim=True)

            out = value_out + adv_out - adv_mean
        else:
            out = self.fc(features).view(batch_size, -1, self.n_atoms)

        # If distributional is active, use softmax to calculate the probabilities.
        # Otherwise, remove last axis.
        if self.extensions["distributional"]:
            out = F.softmax(out, dim=-1).clamp(min=1e-3)
        else:
            out = out.squeeze(-1)

        return out 

    def reset_noise(self) -> None:
        """ Call reset_noise function of all child layers. Only used when 
        "noisy" is active. """
        for module in self.children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class NoisyLinear(torch.nn.Module):
    """ Linear Layer with Noisy parameters. Noise level is set initially and
    kept fixed until "reset_noise" function is called. In training mode,
    noisy layer works stochastically while in evaluation mode it works as a
    standard Linear layer (using mean parameter values).

    Args:
        in_size (int): Input size
        out_size (int): Outout size
        init_std (float): Initial Standard Deviation
    """

    def __init__(self, in_size: int, out_size: int, init_std: float):
        super().__init__()

        self.init_std = init_std
        self.in_feature = in_size
        self.out_feature = out_size
        self.mu_range = (3 / self.in_feature) ** 0.5

        # Define variables
        self.mu_bias = nn.Parameter(torch.FloatTensor(self.out_feature))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(self.out_feature))
        self.register_buffer("epsilon_bias", torch.FloatTensor(self.out_feature)) 

        self.mu_weight = nn.Parameter(torch.FloatTensor(self.out_feature, self.in_feature))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(self.out_feature, self.in_feature))
        self.register_buffer("epsilon_weight", torch.FloatTensor(self.out_feature, self.in_feature))
        
        # Initialize required parameters
        self.parameter_initialization()
        self.reset_noise()

    def parameter_initialization(self) -> None:
        self.mu_weight.data.uniform_(-self.mu_range, self.mu_range)
        self.sigma_weight.data.fill_(self.init_std / math.sqrt(self.in_feature))
        
        self.mu_bias.data.uniform_(-self.mu_range, self.mu_range)
        self.sigma_bias.data.fill_(self.init_std / math.sqrt(self.in_feature))


    def reset_noise(self) -> None:
        """ Reset Noise of the parameters"""
        reset_weight = torch.randn((self.out_feature, self.in_feature)).to(self.mu_weight.device)
        self.epsilon_weight = torch.sign(reset_weight) * torch.sqrt(torch.abs(reset_weight))

        reset_bias = torch.randn((self.out_feature,)).to(self.mu_bias.device)
        self.epsilon_bias = torch.sign(reset_bias) * torch.sqrt(torch.abs(reset_bias))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward function that works stochastically in training mode
        and deterministically in eval mode.

        Args:
            input (torch.Tensor): Layer input

        Returns:
            torch.Tensor: Layer output
        """
        if not self.training:
            return F.linear(input, self.mu_weight, self.mu_bias)
        
        weight = self.mu_weight + self.sigma_weight * self.epsilon_weight
        bias = self.mu_bias + self.sigma_bias * self.epsilon_bias

        return F.linear(input, weight, bias)