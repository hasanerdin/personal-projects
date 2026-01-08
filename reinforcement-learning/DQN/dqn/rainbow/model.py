from typing import Dict, Any
from copy import deepcopy
from collections import namedtuple
import numpy as np
import torch

from dqn.replaybuffer.uniform import UniformBuffer, BaseBuffer
from dqn.replaybuffer.prioritized import PriorityBuffer
from dqn.dqn.model import DQN


class RainBow(DQN):
    """ Rainbow DQN agent with selectable extensions.

    Args:
        valuenet (torch.nn.Module): Q network
        nact (int): Number of actions
        extensions (Dict[str, Any]): Extension information
    """

    def __init__(self, valuenet: torch.nn.Module, nact: int, extensions: Dict[str, Any], *buffer_args):
        self.extensions = extensions
        if extensions["prioritized"]:
            buffer = PriorityBuffer(
                *buffer_args,
                alpha=extensions["prioritized"]["alpha"]
            )
        else:
            buffer = UniformBuffer(*buffer_args)

        super().__init__(valuenet, nact, buffer)
        
        # Defining distribtional ranges
        if self.extensions["distributional"]:
            vmin = self.extensions["distributional"]["vmin"]
            vmax = self.extensions["distributional"]["vmax"]
            n_atoms = self.extensions["distributional"]["natoms"]

            self.supports = torch.linspace(vmin, vmax, n_atoms)

    def greedy_policy(self, state: torch.Tensor, *args) -> int:
        """ The greedy policy that changes its behavior depending on the
        value of the "distributional" option in the extensions dictionary. If
        distributional values are activated, use expected_value method.

        Args:
            state (torch.Tensor): Torch state

        Returns:
            int: action
        """
        if self.extensions["distributional"]:
            value_dist = self.valuenet(state)
            return self.expected_value(value_dist).argmax().item()
        return super().greedy_policy(state)

    def loss(self, batch: BaseBuffer.Transition, gamma: float) -> torch.Tensor:
        """ Loss method that switches loss function depending on the value
        of the "distributional" option in extensions dictionary. 

        Args:
            batch (BaseBuffer.Transition): Batch of Torch Transitions
            gamma (float): Discount Factor

        Returns:
            torch.Tensor: Value loss
        """
        if self.extensions["distributional"]:
            return self.distributional_loss(batch, gamma)
        return self.vanilla_loss(batch, gamma)

    def vanilla_loss(self, batch: BaseBuffer.Transition, gamma: float) -> torch.Tensor:
        """ MSE (L2, L1, or smooth L1) TD loss with double DQN extension in
        mind. Different than DQN loss, we keep the batch axis to make this
        compatible with the prioritized buffer. Note that: For target value calculation 
        "_next_action_network" should be used. Set target network and action network to
        eval mode while calculating target value if the networks are noisy.

        Args:
            batch (BaseBuffer.Transition): Batch of Torch Transitions
            gamma (float): Discount Factor

        Returns:
            torch.Tensor: Value loss
        """
        states, actions, rewards, next_states, dones = batch

        # Calculate qvalues of main network
        current_qvalues = self.valuenet(states).gather(1, actions)

        with torch.no_grad():
            # Find the next actions according to next action network
            next_actions = self._next_action_network(next_states).argmax(1).unsqueeze(-1)
            # Find the next qvalues of the target network by using next_actions
            next_qvalues = self.targetnet(next_states).gather(1, next_actions)
            # Calculate the target reward
            target_qvalues = rewards + gamma * (1 - dones) * next_qvalues 

        # Calculate TD error
        td_error = torch.nn.functional.smooth_l1_loss(current_qvalues, target_qvalues, reduction="none")

        return td_error

    def expected_value(self, values: torch.Tensor) -> torch.Tensor:
        """ Return the expected state-action values. Used when distributional
            value is activated.

        Args:
            values (torch.Tensor): Value tensor of distributional output (B, A, Z). B,
                A, Z denote batch, action, and atom respectively.

        Returns:
            torch.Tensor: the expected value of shape (B, A)
        """
        # return sum of values multiplied by the distribution
        return torch.sum(values * self.supports.to(values.device), dim=2)

    def distributional_loss(self, batch: BaseBuffer.Transition, gamma: float) -> torch.Tensor:
        """ Distributional RL TD loss with KL divergence (with Double
        Q-learning via "_next_action_network" at target value calculation).
        Keep the batch axis. Set noisy network to evaluation mode while calculating
        target values.

        Args:
            batch (BaseBuffer.Transition): Batch of Torch Transitions
            gamma (float): Discount Factor

        Returns:
            torch.Tensor: Value loss
        """
        states, actions, rewards, next_states, dones = batch
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)
        batch_size = len(states)

        with torch.no_grad():
            # Find next distribtion by using next_action_network and target network
            next_target_actions = self._next_action_network(next_states)
            next_target_actions = self.expected_value(next_target_actions).argmax(1)
            next_dist = self.targetnet(next_states)[range(batch_size), next_target_actions]

            min_value, max_value, n_atoms = self.extensions["distributional"].values()
            delta_z = (max_value - min_value) / (n_atoms - 1)

            # Calculate projection distribution
            t_z = rewards + gamma * (1 - dones) * self.supports
            t_z = t_z.clamp(min=min_value, max=max_value)
            b = (t_z - min_value) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (batch_size - 1) * n_atoms, batch_size
                ).long().unsqueeze(1).expand(batch_size, n_atoms)
            )

            proj_dist = torch.zeros(next_dist.size())

            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        # Calculate logarithmic distribution
        dist = self.valuenet(states)[range(batch_size), actions.squeeze(-1)]
        log_p = torch.log(dist)
        
        # Calculate TD error
        td_loss = -(proj_dist * log_p).sum(1)
        return td_loss

    @property
    def _next_action_network(self) -> torch.nn.Module:
        """ Return the network used for the next action calculation (Used for
        Double Q-learning)

        Returns:
            torch.nn.Module: Q network to find target/next action
        """
        if self.extensions["double"]:
            return self.valuenet
        return self.targetnet

