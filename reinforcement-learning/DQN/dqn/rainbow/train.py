from typing import Generator
from collections import namedtuple, deque
from functools import reduce
import argparse
import gym
import torch

from .model import RainBow
from dqn.common import linear_annealing, exponential_annealing, PrintWriter
from dqn.dqn.train import Trainer as BaseTrainer


class Trainer(BaseTrainer):
    """ Training class that organize evaluation, update, and transition
    gathering.
        Arguments:
            - args: Parser arguments
            - agent: RL agent object
            - opt: Optimizer that optimizes agent's parameters
            - env: Gym environment
    """

    def __init__(self, args: argparse.Namespace, agent: RainBow, opt: torch.optim.Optimizer, env: gym.Env):
        """ Training class that organize evaluation, update, and transition gathering.

        Args:
            args (argparse.Namespace): CL arguments
            agent (RainBow): RainBow Agent
            opt (torch.optim.Optimizer): Optimizer for agents parameters
            env (gym.Env): Environment
        """
        super().__init__(args, agent, opt, env)
        # beta = 1 - prioritized_beta
        self.prioritized_beta = linear_annealing(
            init_value=1 - args.beta_init,
            min_value=0,
            decay_range=args.n_iterations
        )

    def update(self, iteration: int) -> None:
        """ One step updating function. Update the agent in training mode.
        - clip gradient if "clip_grad" is given in args.
        - keep track of td loss. Append td loss to "self.td_loss" list
        - Update target network.
        If the prioritized buffer is active:
            - Use the weighted average of the loss where the weights are 
            returned by the prioritized buffer
            - Update priorities of the sampled transitions
        If noisy-net is active:
            - reset noise for valuenet and targetnet
        Check for the training index "iteration" to start the update.

        Args:
            iteration (int): Training iteration
        """
        self.agent.train()

        if iteration >= self.args.start_update:
            if self.agent.extensions["prioritized"]:
                sample_batch, indices, weights = self.agent.buffer.sample(self.args.batch_size, 
                                                                          next(self.prioritized_beta))
            else:
                sample_batch = self.agent.buffer.sample(self.args.batch_size)
            
            # Convert sample_batch to torch from numpy array
            sample_batch = self.agent.batch_to_torch(sample_batch, self.args.device)
            
            # Compute td error
            td_error = self.agent.loss(sample_batch, self.args.gamma)

            # Calculate mean of the loss. If prioritizing is active, weight the loss.
            loss = td_error
            if self.agent.extensions["prioritized"]:
                loss *= torch.from_numpy(weights).to(loss.device).view(-1, 1)
            loss = torch.mean(loss)

            # Save loss
            self.td_loss.append(loss.item())

            # Empty gradients
            self.opt.zero_grad()

            # Backpropagate
            loss.backward()
            if self.args.clip_grad:
                # Clip the error term to be between -1 and 1
                for param in self.agent.valuenet.parameters():
                    param.grad.data.clamp_(-1, 1)
            
            # Optimize
            self.opt.step()

            # Update priorities
            if self.agent.extensions["prioritized"]:
                self.agent.buffer.update_priority(indices, td_error.detach().cpu().numpy())

            # Reset noise parameters
            if self.agent.extensions["noisy"]:
                self.agent.valuenet.reset_noise()
                self.agent.targetnet.reset_noise()

            # Update target network
            if iteration % self.args.target_update_period == 0:
                self.agent.update_target()

    def __iter__(self) -> Generator[RainBow.Transition, None, None]:
        """ n-step transition generator. Yield a transition with
         n-step look ahead. Use the greedy policy if noisy network 
         extension is activate.

        Yields:
            Generator[RainBow.Transition, None, None]: Transition of
            (s_t, a_t, \sum_{j=t}^{t+n}(\gamma^{j-t} r_j), s_{t+n}, done)
        """
        # Start with resetting environment and epsiodic reward
        current_state = self.env.reset()
        episodic_reward = 0
        
        for _ in range(self.args.n_iterations):
            eps = next(self.epsilon)

            # Initiate state of step and total step reward            
            step_state = current_state
            total_step_reward = 0

            # Turn n_steps time to find reward and t+n step from t to t+n
            for step in range(self.args.n_steps):
                torch_state = self.agent.state_to_torch(step_state, self.args.device).unsqueeze(0)

                # If network is already noisy, use directly greedy policy
                # Otherwise, use epsilon greedy policy
                if self.agent.extensions["noisy"]:
                    step_action = self.agent.greedy_policy(torch_state)
                else:
                    step_action = self.agent.e_greedy_policy(torch_state, eps)
                
                # Keep first action
                if step == 0:
                    current_action = step_action

                # Get step results from environment and calculate reward
                next_state, reward, done, _ = self.env.step(step_action)
                total_step_reward += self.args.gamma ** step * reward

                step_state = next_state
            
            # Add total n_step reward to episodic reward
            episodic_reward += total_step_reward

            # yield transition
            yield RainBow.Transition(current_state, current_action, total_step_reward, next_state, done)

            # Set next_state to current state
            current_state = next_state
            
            # If done, execute required operations
            if done:
                self.train_rewards.append(episodic_reward)
                current_state = self.env.reset()
                episodic_reward = 0
