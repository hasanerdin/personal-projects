from collections import defaultdict
from collections import namedtuple
import random
import math
from typing import List
import numpy as np
import time
from collections import deque

from tabular_agent import TabularAgent


class TabularTDAgent(TabularAgent):
    """
    Base class for Tabular TD Agents for sharing training loop.
    """
    def __init__(self, n_acts: int):
        super(TabularTDAgent, self).__init__(n_acts)

    def train(self, env, policy, args) -> List[float]:
        """ Training loop for tabular td agents.
        Initiate an episodic reward list. At each episode decrease the epsilon
        value exponentially using args.eps_decay_rate within the boundries of
        args.init_eps and args.final_eps. For every "args._evaluate_period"'th
        step call evaluation function and store the returned episodic reward
        to the list.

        Arguments:
            - env: Warehouse environment
            - policy: Behaviour policy to be used in training(not in
            evaluation)
            - args: namedtuple of hyperparameters

        Return:
            - Episodic reward list of evaluations (not the training rewards)

        **Note**: This function will be used in both Sarsa and Q learning.
        """

        reward_list = []
        epsilon = args.init_eps
        for episode_index in range(args.episodes):
            epsilon = max(args.final_eps, epsilon * args.eps_decay_rate)

            self.one_episode_train(env, lambda x: policy(x, epsilon), args)
            
            if (episode_index + 1) % args.evaluate_period == 0:
                reward_list.append(self.evaluate(env))
                print("Episode: {}, reward: {}".format(episode_index + 1, np.mean(reward_list[-5:])))

        return reward_list

    def one_episode_train(self, env, policy, args):
        current_state = env.reset()

        terminated = False
        while not terminated:
            current_action = policy(current_state)
            next_state, reward, terminated, _ = env.step(current_action)
            next_action = policy(next_state)

            transition = (current_state, current_action, reward, next_state, next_action)

            self.update(transition, args.alpha, args.gamma)

            current_state = next_state

            env.render()

class QAgent(TabularTDAgent):
    """ 
    Tabular Q leanring agent. Update rule is based on Q learning.
    """

    def __init__(self, n_acts):
        super(QAgent, self).__init__(n_acts)

    def update(self, transition, alpha, gamma):
        """ Update values of a state-action pair based on the given transition
        and parameters.

        Arguments:
            - transition: 5 tuple of state, action, reward, next_state and
            next_action. "next_action" will not be used in q learning update.
            It is there to be compatible with SARSA update in "train" method.
            - alpha: Exponential decay rate of updates
            - gamma: Discount ratio

        Return:
            temporal diffrence error
        """

        current_state, current_action, reward, next_state, _ = transition

        current_value = self.q_values[current_state][current_action]
        next_value = max(self.q_values[next_state])

        temporal_diff = reward + gamma * next_value - current_value
        updated_value = current_value + alpha * temporal_diff

        self.q_values[current_state][current_action] = updated_value

        return temporal_diff


class SarsaAgent(TabularTDAgent):
    """ 
    Tabular Sarsa agent. Update rule is based on
    SARSA(State Action Reward next_State, next_Action).
    """

    def __init__(self, n_acts):
        super(SarsaAgent, self).__init__(n_acts)

    def update(self, trans, alpha, gamma):
        """ Update values of a state-action pair based on the given transition
        and parameters.

        Arguments:
            - transition: 5 tuple of state, action, reward, next_state and
            next_action.
            - alpha: Exponential decay rate of updates
            - gamma: Discount ratio

        Return:
            temporal diffrence error
        """

        current_state, current_action, reward, next_state, next_action = trans

        current_value = self.q_values[current_state][current_action]
        next_value = self.q_values[next_state][next_action]

        temporal_diff = reward + gamma * next_value - current_value
        updated_value = current_value + alpha * temporal_diff

        self.q_values[current_state][current_action] = updated_value

        return temporal_diff
