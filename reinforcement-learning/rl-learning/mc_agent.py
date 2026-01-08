import random
import time
from collections import defaultdict
from typing import List

import gym

from tabular_agent import TabularAgent


class MCAgent(TabularAgent):
    def __init__(self, n_acts: int):
        super(MCAgent, self).__init__(n_acts)
        self.visited_states = {}

    def one_episode_train(self, env, policy, gamma: float) -> float:
        """ Single episode training function.
        Arguments:
            - env: Mazeworld environment
            - policy: Behaviour policy for the training loop
            - gamma: Discount factor
            - alpha: Exponential decay rate of updates

        Returns:
            episodic reward

        """
        state = env.reset()
        done = False
        steps = []
        G = 0
        iter = 0

        # Collect a steps in one episode
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            G += reward * gamma ** iter
            iter += 1

            steps.append((state, action, reward))
            state = next_state
        
        # Update the q values
        self.update_qvalues(steps, gamma)

        return G

    def update_qvalues(self, steps: List[tuple], gamma: float) -> None:
        visited_state_actions = {}

        # Find the cumulative reward for first occurance of each state-action pair.
        total_reward = 0
        for state, action, reward in reversed(steps):
            total_reward = reward + gamma * total_reward
            visited_state_actions[(state, action)] = total_reward
        
        # Add cumulative reward in the visited state_actions.
        for (state, action), total_reward in visited_state_actions.items():
            if (state, action) not in self.visited_states:
                self.visited_states[(state, action)] = [total_reward]
            else:
                self.visited_states[(state, action)].append(total_reward)
        
        # Q values are the expected cumulative reward of the visited state_actions.
        for (state, action), total_rewards in self.visited_states.items():
            self.q_values[state][action] = sum(total_rewards) / len(total_rewards)



            


