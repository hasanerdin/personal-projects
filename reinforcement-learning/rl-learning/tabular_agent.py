from collections import defaultdict
import random
import math
from collections import deque
import numpy as np
import time

class TabularAgent:
    """ 
    Based Tabular Agent class that inludes policies and evaluation function
    """

    def __init__(self, n_acts):
        self.q_values = defaultdict(lambda: [0.0]*n_acts)
        self.n_acts = n_acts

    # Return the action with maximum q value.
    # If there are more than one action with maximum q value, return a random one.
    def greedy_policy(self, state: int) -> int:
        state_action = self.q_values[state]
        max_action = max(state_action)
        best_actions = [i for i, v in enumerate(state_action) if v == max_action]
        
        return random.choice(best_actions)
    
    # Return a random action, if randomly generated number is less than epsilon.
    # Otherwise, return the greedy action.
    def e_greedy_policy(self, state: int, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.choice(range(self.n_acts))
        else:
            return self.greedy_policy(state)

    def evaluate(self, env, render=False):
        """ 
        Single episode evaluation of the greedy agent.
        Arguments:
            - env: Warehouse or Mazeworld environemnt
            - render: If true render the environment(default False)
        Return:
            Episodic reward
        """
        state = env.reset()
        G = 0

        terminated = False
        while not terminated:
            action = self.greedy_policy(state)
            state, reward, terminated, _ = env.step(action)

            G += reward

            if render:
                time.sleep(1/10)
                env.render()

        return G