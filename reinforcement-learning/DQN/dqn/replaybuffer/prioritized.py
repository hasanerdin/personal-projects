from lib2to3.pytree import Base
from typing import Tuple
import numpy as np

from dqn.replaybuffer.uniform import BaseBuffer


class PriorityBuffer(BaseBuffer):
    """ Prioritized Replay Buffer. 

    Args:
        capacity (int): Maximum size of the buffer
        state_shape (Tuple): Shape of a single the state
        state_dtype (np.dtype): Data type of the states
        alpha (float): Exponent of td
        epsilon (float, optional): Minimum absolute td error. If |td| = 0.2,
        we store it as |td| + epsilon. Defaults to 0.1.
    """

    def __init__(self, capacity: int, state_shape: Tuple[int], state_dtype: np.dtype,
                 alpha: float, epsilon: float = 0.1):
        super().__init__(capacity, state_shape, state_dtype)
        self.abs_td_errors = np.zeros(capacity, dtype=np.float32)
        self.epsilon = epsilon
        self.alpha = alpha
        self.write_index = 0  # pointing to the next writing index
        self.size = 0
        self.max_abs_td = epsilon  # Maximum = epsilon at the beginning

    def push(self, transition: BaseBuffer.Transition) -> None:
        """ Push a transition object (with single elements) to the buffer.
        Transitions are pushed with the current maximum absolute td.
        Remember to set <write_index> and <size> attributes.

        Args:
            transition (BaseBuffer.Transition): transition to push to buffer
        """
        for buf, trns in zip(self.buffer, transition):
            buf[self.write_index] = trns

        # If max_abs_td is deleted, update new max_abs_td
        if self.abs_td_errors[self.write_index] == self.max_abs_td:
            self.abs_td_errors[self.write_index] = 0
            self.max_abs_td = max(self.abs_td_errors)

        self.abs_td_errors[self.write_index] = self.max_abs_td
        
        self.write_index = (self.write_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> Tuple[BaseBuffer.Transition, np.ndarray, np.ndarray]:
        """ Sample a batch of transitions based on priorities.

        Args:
            batch_size (int): Batch size
            beta (float): Exponent of IS weights

        Returns:
            Tuple[BaseBuffer.Transition, np.ndarray, np.ndarray]:
                - Transition object of batch of samples
                - Indices of the samples (used for priority update)
                - Importance sampling weights
        """
        if batch_size > self.size:
            return (None, None, None)
        
        # Calculate probabilities
        priorities = self.abs_td_errors[:self.size] ** self.alpha
        prob = priorities / sum(priorities)
        
        # Sample according to the probability 
        indices = np.random.choice(self.size, size=batch_size, p=prob, replace=False)
        transitions = BaseBuffer.Transition(*(x[indices] for x in self.buffer))

        # Calculate importance sampling weights
        batch_weights = (self.size * prob[indices]) ** (-beta)
        batch_weights /= max(batch_weights)

        return (transitions, indices, batch_weights)

    def update_priority(self, indices: np.ndarray, td_values: np.ndarray) -> None:
        """ Update the priority td_values of given indices (returned from sample).
        Update max_abs_td value if there exists a higher absolute td.

        Args:
            indices (np.ndarray): Indices of the samples
            td_values (np.ndarray): New td values
        """
        priorities = abs(td_values) + self.epsilon
        self.abs_td_errors[indices] += priorities.reshape(-1, )

        self.max_abs_td = max(self.abs_td_errors)  
