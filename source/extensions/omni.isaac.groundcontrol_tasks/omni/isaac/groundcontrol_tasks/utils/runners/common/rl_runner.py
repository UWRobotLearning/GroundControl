import torch
from abc import ABC, abstractmethod
from typing import Callable

class RLRunner(ABC):

    @abstractmethod
    def learn(self, num_learning_iterations:int):
        '''Train the RL algorithm'''
        pass

    @abstractmethod
    def reset(self):
        '''Reset the RL runner / optimizer'''
        pass

    @abstractmethod
    def collect_rollouts(self, max_steps:int):
        '''Collect rollouts from the environment'''
        pass
