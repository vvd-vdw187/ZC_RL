import numpy as np
from dataclasses import dataclass, field
from collections import deque, namedtuple
import random
from typing import List
import torch

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

@dataclass
class ReplayBuffer:
    buffer_size: int = 5000
    buffer: deque = field(init=False)

    def __post_init__(self):
        self.buffer = deque(maxlen=self.buffer_size)

    def __len__(self) -> int:
        return len(self.buffer)
    
    def __full__(self) -> bool:
        return len(self.buffer) == self.buffer_size

    def add(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool) -> None:
        # if the ReplayBuffer is full, pop the oldest experience, handled by the deque method
        # if self.__full__():
        #     self.buffer.popleft()
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int = 32) -> List[Experience]:
        return random.sample(self.buffer, k=batch_size)
 