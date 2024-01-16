import torch
import torch.nn as nn

from copy import deepcopy
import numpy as np
from dataclasses import dataclass, field
from typing import List
import gym

from replay_buffer import ReplayBuffer

#TODO:
# First populate the replay buffer with random actions x
# then do further training

@dataclass
class DQN:
    model: nn.Module 
    env: gym.Env 
    lr: float = 0.0001
    optimizer: str = "Adam"
    update_target_model_every: int = 32
    init_buffer_percentage: float = 0.1
    discount_factor: float = 0.99 # What should this be?

    epsilon: float = 1.0
    epsilon_decay: float = 0.01
    epsilon_min: float = 0.0

    losses: list[float] = field(default_factory=list)
    train_rewards: list[float] = field(default_factory=list)
    val_rewards: list[float] = field(default_factory=list)

    replay_buffer: ReplayBuffer = ReplayBuffer()

    def __str__(self) -> str:
        str = "-"*80 + "\n"
        str += "DQN:\n"
        str += f"lr: {self.lr}\n"
        str += f"optimizer: {self.optimizer}\n"
        str += f"epsilon: {self.epsilon}\n"
        str += f"epsilon_decay: {self.epsilon_decay}\n"
        str += f"epsilon_min: {self.epsilon_min}\n"
        str += f"mean loss: {np.mean(self.losses)}\n"
        str += f"mean reward: {np.mean(self.rewards)}\n"
        str += f"Model: {self.model}\n"
        str += "-"*80 + "\n"
        return str

    def __post_init__(self) -> None:
        self.target_model = deepcopy(self.model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)

        if self.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer} not implemented.")

        # self._populate_replay_buffer() # Maybe here or in train function?
        print("Post init:")
        print(self.__str__())

    # Utility functions
    def _process_observation_to_torch(self, observation: np.ndarray) -> torch.Tensor:
        if isinstance(observation, torch.Tensor):
            # print("observation is already a torch tensor")
            return observation
        if len(np.array(observation, dtype=object).shape) ==3:
            # print("observation is a numpy array of length 3")
            observation = np.transpose(observation, (2, 0, 1))
        else:
            # print("observation is a nested numpy array.")
            observation = np.transpose(observation[0], (2, 0, 1))
        # return torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        return torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    
    def _reset_env(self) -> torch.Tensor:
        state = self.env.reset()
        return self._process_observation_to_torch(state)

    def _update_target_model(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def _decay(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def _epsilon_greedy(self, state: torch.Tensor, eval: bool = False) -> int:
        if np.random.random() <= self.epsilon:
            action =  self.env.action_space.sample()
        else:
            if eval:
                with torch.no_grad():
                    action = torch.argmax(self.model(state.to(self.device))).item()
            else:
                action = torch.argmax(self.model(state.to(self.device))).item()
        self._decay()
        return action
    
    def _take_step(self, action: int) -> tuple[torch.Tensor, float, bool]:
        next_state, reward, done, *_ = self.env.step(action) # Will return a different amount of, in this case unimportant, variables depending on the gym version.
        next_state = self._process_observation_to_torch(next_state)
        return next_state, reward, done
        
    def _populate_replay_buffer(self) -> None:    
        # Populate the replay buffer with random actions 
        # until the replay buffer is filled to the init_buffer_percentage size.
        replay_buffer_init_size = self.replay_buffer.buffer_size * self.init_buffer_percentage 
        state = self._reset_env()
        original_epsilon = self.epsilon 

        while len(self.replay_buffer) < replay_buffer_init_size:
            action = self._epsilon_greedy(state, eval=True)
            next_state, reward, done = self._take_step(action)
            self.replay_buffer.add(state, action, reward, next_state, done)
            if done:
                state = self._reset_env()
            else:
                state = next_state  
        self.epsilon = original_epsilon
    
    # Training utility functions
    def _learn_on_batch(self, batch_size: int = 32) -> None:
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        current_q_values = self.model(states.to(self.device)).gather(1, actions.view(actions.size(0), 1))
        next_q_values = self.target_model(next_states.to(self.device))
        max_q_next = torch.max(next_q_values, 1)[0]
        max_q_next = max_q_next.view(max_q_next.size(0), 1)

        expected_q_values = rewards + (1 - dones) * self.discount_factor * max_q_next

        loss = nn.MSELoss()(current_q_values, expected_q_values)

        # Optimize the model
        loss.backward()
        self.optimizer.step()

    # Training functions
    def play_and_train(self, num_episodes: int = 100) -> None: 
        self._populate_replay_buffer()
        state = self._reset_env()     
        for episode in range(num_episodes):

            # take a step in the environment
            action = self._epsilon_greedy(state, eval=False)
            next_state, reward, done = self._take_step(action)

            self.replay_buffer.add(state, action, reward, next_state, done)

            # perform training on a batch
            self._learn_on_batch()

            if episode % self.update_target_model_every == 0:
                self._update_target_model()

            self.train_rewards.append(reward)
            if done:
                state = self._reset_env()
            else:
                state = next_state

            print(f"Episode: {episode}, reward: {np.mean(self.rewards)}.")

    # play without training
    # Do this seperate
    def play(self, num_episodes: int = 100) -> None:
        state = self._reset_env()
        
        for episode in range(num_episodes):
            action = self._epsilon_greedy(state, eval=True)
            next_state, reward, done = self._take_step(action)
            self.val_rewards.append(reward)
            if done:
                state = self._reset_env()
            else:
                state = next_state
            print(f"Episode: {episode}, reward: {np.mean(self.rewards)}.")
