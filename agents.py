import torch
import torch.nn as nn

from copy import deepcopy
import numpy as np

from replay_buffer import ReplayBuffer

# https://github.dev/KarlXing/RLCodebase
class A2C:
    def __init__(self, model, lr=0.001, optimizer="Adam"):
        self.model = model
        self.lr = lr
        self.optimizer = optimizer

    def inference(self, state):
        action, action_log_prob, value, entropy = self.model(state)

    def learn(self, state, action, returns, advantages):
        action, action_log_prob, value, entropy = self.model(state)
        # Calculate the loss
        actor_loss = -(action_log_prob * advantages).mean()
        critic_loss = (returns - value).pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy.mean()

        # Perform the backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Update the parameters
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy.mean().item()
    

class DQN:
    def __init__(self, model, lr=0.001, optimizer="Adam"):
        self.model = model
        self.target_model = deepcopy(model)
        self.lr = lr
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"Optimizer {optimizer} not implemented.")
        
        print("DQN:")
        print("model in DQN:", self.model)

        # Set the exploration rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.0

        self.env = None
        self.losses = []
        self.rewards = []

        self.replay_buffer = ReplayBuffer()

    def process_observation_to_torch(self, observation):
        if isinstance(observation, torch.Tensor):
            return observation
        if len(np.array(observation, dtype=object).shape) ==3:
            observation = np.transpose(observation, (2, 0, 1))
        else:
            observation = np.transpose(observation[0], (2, 0, 1))
        return torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

    def inference(self, state):
        return self.model(state) # returns action
    
    def loss(self, action, old_state, new_state, reward, done):
        # TODO make sure to do a step in the environment before calling this
        # TODO transform this to work in batches.

        # Original code:
        # weights = batch['weights'] if 'weights' in batch else torch.ones_like(reward).unsqueeze(-1)
        weights = torch.ones_like(torch.tensor(reward)).unsqueeze(-1) 
        # weights = torch.ones_like(reward)

        # update q net
        # old_state = self.process_observation_to_torch(old_state) # Old state is already a torch tensor
        new_state = self.process_observation_to_torch(new_state)
        with torch.no_grad():
            target_q = self.target_model(new_state).max(1)[0] * (1 - done) + reward
        # q = self.model(old_state).gather(1, action.unsqueeze(1)).squeeze(1) # for batch updates
        q_int = self.model(old_state)[0]
        # print("q_int",q_int)
        # print("q_int_shape:", q_int.shape)
        # print("action",action)
        q = q_int[action]
        q = self.model(old_state)[0][action]
        q_loss = (((q - target_q)**2)*weights).mean()

        return q_loss
    
    # Functions used for Training

    def set_env(self, env):
        self.env = env
        self.state = self.env.reset()
        self.state = self.process_observation_to_torch(self.state)

    def reset_env(self):
        assert self.env is not None, "No environment set for the agent."
        self.state = self.env.reset()
        self.state = self.process_observation_to_torch(self.state)
        print("Reset environment")

    def get_env_state(self):
        assert self.env is not None, "No environment set for the agent."
        return self.env.observation_space
    
    def learn_on_batch(self, batch):
        state, action, next_state, reward, done = batch
        action = action.long() # what does this do and does it even work?

        # Calculate the loss
        loss = self.loss(action, state, next_state, reward, done)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def step(self):
        # print("state at step:", self.state.shape)
        # Perform a forward pass in the environment and update the targetmodel and model
        # Perform the epsilon greedy policy
        # state = self.get_env_state()
        # state = self.process_observation_to_torch(state)  
        self.state = self.process_observation_to_torch(self.state)

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            with torch.no_grad():
                action = self.inference(self.state)
                action = action.argmax().item()
        # Decay the epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Get the step from the environment
        next_state, reward, done, _, info = self.env.step(action)
        loss = self.loss(action, self.state, next_state, reward, done)
        loss.backward()
        self.optimizer.step()

        # Updates
        self.target_model.load_state_dict(self.model.state_dict())
        self.losses.append(loss.item())
        self.rewards.append(reward)
        self.state = self.process_observation_to_torch(next_state)

        return next_state, reward, done, info
    
    def play_and_train(self, turns=200):
        self.reset_env()
        for turn in range(turns):
            _, reward, done, _ = self.step()
            print(turn, reward)
            if done:

                self.state = self.env.reset()