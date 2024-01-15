
"""
Random search for architecture search
#TODO:
 - make the loops
 - add the architecture imports
 - add the zero cost warmup
 - define in and out size of layers
"""

import inspect
import base_layers
import random
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import gym

import argparse

from replay_buffer import ReplayBuffer
# from agents import DQN
from DQN import DQN

random.seed = 42
class RandomSearch:
    def __init__(self, env, max_size=3):
        self.architecture_dict = {0:"None"}
        self.functions_list = inspect.getmembers(base_layers, inspect.isfunction)
        # get all of the architectures from base_layers.py
        # architecture_dict will be a dictionary of the form:
        # {0:"None", 1:"architecture1", 2:"architecture2", ...}
        functions_list = inspect.getmembers(base_layers, inspect.isfunction)
        for i, func in enumerate(functions_list):
            self.architecture_dict[i+1] = func[1]
        print(self.architecture_dict)
        self.max_size = max_size

        self.env = env

    def sample_arch(self, in_channels, output_channels):
        possible_channels= [16,24,32,64]
        # in_channels = 5
        # out = 5

        # test returning the model from a function
        # model = self.architecture_dict[1](in_channels=3, out_channels=10)

        # sample a random architecture and initiate the layers
        # in_channels = 1
        model = nn.Sequential()
        for i in range(self.max_size):
            # sample a random architecture
            sampled_arch = random.choice(list(self.architecture_dict.values()))
            # initiate the layer
            if sampled_arch == "None" or i == self.max_size-1:
                continue
            # Randomly sample out channels for the layer
            # out_channels = random.choice(possible_channels) if i != self.max_size-1 else output_channels
            out_channels = random.choice(possible_channels)
            layer = sampled_arch(in_channels=in_channels, out_channels=out_channels)
            # add the layer to the model
            model.add_module(str(i), layer)
            # set the in_channels for the next layer
            in_channels = out_channels

        model.add_module(str(i), nn.Flatten())

        input_size = self.calculate_linear_output_size(model)
        model.add_module(str(i+1), nn.Linear(input_size, output_channels))

        # model.add_module(str(i+1), nn.Linear(int((out_channels * 10) / 2) , output_channels))
        model.add_module(str(i+2), nn.ReLU()) # Maybe?
        print("full model: ", model)
        return model
    
    def calculate_linear_output_size(self, model):
        dummy_input = torch.randn(1, self.env.observation_space.shape[2], self.env.observation_space.shape[1], self.env.observation_space.shape[0])
        with torch.no_grad():
            # Forward pass through the model up to the Flatten layer
            output = model(dummy_input)
        return output.numel()

    def search(self, max_models = 1, zero_cost_warmup = 0, train_iterations = 1000):
        num_trained_models = 0
        pool = [] # (reward, losses, model) tuples

        if zero_cost_warmup > 0:
            #TODO
            pass

        channels = self.env.observation_space.shape[2]
        action_size = self.env.action_space.n
        
        for i in range(max_models):
            # sample a random architecture
            model = self.sample_arch(channels, action_size)
            # train the model
            dqn = DQN(model, deepcopy(self.env))
            dqn.play_and_train(train_iterations)
            # add the model to the pool
            pool.append((np.mean(dqn.rewards), np.mean(dqn.losses), model))
            print("model ", i, " trained:")
            print("reward: ", np.mean(dqn.rewards))
            print("losses: ", np.mean(dqn.losses))

        print("pool: ", pool)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--Testing", help="Run 1 of everything to test the code.", action="store_true")
    args = parser.parse_args()
    if args.Testing:
        num_models = 1
        train_iterations = 1
    else:
        # Update later
        num_models = 1
        train_iterations = 1000

    env = gym.make('Freeway-v4')
    RS = RandomSearch(env)
    RS.search(max_models=num_models, train_iterations=train_iterations)

    

    