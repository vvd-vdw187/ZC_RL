
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
from GraSP import GraSP
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

        model.add_module(str(i+2), nn.ReLU()) # Maybe?
        return model
    
    def calculate_linear_output_size(self, model):
        dummy_input = torch.randn(1, self.env.observation_space.shape[2], self.env.observation_space.shape[1], self.env.observation_space.shape[0])
        with torch.no_grad():
            # Forward pass through the model up to the Flatten layer
            output = model(dummy_input)
        return output.numel()

    def search(self, max_models = 1, zero_cost_warmup = 0, train_iterations = 1000):
        num_trained_models = 0
        pool = [] # (i, reward, losses, model) tuples
        zero_cost_pool = [] # (i, grasp_metric, model) tuples
        channels = self.env.observation_space.shape[2]
        action_size = self.env.action_space.n
        if zero_cost_warmup > 0:
            for i in range(zero_cost_warmup):
                # sample a random architecture
                model = self.sample_arch(channels, action_size)
                dqn = DQN(model, deepcopy(self.env))
                mask = GraSP(dqn)
                i = 0
                for layer in dqn.model.modules():
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        with torch.no_grad():
                            layer.weight.data *= mask[i]
                        i+=1
                dqn.play_and_train(train_iterations)
                dqn.play()
        #     # add the model to the pool
                print("model ", i, " trained:")
                print("train reward: ", np.mean(dqn.train_rewards))
                print("val reward: ", np.mean(dqn.val_rewards))
                zero_cost_pool.append((i,np.mean(dqn.train_rewards), np.mean(dqn.val_rewards), model))
                # zero_cost_pool.append((i, model))
        # Print every row of the tensor in the pool
        # torch.set_printoptions(profile="full")
        # [[print(row) for row in item[1]] for item in zero_cost_pool]


        # for i in range(max_models):
        #     # sample a random architecture
        #     model = self.sample_arch(channels, action_size)
        #     # train the model
        #     dqn = DQN(model, deepcopy(self.env))
        #     dqn.play_and_train(train_iterations)
        #     dqn.play()
        #     # add the model to the pool
        #     pool.append((i,np.mean(dqn.train_rewards), np.mean(dqn.val_rewards), model))
            
        #     print("model ", i, " trained:")
        #     print("train reward: ", np.mean(dqn.train_rewards))
        #     print("val reward: ", np.mean(dqn.val_rewards))
        #     print("losses: ", np.mean(dqn.losses))

        # Dont print the full models for clarity
        print("pool: ", [item[:3] for item in pool])
        print("zero cost pool: ", [item[:3] for item in zero_cost_pool])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--Testing", help="Run 1 of everything to test the code.", action="store_true")
    args = parser.parse_args()
    if args.Testing:
        num_models = 10
        train_iterations = 1
        max_model_size = 5
        zero_cost_warmup = 10
    else:
        # Update later
        num_models = 5
        train_iterations = 1000
        max_model_size = 5
        zero_cost_warmup = 1

    env = gym.make('Freeway-v4')
    RS = RandomSearch(env, max_size=max_model_size)
    RS.search(max_models=num_models, zero_cost_warmup=zero_cost_warmup, train_iterations=train_iterations)

    

    