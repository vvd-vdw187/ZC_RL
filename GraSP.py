import torch 
import torch.nn as nn
import torch.autograd as autograd
import gym
import random
from DQN import DQN

#https://github.dev/SamsungLabs/zero-cost-nas
def GraSP(agent: DQN, iters: int=10, data_loop_iters: int=10, training_batch_size: int=64):
    # Make baseclass for all agents
    #TODO implement self.device
    #TODO implement the split data loops as samples from the replay buffer

    # Get all the weights from the agent.model
    weights = []
    for layer in agent.model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)

    # Get the gradients of the loss w.r.t the weights
    agent.model.zero_grad()
    # So far only dqn is implemented and it uses this, maybe change in the future.
    agent.populate_replay_buffer()
    random.seed = 42
    for _ in range(data_loop_iters):
        batch = agent.replay_buffer.sample(training_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        grad_w = None
        for _ in range(iters):
            # In principle this should be the differentiable loss function
            # #TODO use Q_loss instead
            loss = agent.loss(states, actions, rewards, next_states, dones)
            grad_w_p = autograd.grad(loss, weights)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

    # Another split data loop
    random.seed = 42
    for _ in range(data_loop_iters):
        batch = agent.replay_buffer.sample(training_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        grad_w = None

        loss = agent.loss(states, actions, rewards, next_states, dones)
        grad_w_p = autograd.grad(loss, weights, create_graph=True, allow_unused=True)

        z, count = 0, 0
        for layer in agent.model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if grad_w_p[count] is not None:
                    z += (grad_w_p[count] * layer.weight).sum()
                count += 1
        z.backward()

        # Here the mode is used but not sure if necessary
    def grasp_metric(layer):
        if layer.weight.grad is not None:
            return -layer.weight.data * layer.weight.grad
        else:
            return torch.zeros_like(layer.weight)

    grads = []
    for layer in agent.model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads.append(grasp_metric(layer))

    return grads



    
