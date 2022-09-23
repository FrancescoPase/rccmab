import numpy as np
from copy import copy
import pandas as pd

from comm_bandits.agent.agent import Agent


class Exp3Agent(Agent):
    """
        Implement the EXP3 agent at the server side
    """

    def __init__(self, n_states, n_actions, n_agents, gamma=0):
        super().__init__(n_states, n_actions, n_agents)
        self.gamma = gamma
        self.weights = np.ones((n_states, n_actions))

    def observe(self, state, action, reward, iter_num, update_weights=True):
        super().observe(state, action, reward, iter_num, update_weights)
        if update_weights:
            self.update_weights(state, action, reward)

    def draw(self, state):
        arm = np.random.choice(np.arange(self.n_actions), p=self.policies[state, :], replace=False)
        return arm

    def update_weights(self, state, action, reward):

        estimated_reward = 1.0 * reward / self.policies[state, action]
        self.weights[state, action] *= np.exp(estimated_reward * self.gamma / self.n_actions)

        self.policies[state, :] = self.weights[state, :] / np.sum(self.weights[state, :])
        self.policies[state, :] *= (1.0 - self.gamma)
        self.policies[state, :] += (self.gamma / self.n_actions)

