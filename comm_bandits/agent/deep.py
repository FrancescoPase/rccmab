import numpy as np
import pandas as pd
import torch

from comm_bandits.agent.agent import Agent


class DeepAgent(Agent):
    def __init__(self, n_states, n_actions, n_agents):
        super().__init__(n_states, n_actions, n_agents, state)
