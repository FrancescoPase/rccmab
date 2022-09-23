import pandas as pd
import numpy as np
from copy import copy


class Agent:
    def __init__(self, n_states: int, n_actions: int, n_agents: int):
        self.n_actions = n_actions
        self.n_states = n_states
        self.n_agents = n_agents
        self.history = dict()
        self.rate_history = []
        self.dist_history = []
        self.policy_tx_number = 0
        self.policies = np.ones((n_states, n_actions))
        row_sums = np.sum(self.policies, axis=1)
        self.policies /= row_sums[:, np.newaxis]

        df_fields = ["action", "reward", "iter_num"]
        self.histories = dict()
        for s in range(n_states):
            self.histories[s] = pd.DataFrame(columns=df_fields)
        self.n = 1

    def observe(self, state, action, reward, iter_num):
        self.histories[state] = self.histories[state].append({
             'action': action,
             'reward': reward,
             'iter_num': iter_num},
             ignore_index=True)

    def draw(self, state):
        ...

    def get_history(self):
        return self.histories

    def get_policy(self, state=None):
        if state is None:
            return copy(self.policies)
        return copy(self.policies[state, :])

    def set_policy(self, policies):
        self.policies = copy(policies)

    def add_rate(self, rate):
        self.rate_history.append(rate)

    def add_dist(self, dist):
        self.dist_history.append(dist)
