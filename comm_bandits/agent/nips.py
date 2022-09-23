import numpy as np
from comm_bandits.agent.policy_agent import PolicyAgent


class NipsAgent(PolicyAgent):
    def __init__(self, n_states: int, n_actions: int, n_agents: int):
        super().__init__(n_states, n_actions, n_agents)
        self.p_t = 1

    def update_random_prob(self, p):
        self.p_t = p

    def draw(self, state):
        if np.random.random() > self.p_t:
            return super().draw(state)
        else:
            return np.random.choice(np.arange(self.n_actions))
