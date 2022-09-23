from comm_bandits.agent.agent import Agent
import numpy as np


class PolicyAgent(Agent):
    def __init__(self, n_states: int, n_actions: int, n_agents: int):
        super().__init__(n_states, n_actions, n_agents)
        self.policies = np.ones((n_states, n_actions))
        self.policies /= self.n_actions

    def draw(self, state):
        return np.random.choice(np.arange(self.n_actions), p=self.policies[state, :])


class OptimalAgent(PolicyAgent):
    def __init__(self, n_states: int, n_actions: int, n_agents: int, optimal_policy):
        super().__init__(n_states, n_actions, n_agents)
        self.optimal_policy = optimal_policy

    def draw(self, state):
        return self.optimal_policy[state]

