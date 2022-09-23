import numpy as np
from scipy.stats import beta, norm
import copy

from comm_bandits.agent.agent import Agent


class BernoulliThompsonAgent(Agent):
    def __init__(self, n_states: int, n_actions: int, n_agents: int):
        super().__init__(n_states, n_actions, n_agents)
        self.alphas = np.ones((n_states, n_actions))
        self.betas = np.ones((n_states, n_actions))
        self.policies = np.ones((n_states, n_actions))
        self.policies /= n_actions
        self.delta_int = 0.1

    def observe(self, state, action, reward, iter_num):
        super().observe(state, action, reward, iter_num)
        self.n += 1
        self.alphas[state][action] += reward
        self.betas[state][action] += (1 - reward)

    def draw(self, state):
        vals = []
        for a in range(self.n_actions):
            vals.append(np.random.beta(self.alphas[state][a], self.betas[state][a]))
        return np.argmax(vals)

    def update_policy(self):
        for state in range(self.n_states):
            rvs = [beta(self.alphas[state, action], self.betas[state, action]) for action in range(self.n_actions)]
            for i in range(self.n_actions):
                prob = 0
                for p in np.arange(self.delta_int, 1, self.delta_int):
                    prod = np.prod([rvs[j].cdf(p) for j in range(self.n_actions) if i != j])
                    prob += rvs[i].pdf(p) * prod * self.delta_int

                self.policies[state, i] = prob

            self.policies[state, :] /= np.sum(self.policies[state, :])


class GaussianThompsonAgent(Agent):
    def __init__(self, n_states: int, n_actions: int, n_agents: int):
        super().__init__(n_states, n_actions, n_agents)
        self.tau = np.ones((self.n_states, self.n_actions)) * 0.0001  # the posterior precision
        self.mu = np.ones((self.n_states, self.n_actions))  # the posterior mean
        self.means = np.zeros((self.n_states, self.n_actions))
        self.policies = np.ones((n_states, n_actions))
        self.policies /= n_actions
        self.delta_int = 0.1

    def observe(self, state, action, reward, iter_num, update_weights=True):
        super().observe(state, action, reward, iter_num)
        self.n += 1
        self.means[state, action] = (1 - 1.0 / self.n) * self.means[state][action] + (1.0 / self.n) * reward
        self.mu[state, action] = ((self.tau[state][action] * self.mu[state][action]) +
                                  (self.n * self.means[state][action])) / (self.tau[state][action] + self.n)
        self.tau[state][action] += 1
        for s in range(self.n_states):
            self.update_policy(s)

    def draw(self, state):
        return np.argmax(np.random.randn(self.n_actions) / np.sqrt(self.tau[state]) + self.means[state])

    def update_policy(self, state):
        rvs = [norm(self.means[state, action], self.mu[state, action]) for action in range(self.n_actions)]
        for i in range(self.n_actions):
            prob = 0
            for mu in np.arange(-10, 10, self.delta_int):
                prod = np.prod([rvs[j].cdf(mu) for j in range(self.n_actions) if i != j])
                prob += rvs[i].pdf(mu) * prod * self.delta_int

            self.policies[state, i] = prob

        self.policies[state, :] /= np.sum(self.policies[state, :])


class RateBernoulliThompson(BernoulliThompsonAgent):

    def __init__(self, n_states: int, n_actions: int, n_agents: int):
        super(RateBernoulliThompson, self).__init__(n_states, n_actions, n_agents)
        self.sampling_policies = None

    def draw(self, state):
        return np.random.choice(np.arange(self.n_actions), size=1, p=self.sampling_policies[state, :])[0]

    def set_sampling_policy(self, policy):
        self.sampling_policies = copy.copy(policy)

