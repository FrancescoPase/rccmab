import numpy as np
import sys
import os
from copy import copy
import pickle
from scipy.stats import binom, bernoulli, norm, beta
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from comm_bandits.it.kl_minimization import joint_entropy, blahut_arimoto_kl_d_r, blahut_arimoto_kl_r_d
from comm_bandits.cluster.cluster import state_reduction
from comm_bandits.agent.thompson import BernoulliThompsonAgent, RateBernoulliThompson
from comm_bandits.agent.policy_agent import OptimalAgent
from comm_bandits.it.reverse_kl_minimization import blahut_arimoto_d_r

# Exp Data
# np.random.seed(6)

eps_0 = 0
eps_1 = 0.3
max_distortion = 0

eps = 0.01

n_states = 32
n_actions = 32
n_agents = 50
group_size = 8


number_of_iter = 400
for sim_num in range(1):
    print("-------------- Simulation {} ----------------------".format(sim_num))
    experiment = 'lim_training_uniform_fs_dist_0_{}_a_{}_s_{}'.format(n_actions, n_states, group_size)
    path_results = 'results/{}'.format(experiment)
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    xs = np.arange(0, n_actions)
    fs = np.ones(n_states)

    fs /= np.sum(fs)
    states = np.arange(n_states)

    # ****************************           Generate Environment       ********************************

    rew_distributions = dict()

    for s in range(n_states):
        rew_distributions[s] = dict()
        for a in range(n_actions):
            if int(np.floor(s / group_size)) == a:
                p = 0.8
            else:
                p = np.random.random() * 0.65

            rv = bernoulli(p)
            rew_distributions[s][a] = rv

    group_num = 0
    optimal_policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        if s < (group_num + 1) * group_size:
            optimal_policy[s] = group_num
        else:
            group_num += 1
            optimal_policy[s] = group_num

    # rate_max = np.log2(group_num + 1)
    rate_max = 1

    # ****************************           Generate Agents       ********************************

    perfect_agent = BernoulliThompsonAgent(n_states, n_actions, n_agents)
    comm_agent = RateBernoulliThompson(n_states, n_actions, n_agents)
    comm_rev_agent = RateBernoulliThompson(n_states, n_actions, n_agents)
    cluster_agent = RateBernoulliThompson(n_states, n_actions, n_agents)
    cluster_rev_agent = RateBernoulliThompson(n_states, n_actions, n_agents)
    optimal_agent = OptimalAgent(n_states, n_actions, n_agents, optimal_policy)
    comm_fixed_agent = RateBernoulliThompson(n_states, n_actions, n_agents)
    comm_marginal_agent = RateBernoulliThompson(n_states, n_actions, n_agents)

    # ****************************           Run Simulation       ********************************

    # Save metrics
    comm_dist = []
    cls_dist = []
    cls_l1_dist = []
    comm_l1 = []
    comm_fixed_dist = []
    comm_rev_dist = []
    clst_rev_dist = []

    mutual_info = []
    avg_perfect_r = []
    target_kl_list = []
    target_rev_list = []

    number_policy_update = 1

    # Policy used to construct the clustered one
    last_perfect_policy = copy(cluster_agent.get_policy())
    last_perfect_policy_rev = copy(cluster_rev_agent.get_policy())
    cluster_agent.set_sampling_policy(copy(cluster_agent.get_policy()))
    cluster_rev_agent.set_sampling_policy(copy(cluster_rev_agent.get_policy()))

    print("*****************    Simulation started   **********************")
    for t in range(1, number_of_iter+1):

        # Insert here the available rate scheme...
        # Compute the rate of the perfect_agent
        U_0, U_marginal_0, perfect_dist, perfect_r, _ = blahut_arimoto_kl_r_d(
            P=perfect_agent.get_policy(), f_x=fs, distortion=0, delta=0.01, eps=eps
        )

        # Compute policy and rate of comm_agent
        U_1, U_marginal_1, comm_dist, comm_r, _ = blahut_arimoto_kl_d_r(
            P=comm_agent.get_policy(), f_x=fs, rate=rate_max, eps=eps
        )

        # Compute policy and distortion of the rev kl agent
        U_rev, U_rev_marginal_, comm_rev_dist, comm_rev_r, _ = blahut_arimoto_d_r(
            P=comm_rev_agent.get_policy(), f_x=fs, rate=rate_max, eps=eps
        )

        # Update values

        # Impose here the clusters' schemes rate
        # b = np.ceil(perfect_r)
        b = rate_max

        # Keep track of last sent and current policies
        target_kl = joint_entropy(last_perfect_policy, cluster_agent.get_policy(), f_x=fs)
        target_kl_list.append(target_kl)

        # In this case update it at each round, otherwise compare target_kl and actual kl divergence.
        if True:

            last_perfect_policy = cluster_agent.get_policy()
            number_policy_update += 1
            # check copy

            centroids, association_rule, state_cls = state_reduction(cluster_agent.get_policy(), fs, b,
                                                                     rule='kl', num_rounds=20)
            U_cls = np.zeros((n_states, n_actions))
            for s in range(n_states):
                U_cls[s, :] = centroids[state_cls[s]]
            cluster_agent.set_sampling_policy(U_cls)

        # Keep track of last sent and current policies in rev
        target_kl = joint_entropy(last_perfect_policy_rev, cluster_rev_agent.get_policy(), f_x=fs)
        target_rev_list.append(target_kl)

        # In this case update it at each round, otherwise compare target_kl and actual kl divergence.
        if True:
            last_perfect_policy_rev = cluster_rev_agent.get_policy()
            number_policy_update += 1
            centroids, association_rule, state_cls = state_reduction(cluster_rev_agent.get_policy(), fs, b,
                                                                     rule='reverse_kl', num_rounds=20)
            U_cls_rev = np.zeros((n_states, n_actions))
            for s in range(n_states):
                U_cls_rev[s, :] = centroids[state_cls[s]]
            cluster_rev_agent.set_sampling_policy(U_cls_rev)

        # Set policies
        comm_agent.set_sampling_policy(U_1)
        comm_rev_agent.set_sampling_policy(U_rev)

        if perfect_r <= rate_max:
            comm_fixed_agent.set_sampling_policy(copy(comm_fixed_agent.get_policy()))
            comm_marginal_agent.set_sampling_policy(copy(comm_marginal_agent.get_policy()))
        else:
            comm_fixed_agent.set_sampling_policy(np.ones((n_states, n_actions))/n_actions)
            Q = np.tile(np.dot(fs, comm_marginal_agent.get_policy()), [n_states, 1])
            comm_fixed_agent.set_sampling_policy(Q)

        comm_agent.add_dist(comm_dist)
        cluster_agent.add_dist(joint_entropy(cluster_agent.get_policy(), perfect_agent.get_policy(), fs))
        cluster_rev_agent.add_dist(joint_entropy(perfect_agent.get_policy(), cluster_rev_agent.get_policy(), fs))

        s_t = np.random.choice(states, n_agents, p=fs)

        perfect_a_t = []
        perfect_r_t = []

        optimal_a_t = []
        optimal_r_t = []

        cls_a_t = []
        cls_r_t = []

        cls_l1_a_t = []
        cls_l1_r_t = []

        cls_rev_a_t = []
        cls_rev_r_t = []

        comm_a_t = []
        comm_r_t = []

        comm_rev_a_t = []
        comm_rev_r_t = []

        comm_marginal_a_t = []
        comm_marginal_r_t = []

        comm_fixed_a_t = []
        comm_fixed_r_t = []

        marginal_a_t = []
        marginal_r_t = []

        for s in s_t:
            perfect_a = perfect_agent.draw(s)
            perfect_a_t.append(perfect_a)
            perfect_r_t.append(rew_distributions[s][perfect_a].rvs())

            optimal_a = optimal_agent.draw(s)
            optimal_a_t.append(optimal_a)
            optimal_r_t.append(rew_distributions[s][optimal_a].rvs())

            comm_a = comm_agent.draw(s)
            comm_a_t.append(comm_a)
            comm_r_t.append(rew_distributions[s][comm_a].rvs())

            comm_rev_a = comm_rev_agent.draw(s)
            comm_rev_a_t.append(comm_rev_a)
            comm_rev_r_t.append(rew_distributions[s][comm_rev_a].rvs())

            cls_a = cluster_agent.draw(s)
            cls_a_t.append(cls_a)
            cls_r_t.append(rew_distributions[s][cls_a].rvs())

            cls_rev_a = cluster_rev_agent.draw(s)
            cls_rev_a_t.append(cls_rev_a)
            cls_rev_r_t.append(rew_distributions[s][cls_rev_a].rvs())

            comm_fixed_a = comm_fixed_agent.draw(s)
            comm_fixed_a_t.append(comm_fixed_a)
            comm_fixed_r_t.append(rew_distributions[s][comm_fixed_a].rvs())

            comm_marginal_a = comm_marginal_agent.draw(s)
            comm_marginal_a_t.append(comm_marginal_a)
            comm_marginal_r_t.append(rew_distributions[s][comm_marginal_a].rvs())

        perfect_agent.add_rate(perfect_r)
        comm_agent.add_rate(comm_r)
        cluster_agent.add_rate(b)
        comm_rev_agent.add_rate(comm_rev_r)
        cluster_rev_agent.add_rate(b)
        comm_fixed_agent.add_rate(np.min([perfect_r, rate_max]))
        comm_marginal_agent.add_rate(np.min([perfect_r, rate_max]))

        for idx in range(n_agents):
            perfect_agent.observe(int(s_t[idx]), int(perfect_a_t[idx]), perfect_r_t[idx], iter_num=t)
            cluster_agent.observe(int(s_t[idx]), int(cls_a_t[idx]), cls_r_t[idx], iter_num=t)
            cluster_rev_agent.observe(int(s_t[idx]), int(cls_rev_a_t[idx]), cls_rev_r_t[idx], iter_num=t)
            optimal_agent.observe(int(s_t[idx]), int(optimal_a_t[idx]), optimal_r_t[idx], iter_num=t)
            comm_agent.observe(int(s_t[idx]), int(comm_a_t[idx]), comm_r_t[idx], iter_num=t)
            comm_rev_agent.observe(int(s_t[idx]), int(comm_rev_a_t[idx]), comm_rev_r_t[idx], iter_num=t)
            comm_fixed_agent.observe(int(s_t[idx]), int(comm_fixed_a_t[idx]), comm_fixed_r_t[idx], iter_num=t)
            comm_marginal_agent.observe(int(s_t[idx]), int(comm_marginal_a_t[idx]), comm_marginal_r_t[idx], iter_num=t)

        perfect_agent.update_policy()
        comm_agent.update_policy()
        comm_rev_agent.update_policy()
        cluster_rev_agent.update_policy()
        cluster_agent.update_policy()
        comm_fixed_agent.update_policy()
        comm_marginal_agent.update_policy()

        print(t)
        if not(t % 20):
            print("Saving agents...")
            pickle.dump(perfect_agent, open("results/{}/perfect_agent_{}.pkl".format(experiment, sim_num), "wb"))
            pickle.dump(comm_agent, open("results/{}/comm_agent_{}.pkl".format(experiment, sim_num), "wb"))
            pickle.dump(comm_rev_agent, open("results/{}/comm_rev_agent_{}.pkl".format(experiment, sim_num), "wb"))
            pickle.dump(comm_fixed_agent, open("results/{}/comm_fixed_agent_{}.pkl".format(experiment, sim_num), "wb"))
            pickle.dump(cluster_agent, open("results/{}/cluster_agent_{}.pkl".format(experiment, sim_num), "wb"))
            pickle.dump(cluster_rev_agent,
                        open("results/{}/cluster_rev_agent_{}.pkl".format(experiment, sim_num), "wb"))
            pickle.dump(optimal_agent, open("results/{}/optimal_agent_{}.pkl".format(experiment, sim_num), "wb"))
            pickle.dump(target_kl_list, open("results/{}/kl_target_{}.pkl".format(experiment, sim_num), "wb"))
            pickle.dump(comm_marginal_agent,
                        open("results/{}/comm_marginal_agent_{}.pkl".format(experiment, sim_num), "wb"))

    system_params = {
        'N': n_agents,
        'n_actions': n_actions,
        'n_state': n_states,
        'rate_max': rate_max,
        'eps_kl': eps_1,
        'target_sent_ths': max_distortion,
        'env': rew_distributions
    }

    pickle.dump(system_params, open("results/{}/system_params_{}.pkl".format(experiment, sim_num), "wb"))

