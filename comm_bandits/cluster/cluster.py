import numpy as np
from scipy.stats import entropy
from copy import copy


def cluster_state_with_centroids(centroids, Pi, f_x, rule, base=2):
    n_state = Pi.shape[0]
    n_clst_state = centroids.shape[0]

    association_rule = dict()
    for c in range(n_clst_state):
        association_rule[c] = []

    for p in np.arange(n_state):
        a = np.random.choice(np.arange(centroids.shape[0]))
        m = single_score_rule[rule](Pi[p, :], centroids[a, :], base=base)
        associated_cls = a
        for c in np.arange(n_clst_state):
            s_score = single_score_rule[rule](Pi[p, :], centroids[c, :], base=base)
            if s_score < m:
                m = s_score
                associated_cls = c
        association_rule[associated_cls].append(p)

    return association_rule


def compute_centroids(Pi, centroids, f_x, association_rule, rule):
    n_action = Pi.shape[1]
    n_cls_state = centroids.shape[0]

    for c in range(n_cls_state):
        if len(association_rule[c]):
            if rule == 'kl':
                new_centroid = np.ones(n_action)
            else:
                new_centroid = np.zeros(n_action)

            a_s = np.sum([f_x[p] for p in association_rule[c]])
            for p in association_rule[c]:
                if rule == 'kl':
                    new_centroid *= np.power(Pi[p, :], f_x[p]/a_s)
                elif rule == 'l2_squared':
                    new_centroid += (f_x[p] * Pi[p, :])
                elif rule == 'reverse_kl':
                    new_centroid += (f_x[p] * Pi[p, :])
            # new_centroid /= a_s
            new_centroid /= np.sum(new_centroid)
            centroids[c, :] = new_centroid
    return centroids


def state_reduction(Pi, f_x, b, rule, max_iter=100, base=2, num_rounds=1):
    n_state = Pi.shape[0]
    n_action = Pi.shape[1]
    n_csl = int(np.floor(np.power(2, b)))

    best_score = 100000
    best_association_rule = None
    best_centroids = None
    best_state_cluster = None

    for r in range(num_rounds):

        centroids = np.random.random((n_csl, n_action))
        if rule == 'reverse_kl':
            arm_peaks = np.random.choice(np.arange(n_action), size=n_csl, replace=False)
            for i, e in enumerate(centroids):
                e[arm_peaks[i]] = 6
        row_sums = np.sum(centroids, axis=1)
        centroids /= row_sums[:, np.newaxis]

        association_rule = cluster_state_with_centroids(centroids, Pi, f_x=f_x, rule=rule, base=base)
        centroids = compute_centroids(Pi, centroids, f_x, association_rule, rule=rule)

        i = 0
        keep_going = True
        while keep_going and i < max_iter:
            new_association_rule = cluster_state_with_centroids(centroids, Pi, f_x=f_x, rule=rule, base=2)
            centroids = compute_centroids(Pi, centroids, f_x, new_association_rule, rule=rule)
            state_cluster = dict()
            for cls, points in association_rule.items():
                for p in points:
                    state_cluster[p] = cls
            if new_association_rule == association_rule:
                keep_going = False
            association_rule = copy(new_association_rule)
            i += 1
        cluster_score = score_rule[rule](Pi, centroids, f_x, association_rule, base=base)
        if cluster_score < best_score:
            best_association_rule = copy(association_rule)
            best_centroids = copy(centroids)
            best_state_cluster = copy(state_cluster)
            best_score = cluster_score

    return best_centroids, best_association_rule, best_state_cluster


def compute_cluster_score_kl(P, mu, f_x, association_rule, base=2):
    """
       Evaluate the quality of the clusters based on KL Divergence
    """
    state_kl = np.zeros(len(f_x))
    for k, v in association_rule.items():
        for p in v:
            state_kl[p] = entropy(mu[k, :], P[p, :], base=base)
    return np.dot(f_x, state_kl)


def compute_cluster_score_reverse_kl(P, mu, f_x, association_rule, base=2):
    """
       Evaluate the quality of the clusters based on KL Divergence
    """
    state_kl = np.zeros(len(f_x))
    for k, v in association_rule.items():
        for p in v:
            state_kl[p] = entropy(P[p, :], mu[k, :], base=base)
    return np.dot(f_x, state_kl)


def compute_cluster_score_l2(P, mu, f_x, association_rule, base=2):
    """
       Evaluate the quality of the clusters based on KL Divergence
    """
    state_tv = np.zeros(len(f_x))
    for k, v in association_rule.items():
        for p in v:
            state_tv[p] = np.sum(np.power(mu[k, :] - P[p, :], 2))
    return np.dot(f_x, state_tv)


def compute_single_score_l2(p, q, base=2):
    return np.sum(np.power(p-q, 2))


def compute_single_score_reverse_kl(p, q, base=2):
    return entropy(p, q, base=2)


def compute_single_score_kl(p, q, base=2):
    return entropy(q, p, base=2)


score_rule = {'kl': compute_cluster_score_kl,
              'l2_squared': compute_cluster_score_l2,
              'reverse_kl': compute_cluster_score_reverse_kl}

single_score_rule = {'kl': compute_single_score_kl,
                     'l2_squared': compute_single_score_l2,
                     'reverse_kl': compute_single_score_reverse_kl }
