import numpy as np
from scipy.special import lambertw
from copy import copy

from comm_bandits.it.kl_minimization import joint_entropy


def compute_u_l2(Q: np.ndarray, P: np.ndarray, lam, dist=True):
    """
        Compute the distribution given lambda
    """
    if dist:
        U = copy(1/(2*lam) * np.real(lambertw(2 * lam * Q * np.exp(2 * lam * P))))
        n = len(Q.shape)
        if n == 1:
            return U / np.sum(U)
        else:
            n_state = Q.shape[0]
            for row in range(n_state):
                U[row, :] /= np.sum(U[row, :])
        return U
    else:
        U = copy(2 * lam * np.real(lambertw(1 / (2 * lam) * Q * np.exp(P / (2 * lam)))))
        n = len(Q.shape)
        if n == 1:
            return U / np.sum(U)
        else:
            n_state = Q.shape[0]
            for row in range(n_state):
                U[row, :] /= np.sum(U[row, :])
        return U


def optimize_lambda(Q, P, f_x, dist_rate, delta=1.0, fixed='rate'):
    """
        Compute the optimal lambda given the distortion constraint
    """
    # Q is the marginal of P
    n_state = len(f_x)
    U = copy(Q)
    if fixed == 'distortion':
        delta = 1.0
        lam = delta
        if joint_l2(U, P, f_x) <= dist_rate:
            return U, 0
        if dist_rate == 0:
            return copy(P), np.infty
        lam = delta
        while joint_l2(U, P, f_x) > dist_rate:
            U = compute_u_l2(Q, P, lam)
            lam += delta
        return U, lam
    elif fixed == 'rate':
        if joint_entropy(P, Q, f_x) <= dist_rate:
            return copy(P), 0
        if dist_rate == 0:
            return U, 0
        delta = 0.005
        lam = delta
        U = copy(P)
        while joint_entropy(U, Q, f_x) > dist_rate:
            U = compute_u_l2(Q, P, lam, dist=False)
            lam += delta
            U /= np.expand_dims(np.sum(U, axis=1), 1)
            Q = np.dot(f_x, U)
            Q = np.tile(Q, [n_state, 1])
        return U, lam-delta


def joint_l2(Q, P, f_x):
    return np.dot(np.linalg.norm(np.abs(Q - P), ord=2, axis=1), f_x)


def blahut_arimoto_l1_rate(P, f_x, rate=None, delta=1.0, eps=0.001):
    n_state = len(f_x)
    dist_list = [100]
    done = False
    Q = np.dot(f_x, P)
    Q = np.tile(Q, [n_state, 1])
    U = None
    while not done:
        U, lam = optimize_lambda(Q, P, f_x, rate, delta=delta, fixed='rate')
        Q = np.dot(f_x, U)
        Q = np.tile(Q, [n_state, 1])
        dist = joint_l2(U, P, f_x)
        dist_list.append(dist)
        if np.abs(dist_list[-1] - dist_list[-2]) < eps:
            done = True
    r = joint_entropy(U, Q, f_x)
    dist = joint_l2(U, P, f_x)
    return U, Q, dist, r
