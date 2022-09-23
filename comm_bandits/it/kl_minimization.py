import numpy as np
from copy import copy
from comm_bandits.it.utils import joint_entropy


def compute_u_kl_d_r(Q: np.ndarray, P: np.ndarray, lam):
    """
        Compute the distribution given lambda
    """
    U = copy(np.power(P, lam) * np.power(Q, 1-lam))
    n = len(Q.shape)
    if n == 1:
        return U / np.sum(U)
    else:
        n_state = Q.shape[0]
        for row in range(n_state):
            U[row, :] /= np.sum(U[row, :])
    return U


def optimize_lambda_kl_d_r(Q, P, f_x, rate, delta=0.01, base=2):
    """
        Compute the optimal lambda given the distortion constraint
    """
    U = copy(Q)
    lam = delta
    while joint_entropy(U, Q, f_x, base=2) < rate:
        U = compute_u_kl_d_r(Q, P, lam)
        lam += delta
    return U, lam


def blahut_arimoto_kl_d_r(P, f_x, rate, base=2, delta=0.01, eps=0.001):
    n_state = len(f_x)
    j_hs = [100]
    done = False
    Q = np.dot(f_x, P)
    Q = np.tile(Q, [n_state, 1])
    U = None
    if rate >= joint_entropy(P, Q, f_x, base=2):
        return copy(P), Q, 0, rate, 0
    elif rate == 0:
        return copy(Q), Q, 0, rate,  joint_entropy(Q, P, f_x)
    for e in P:
        e += 0.0000001
        e /= np.sum(e)
    while not done:

        U, lam = optimize_lambda_kl_d_r(Q, P, f_x, rate, base=base, delta=delta)
        Q = np.dot(f_x, U)
        Q = np.tile(Q, [n_state, 1])
        h = joint_entropy(U, P, f_x)
        j_hs.append(h)
        if np.abs(j_hs[-1] - j_hs[-2]) < eps:
            done = True
    r = joint_entropy(U, Q, f_x)
    h = joint_entropy(U, P, f_x)
    return U, Q, h, r, lam


def compute_u_kl_r_d(Q: np.ndarray, P: np.ndarray, lam):
    """
        Compute the distribution given lambda
    """
    U = copy(np.power(Q, lam) * np.power(P, 1-lam))
    n = len(Q.shape)
    if n == 1:
        return U / np.sum(U)
    else:
        n_state = Q.shape[0]
        for row in range(n_state):
            U[row, :] /= np.sum(U[row, :])
    return U


def optimize_lambda_kl_r_d(Q, P, f_x, distortion, delta=0.01, base=2):
    """
        Compute the optimal lambda given the distortion constraint
    """
    U = copy(P)
    if distortion == 0:
        return copy(P), 0
    lam = delta
    while joint_entropy(U, P, f_x, base=2) < distortion:
        U = compute_u_kl_r_d(Q, P, lam)
        lam += delta
    return U, lam


def blahut_arimoto_kl_r_d(P, f_x, distortion, base=2, delta=0.01, eps=0.001):
    n_state = len(f_x)
    j_hs = [100]
    done = False
    Q = np.dot(f_x, P)
    Q = np.tile(Q, [n_state, 1])
    U = None
    lam = 0
    while not done:
        U, lam = optimize_lambda_kl_r_d(Q, P, f_x, distortion, delta=delta, base=base)
        Q = np.dot(f_x, U)
        Q = np.tile(Q, [n_state, 1])
        h = joint_entropy(U, P, f_x)
        j_hs.append(h)
        if np.abs(j_hs[-1] - j_hs[-2]) < eps:
            done = True
    r = joint_entropy(U, Q, f_x)
    h = joint_entropy(U, P, f_x)
    return U, Q, h, r, lam

