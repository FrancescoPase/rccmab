import numpy as np
from copy import copy
from scipy.special import lambertw
from comm_bandits.it.utils import joint_entropy


def compute_u_r_d(Q: np.ndarray, P: np.ndarray, lam, dist=True):
    """
        Compute the distribution given lambda
    """

    U = copy(lam * P * (1 / np.real(lambertw((lam*P/Q)))))
    n = len(Q.shape)
    if n == 1:
        return U / np.sum(U)
    else:
        n_state = Q.shape[0]
        for row in range(n_state):
            U[row, :] /= np.sum(U[row, :])
    return U


def optimize_lambda_r_d(Q, P, f_x, distortion=None, delta=0.1, base=2):
    """
        Compute the optimal lambda given the distortion constraint
    """
    U = copy(Q)
    n_states = len(f_x)
    if distortion == 0:
        return copy(P), 0
    delta = 0.01
    lam = 0 + delta
    while joint_entropy(P, U, f_x, base=2) > distortion and lam < 100:
        U = compute_u_r_d(Q, P, lam)
        lam += 0.01
    return U, lam


def blahut_arimoto_r_d(P, f_x, distortion, base=2, delta=0.01, eps=0.001):
    n_state = len(f_x)
    j_rs = [100]
    done = False
    Q = np.dot(f_x, P)
    Q = np.tile(Q, [n_state, 1])
    U = None
    while not done:
        U, lam = optimize_lambda_r_d(Q, P, f_x, distortion=distortion, base=base, delta=delta)
        Q = np.dot(f_x, U)
        Q = np.tile(Q, [n_state, 1])
        j_rs.append(joint_entropy(U, Q, f_x))
        if np.abs(j_rs[-1] - j_rs[-2]) < eps:
            done = True
    r = joint_entropy(U, Q, f_x)
    d = joint_entropy(P, U, f_x)
    return U, Q, d, r, lam


def compute_u_d_r(Q: np.ndarray, P: np.ndarray, lam):
    """
        Compute the distribution given lambda
    """

    U = copy(P / (lam * np.real(lambertw(P / (lam * Q)))))
    n = len(Q.shape)
    if n == 1:
        return U / np.sum(U)
    else:
        n_state = Q.shape[0]
        for row in range(n_state):
            U[row, :] /= np.sum(U[row, :])
    return U


def optimize_lambda_d_r(Q, P, f_x, rate, delta=0.1, base=2):
    """
        Compute the optimal lambda given the distortion constraint
    """
    U = copy(P)
    n_states = len(f_x)
    delta = 0.001
    lam = delta

    while joint_entropy(U, Q, f_x, base=2) > rate and lam < 100:
        if lam != 0:
            U = compute_u_d_r(Q, P, lam)
        lam += delta
    return U, lam


def blahut_arimoto_d_r(P, f_x, rate, base=2, delta=0.01, eps=0.001):
    n_state = len(f_x)
    j_rs = [100]
    done = False
    Q = np.dot(f_x, P)
    Q = np.tile(Q, [n_state, 1])
    U = None
    if rate >= joint_entropy(P, Q, f_x, base=2):
        return copy(P), Q, 0, rate, 0
    elif rate == 0:
        return copy(Q), Q, rate, 0
    for e in P:
        e += 0.0000001
        e /= np.sum(e)
    while not done:
        U, lam = optimize_lambda_d_r(Q, P, f_x, rate=rate, base=base, delta=delta)
        Q = np.dot(f_x, U)
        Q = np.tile(Q, [n_state, 1])
        j_rs.append(joint_entropy(U, Q, f_x))
        if np.abs(j_rs[-1] - j_rs[-2]) < eps:
            done = True
    r = joint_entropy(U, Q, f_x)
    d = joint_entropy(P, U, f_x)
    return U, Q, d, r, lam
