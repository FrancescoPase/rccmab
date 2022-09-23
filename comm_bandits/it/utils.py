import numpy as np
from scipy.stats import entropy


def joint_entropy(A, B, f_x, bound=False,  base=2):
    n = len(f_x)
    if n == 1:
        return entropy(A, B, base=base)

    assert n == np.shape(B)[0] == np.shape(A)[0]

    hs = []
    for i in range(n):
        h = entropy(A[i, :], B[i, :], base=base)
        if bound:
            h /= 2
            h = np.sqrt(h)
        hs.append(h)
    return np.dot(hs, f_x)
