import numpy as np

def make_symmetric_matrix(N, max_eigen, stdev, exact=False):
    A = np.zeros([N, N])
    for i in range(N):
        A[i, i] = np.random.normal(0, stdev) if not exact else stdev
    A[0, 0] = max_eigen
    vecs = make_random_subspace(N, N)
    return vecs @ A @ np.linalg.inv(vecs)

def make_random_direction(N):
    vec = np.random.normal(0, 1, N)
    return vec / np.linalg.norm(vec)

def make_random_subspace(N, dims):
    basis = np.array([make_random_direction(N) for _ in range(dims)]).T
    Q, _ = np.linalg.qr(basis)
    return Q

def project(vec, A):
    b, _, _, _ = np.linalg.lstsq(A, vec, rcond=None)
    return A @ b

def make_random_diagonal_matrix(N, max_lambda, ratio):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = np.random.normal(max_lambda, max_lambda * ratio ** 2)
        max_lambda *= ratio
    return A