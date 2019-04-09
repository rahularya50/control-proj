import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def make_symmetric_matrix(N, max_eigen, stdev, exact=False):
    """
    Makes a random N x N symmetric matrix
    -max_eigen is the first and largest eigenvalue
    -if exact, all other eigenvalues are set to stdev
    -if not exact, all other eigenvalues are ~normal(0, stdev)
    """
    A = np.zeros([N, N])
    for i in range(N):
        A[i, i] = np.random.normal(0, stdev) if not exact else stdev
    A[0, 0] = max_eigen
    vecs = make_random_subspace(N, N)
    return vecs @ A @ np.linalg.inv(vecs)

def make_random_direction(N):
    """
    Makes a normalized N-vector where each of its entries are ~normal(0, 1)
    Returns an array of shape (N,)
    """
    vec = np.random.normal(0, 1, N)
    return vec / np.linalg.norm(vec)

def make_random_subspace(N, dims):
    """
    Makes a random N x dims matrix of orthogonal column vectors
    """
    basis = np.array([make_random_direction(N) for _ in range(dims)]).T
    Q, _ = np.linalg.qr(basis)
    return Q

def project(vec, A):
    """
    Projects a vector, vec, onto the columns of a matrix A
    Returns an array of shape (N,)
    """
    b, _, _, _ = np.linalg.lstsq(A, vec, rcond=None)
    return A @ b

def make_random_diagonal_matrix(N, max_lambda, ratio, k=0):
    """
    Makes a random diagonal N x N matrix where eigenvalues are related to l and r
    -Each diagonal element is ~normal(l, l * r^2)
    -Only the subsequent eigenvalues after k eigenvalues will have the previous l scaled by factor r
    -May not be useful to do a dual purpose with the ratio, since it fails to be useful for ratios near 1
    """
    A = np.zeros((N, N))
    k -= 2
    for i in range(N):
        A[i, i] = np.random.normal(max_lambda, max_lambda * ratio ** 2)
        if i > k:
            max_lambda *= ratio
    return A

def simulate_free_response(A, N, x_init):
    """
    Simulates x[k+1] = Ax[k] + Bu
     Parameters
       A - state evolution matrix
       x_init - initial state, numpy row vector
       N - time step to simulate to
     Output
       State trace x - set of all states from x[0] to x[N] as row vectors
    """
    # Do an input validation with the shape of u somewhere
    dim = A.shape[0]
    x = np.zeros([N, dim])
    x[0,:] = x_init
    # B = np.ones([1, dim])
    for i in range(N-1):
        x[i+1,:] = A.dot(x[i,:])
    return x

#edit K, add feedback vector B
def plot_state_trace(x):
    """
    Plots each of the states of some trace x individually
     Parameters
      x - state trace
    """
    N = x.shape[0]
    dim = x.shape[1]
    for i in range(dim):
        fig, ax = plt.subplots()
        ax.plot(range(N), x[:, i])
        ax.set_title('State x{0}'.format(i+1))
    plt.show()