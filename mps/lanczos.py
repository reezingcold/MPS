# lanczos method written here is modified from 
# https://tenpy.readthedocs.io/en/latest/toycodes/lanczos.html

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import expm
# Lanczos algorithm for compute V, T such that H = VT(V+)
def lanczos_method(H: np.ndarray, vec0: np.ndarray, n: int = 16):
    if vec0.ndim != 1:
        vec0 = np.reshape(vec0, (vec0.shape[0]*vec0.shape[1], ))
    if vec0.shape[0] != H.shape[1]:
        raise Exception("H doesn't correspond with vec0")
    # normlization
    vec0 = vec0/np.linalg.norm(vec0)
    # a list for save vectors
    V = [vec0]
    # the tri-diagnal matrix
    T = np.zeros((n, n))
    # the following procedure purely depends on 
    # wikipedia (https://en.wikipedia.org/wiki/Lanczos_algorithm)
    omega = H @ vec0
    alpha = np.inner(vec0.conj(), omega).real
    omega = omega - alpha * V[-1]
    T[0, 0] = alpha
    for i in range(1, n):
        beta = linalg.norm(omega)
        if beta <= 1.e-10:
            # if beta -> 0, we stop
            T = T[:i, :i]
            break
        omega /= beta
        V.append(omega)
        omega = H @ omega - beta * V[-2]
        alpha = np.inner(V[-1].conj(), omega).real
        omega = omega - alpha * V[-1]
        T[i, i], T[i-1, i], T[i, i-1] = alpha, beta, beta
    return T, np.array(V).T

# use lanczos method to find ground state
# in each iteration, the initial vector is from last iteration
# we stop until the ground energy converges
def lanczos_gs(H: np.ndarray, vec0: np.ndarray, n: int = 16, 
                iter_max: int = 10, error: float = 1e-3):
    T, V = lanczos_method(H, vec0, n)
    E, v = linalg.eigh_tridiagonal(np.diag(T), np.diag(T[:-1, 1:]))
    Eg, vec = E[0], V @ v[:, 0]
    i = 1
    while i <= iter_max:
        T, V = lanczos_method(H, vec, n)
        E, v = linalg.eigh_tridiagonal(np.diag(T), np.diag(T[:-1, 1:]))
        vec = V @ v[:, 0]
        i += 1
        if abs(E[0]-Eg) <= error:
            Eg = E[0]
            break
        else:
            Eg = E[0]
    return Eg, vec

# lanczos method is also a good way to calculate the action 
# e^(tA) on a given vector psi, the basic idea is actually diagonalizing 
# the matrix A in Krylov subspace, i.e., 
# A = PD(P^+) then e^(tA) = P e^(tD) e_0
def lanczos_expm_multiply(H: np.ndarray, psi: np.ndarray, dt: float, n: int = 16):
    T, V = lanczos_method(H, psi, n)
    result = V @ expm(-1.j*T*dt)[:, 0]
    if linalg.norm(result) - 1. >= 1.e-5:
        print("poorly conditioned lanczos")
        result /= linalg.norm(result)
    result = np.reshape(result, (result.shape[0], 1))
    return result