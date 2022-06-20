'''
the functions below are widely used in this mps pacakge.

create date: 2022/01/24
last change: 2022/06/20
'''

import numpy as np
from scipy import linalg

# calculate matrix product of a matrix list
def matprod(array):
    if type(array[0]) == int:
        output = array[0]
        for i in range(1,len(array)):
            output = output*array[i]
    else:
        output = array[0]
        for i in range(1,len(array)):
            output = output@array[i]
    return output

# generate a diagonal matrix for non square matrix
def special_diag(array, shape)->np.ndarray:
    output=np.zeros(shape)
    for i in range(min(min(shape),len(array))):
        output[i][i] = array[i]
    return output

#define commonly used matrix
def eye(n = 2, g = 1)->np.ndarray:
    return g*np.diag([1]*n)

def empty(n = 2)->np.ndarray:
    return np.zeros((n, n))

#define creation and annihilation operators
def create(n = 2)->np.ndarray:
    a = np.zeros((n, n))
    for i in range(1, n):
        a[i][i-1] = np.sqrt(i)
    return a

def annihilate(n = 2)->np.ndarray:
    b = np.zeros((n, n))
    for i in range(0, n-1):
        b[i][i+1] = np.sqrt(i+1)
    return b

#define raising and lowering operator, actually they are 
#creation and annihilation operator when n == 2.
def raising()->np.ndarray:
    return np.array([[0, 0], [1, 0]])

def lowering()->np.ndarray:
    return np.array([[0, 1], [0, 0]])

#define particle number operator
def n()->np.ndarray:
    return np.array([[0, 0], [0, 1]])

#define Pauli matrices
def sigmax(g = 1)->np.ndarray:
    return g*np.array([[0, 1], [1, 0]])

def sigmay(g = 1)->np.ndarray:
    return g*np.array([[0, -1j], [1j, 0]])

def sigmaz(g = 1)->np.ndarray:
    return g*np.array([[1, 0], [0, -1]])

#define direct sum of two matrices
def dsum(A: np.ndarray, B: np.ndarray)->np.ndarray:
    am, an = A.shape
    bm, bn = B.shape
    C = np.vstack((np.hstack((A, np.zeros((am, bn)))), np.hstack((np.zeros((bm, an)), B))))
    return C

# svd decomposition via linalg.svd, but with bond dimension constraint
def svd_bdc(mat, bdc=8):
    '''
    @mat: matrix to be decomposed
    @bdc: bond dimension constraints
    '''
    U, Svals, V = linalg.svd(mat)
    trunc = min(bdc, len(Svals))
    S_cut = linalg.diagsvd(Svals[:trunc], trunc, trunc)
    U_cut, V_cut = U[:,:trunc], V[:trunc,:]
    return U_cut, S_cut, V_cut, trunc