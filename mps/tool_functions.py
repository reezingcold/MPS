'''
the functions below are widely used in this mps pacakge.
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

#---------------------------------------------------------------------------------#

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

#---------------------------------------------------------------------------------#

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

#---------------------------------------------------------------------------------#

#define direct sum of two matrices
def dsum(A: np.ndarray, B: np.ndarray)->np.ndarray:
    am, an = A.shape
    bm, bn = B.shape
    C = np.vstack((np.hstack((A, np.zeros((am, bn)))), np.hstack((np.zeros((bm, an)), B))))
    return C

# svd decomposition via linalg.svd, but with bond dimension constraint
def svd_bdc(mat: np.ndarray, bdc: int=8)->tuple:
    '''
    @mat: matrix to be decomposed
    @bdc: bond dimension constraints
    '''
    U, Svals, V = linalg.svd(mat)
    trunc = min(bdc, len(Svals))
    S_cut = linalg.diagsvd(Svals[:trunc], trunc, trunc)
    U_cut, V_cut = U[:,:trunc], V[:trunc,:]
    return U_cut, S_cut, V_cut, trunc

#---------------------------------------------------------------------------------#
# an efficient way to generate new left/right environment, used in oneTDVP/twoTDVP
# this is much more faster than np.einsum, the einsum usually 
# doesn't find the most optimal way to contract, even if it does, 
# it's still a little slower than @, reshape and transpose operations, 
# which is highly optimized by openBLAS or intel mkl
def update_leftenv(left: np.ndarray, local_oper: np.ndarray, site_tensor: np.ndarray)->np.ndarray:
    a, b, c = site_tensor.shape
    i, j, k, l = local_oper.shape
    p, q, r = left.shape
    left_env = np.reshape(left.transpose(0, 2, 1), (p*r, q))
    H = np.reshape(local_oper, (i, j*k*l))
    LH = np.reshape(left_env@H, (p, r, j, k, l))
    LH = np.reshape(LH.transpose(0, 3, 2, 1, 4), (p*k*j, r*l))
    LHpsi = np.reshape(LH@np.reshape(site_tensor, (a*b, c)), (p*k, j*c))
    psih = site_tensor.conjugate()
    psih = np.reshape(psih.transpose(2, 0, 1), (c, a*b))
    result = np.reshape(psih@LHpsi, (c, j, c))
    return result

def update_rightenv(right: np.ndarray, local_oper: np.ndarray, site_tensor: np.ndarray)->np.ndarray:
    a, b, c = site_tensor.shape
    i, j, k, l = local_oper.shape
    p, q, r = right.shape
    right_env = np.reshape(right.transpose(1, 0, 2), (q, p*r))
    H = np.reshape(local_oper.transpose(0, 2, 3 ,1), (i*k*l, j))
    HR = np.reshape(H@right_env, (i, k, l, p, r)).transpose(1, 3, 0, 2, 4)
    HR = np.reshape(HR, (k*p*i, l*r))
    HRpsi = HR@np.reshape(site_tensor.transpose(1, 2, 0), (b*c, a))
    psih = np.reshape(site_tensor.conjugate(), (a, b*c))
    result = np.reshape(psih@np.reshape(HRpsi, (k*p, i*a)), (a, i, a))
    return result

#---------------------------------------------------------------------------------#
# an more efficient way to obtain effective Hamiltonian for the 
# next tensor, used in twoTDVP
# this is two times faster than np.einsum which is tested both on 
# my laptop and the server
def lngetHeff(site_tensor: np.ndarray, Heffs: np.ndarray)->np.ndarray:
    a, b, c = site_tensor.shape
    i, j, k, l, w, x, y, z = Heffs.shape
    phi = np.reshape(site_tensor, (a*b, c))
    H = np.reshape(Heffs.transpose(0, 1, 2, 3, 6, 7, 4, 5), (i*j*k*l*y*z, w*x))
    Hphi = np.reshape(H@phi, (i*j, k*l*y*z*c))
    phi_dag = np.reshape((site_tensor.conjugate()).transpose(2, 0, 1), (c, a*b))
    pHp = np.reshape(phi_dag@Hphi, (c, k, l, y, z, c)).transpose(0, 1, 2, 5, 3, 4)
    return pHp

def rngetHeff(site_tensor: np.ndarray, Heffs: np.ndarray)->np.ndarray:
    a, b, c = site_tensor.shape
    i, j, k, l, w, x, y, z = Heffs.shape
    Hp = np.reshape(Heffs, (i*j*k*l*w*x, y*z))@np.reshape(site_tensor.transpose(1, 2, 0), (b*c, a))
    Hp = np.reshape(Hp, (i, j, k, l, w, x, a)).transpose(2, 3, 0, 1, 4, 5, 6)
    phi_dag = site_tensor.conjugate()
    pHp = np.reshape(phi_dag, (a, b*c))@np.reshape(Hp, (k*l, i*j*w*x*a))
    pHp = np.reshape(pHp, (a, i, j, w, x, a)).transpose(1, 2, 0, 3, 4, 5)
    return pHp

# an more efficient way to obtain effective Hamiltonian for the 
# two site tensor, used in twoTDVP
# this is ten times or even twenty times faster than np.einsum, even the 
# optimize parameter in np.einsum is True or 'optimal'
# it seems that the combination of @, reshape, transpose are sometimes 
# much more faster that np.einsum
def getHeffs(left: np.ndarray, oper0: np.ndarray, oper1: np.ndarray, right: np.ndarray)->np.ndarray:
    a, b, c = left.shape
    i, j, k, l = oper0.shape
    w, x, y, z = oper1.shape
    d, e, f = right.shape
    la = np.reshape(left.transpose(0, 2, 1), (a*c, b))@np.reshape(oper0, (i, j*k*l))
    la = np.reshape(np.reshape(la, (a, c, j, k, l)).transpose(0, 3, 1, 4, 2), (a*k*c*l, j))
    rb = np.reshape(oper1.transpose(0, 2, 3, 1), (w*y*z, x))@np.reshape(right.transpose(1, 0, 2), (e, d*f))
    rb = np.reshape(rb, (w, y*z*d*f))
    result = np.reshape(la@rb, (a, k, c, l, y, z, d, f)).transpose(0, 1, 4, 6, 2, 3, 5, 7)
    return result

#---------------------------------------------------------------------------------#
# an more efficient way to get effective Hamiltonian for one site tensor, used in oneTDVP
def getHeff(left: np.ndarray, oper: np.ndarray, right: np.ndarray)->np.ndarray:
    a, b, c = left.shape
    i, j, k, l = oper.shape
    d, e, f = right.shape
    LH = np.reshape(left.transpose(0, 2, 1), (a*c, b))@np.reshape(oper, (i, j*k*l))
    LH = np.reshape(np.reshape(LH, (a, c, j, k, l)).transpose(0, 1, 3, 4, 2), (a*c*k*l, j))
    LHR = np.reshape(LH@np.reshape(right.transpose(1, 0, 2), (e, d*f)), (a, c, k, l, d, f))
    LHR = LHR.transpose(0, 2, 4, 1, 3, 5)
    return LHR

# get Hbond when sweeping from left to right, an alternative way for np.einsum
def lgetHbond(site_tensor, local_oper, site_tensor_h):
    a, b, c = site_tensor.shape
    i, j, k, l, m, n = local_oper.shape
    H = np.reshape(local_oper.transpose(0, 1, 2, 5, 3, 4), (i*j*k*n, l*m))
    Hpsi = np.reshape(H@np.reshape(site_tensor, (a*b, c)), (i, j, k, n, c))
    Hpsi = np.reshape(Hpsi.transpose(0, 1, 2, 4, 3), (i*j, k*c*n))
    result = np.reshape(np.reshape(site_tensor_h.transpose(2, 0, 1), (c, a*b))@Hpsi, (c, k, c, n))
    return result

# get Hbond when sweeping from right to left, an alternative way for np.einsum
def rgetHbond(site_tensor, local_oper, site_tensor_h):
    a, b, c = site_tensor.shape
    i, j, k, l, m, n = local_oper.shape
    Hp = np.reshape(local_oper, (i*j*k*l, m*n))@np.reshape(site_tensor.transpose(1, 2, 0), (b*c, a))
    Hp = np.reshape(np.reshape(Hp, (i, j, k, l, a)).transpose(1, 2, 0, 3, 4), (j*k, i*l*a))
    pHp = np.reshape(np.reshape(site_tensor_h, (a, b*c))@Hp, (a, i, l, a)).transpose(1, 0, 2, 3)
    return pHp