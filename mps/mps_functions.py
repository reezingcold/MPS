'''
introducion: some useful functions for calculating in MPS form
'''

from .mps import MPS
from .mpo import MPO
from .tool_functions import matprod, dsum
import numpy as np
from numpy import linalg

# O|a>
def operate(operator: MPO, state: MPS)->MPS:
    '''
    @operator: operator is the mpo of a Hamiltonian
    @state: state should be a corresponding mps state
    @bond_D: upper limit of bond dimension of the result mps
    '''
    N = state.site_number
    result_state = state.copy()
    for s in range(0, N):
        site_tensor = state.get_data(s)
        site_operator = operator.get_data(s)
        if s == 0:
            new_site_tensor = np.einsum('ijk,kb->jib', site_operator, site_tensor, optimize=True)
            j, i, b = new_site_tensor.shape
            new_site_tensor = np.reshape(new_site_tensor, (j, i*b))
        elif s == N-1:
            new_site_tensor = np.einsum('ijk,ak->iaj', site_operator, site_tensor, optimize=True)
            i, a, j = new_site_tensor.shape
            new_site_tensor = np.reshape(new_site_tensor, (i*a, j))
        else:
            new_site_tensor = np.einsum('ijkl,alc->iakjc', site_operator, site_tensor, optimize=True)
            i, a, k, j, c = new_site_tensor.shape
            new_site_tensor = np.reshape(new_site_tensor, (i*a, k, j*c))
        result_state.replace(new_site_tensor, s)
    result_state.update()
    return result_state

# <a|b>
def multiply(phi_dag: MPS, psi: MPS)->complex:
    '''
    @phi_dag: bra, <phi|
    @psi: ket, |psi>
    '''
    if phi_dag.site_number == psi.site_number:
        N = phi_dag.site_number
    else:
        raise Exception("Two mps should have the same length.")
    matrix_list = [0]*N
    for s in range(0, N):
        site_psi = psi.get_data(s)
        site_phi = phi_dag.get_data(s)
        if s != 0 and s != N-1:
            site_matrix = np.einsum('def,aec->dafc', site_phi, site_psi, optimize=True)
            d, a, f, c = site_matrix.shape
            site_matrix = np.reshape(site_matrix, (d*a, f*c))
        elif s == 0:
            site_matrix = np.einsum('cd,cb->db', site_phi, site_psi, optimize=True)
            d, b = site_matrix.shape
            site_matrix = np.reshape(site_matrix, (1, d*b))
        else:
            site_matrix = np.einsum('cd,ad->ca', site_phi, site_psi, optimize=True)
            c, a = site_matrix.shape
            site_matrix = np.reshape(site_matrix, (c*a, 1))
        matrix_list[s] = site_matrix
    result = matprod(matrix_list)[0][0]
    return result

# <a_i|b_i>
def site_multiply(phi_dag: MPS, psi: MPS, site: int)->np.ndarray:
    '''
    @phi_dag: bra, <phi|
    @psi: ket, |psi>
    @site: where to contract
    '''
    if phi_dag.site_number == psi.site_number:
        N = phi_dag.site_number
    else:
        raise Exception("Two mps should have the same length.")
    tensor_up = phi_dag.get_data(site)
    tensor_down = psi.get_data(site)
    if site != 0 and site != N-1:
        site_tensor = np.einsum('def,aec->dafc', tensor_up, tensor_down, optimize=True)
        d, a, f, c = site_tensor.shape
        site_tensor = np.reshape(site_tensor, (d*a, f*c))
    elif site == 0:
        site_tensor = np.einsum('cd,cb->db', tensor_up, tensor_down, optimize=True)
        d, b = site_tensor.shape
        site_tensor = np.reshape(site_tensor, (1, d*b))
    else:
        site_tensor = np.einsum('cd,ad->ca', tensor_up, tensor_down, optimize=True)
        c, a = site_tensor.shape
        site_tensor = np.reshape(site_tensor, (c*a, 1))
    return site_tensor

def contract(phi_dag: MPS, psi: MPS, site: int):
    '''
    @phi_dag: state to be optimized
    @psi: given state
    @site: where to leave uncontracted index
    '''
    if phi_dag.site_number == psi.site_number:
        N = phi_dag.site_number
    else:
        raise Exception("two states should have the same length.")
    if 0 <= site < N:
        pass
    else:
        raise Exception("site number should between 0 and N-1.")
    
    site_psi, site_phi = psi.get_data(site), phi_dag.get_data(site)
    if site != 0 and site != N-1:
        rhs = matprod([site_multiply(phi_dag, psi, i) for i in range(0, site)])
        lhs = matprod([site_multiply(phi_dag, psi, i) for i in range(site+1, N)])
        a, b, c = site_psi.shape
        d, e, f = site_phi.shape
        rhs = np.reshape(rhs, (d, a))
        lhs = np.reshape(lhs, (f, c))
        vec = np.reshape(np.einsum('ha,abc,jc->hbj', rhs, site_psi, lhs, optimize=True), (1, d*b*f))
    elif site == 0:
        rhs = matprod([site_multiply(phi_dag, psi, i) for i in range(1, N)])
        a, b = site_psi.shape
        c, d = site_phi.shape
        rhs = np.reshape(rhs, (d, b))
        vec = np.reshape(np.einsum('ab,eb->ae', site_psi, rhs, optimize=True), (1, a*d))
    else:
        lhs = matprod([site_multiply(phi_dag, psi, i) for i in range(0, N-1)])
        a, b = site_psi.shape
        c, d = site_phi.shape
        lhs = np.reshape(lhs, (c, a))
        vec = np.reshape(np.einsum('ea,ab->eb', lhs, site_psi, optimize=True), (1, c*b))
    return vec

# |mps1>+|mps2>, no truncation
def accurate_plus(phi: MPS, psi: MPS)->MPS:
    """
    @phi, psi: two states, mps form
    """
    if phi.site_number == psi.site_number:
        N = phi.site_number
    else:
        raise Exception("Two mps should have the same length.")
    
    result = MPS(N, phy_dims = phi.phy_dims, initial = 'z')
    # site == 0
    # print(phi.get_data(0).shape, psi.get_data(0).shape)
    result.replace(np.hstack((phi.get_data(0), psi.get_data(0))), 0)
    # print(result.get_dim(0),'line')
    # 1 <= site <= N-2
    for s in range(1, N-1):
        if phi.get_dim(s)[1] == psi.get_dim(s)[1]:
            phy_dim = phi.get_dim(s)[1]
            site_tensor = np.einsum('abc->bac', result.get_data(s))
            site_phi = np.einsum('abc->bac', phi.get_data(s))
            site_psi = np.einsum('abc->bac', psi.get_data(s))
        else:
            raise Exception("Two site tensor should have same physical dimensions.")
        tensor_list = []
        for d in range(0, phy_dim):
            tensor_list.append(dsum(site_phi[d], site_psi[d]))
        site_tensor = np.array(tensor_list)
        result.replace(np.einsum('bac->abc', site_tensor), s)
        # print(result.get_dim(s),'line')
    result.replace(np.vstack((phi.get_data(N-1), psi.get_data(N-1))), N-1)
    result.update()
    return result

# mps -> vector
def mps_contract(phi: MPS)->np.ndarray:
    N = phi.site_number
    vec = phi.get_data(0)
    for i in range(1, N-1):
        a, b = vec.shape
        b, c, d = phi.get_dim(i)
        vec = np.reshape(np.einsum('ab,bcd->acd', vec, phi.get_data(i), optimize=True), (a*c, d))
    a, c = vec.shape[0], phi.get_dim(N-1)[1]
    vec = np.reshape(np.einsum('ab,bc->ac', vec, phi.get_data(N-1), optimize=True), (a*c, 1))
    return vec

# mpo -> matrix
def mpo_contract(oper: MPO)->np.ndarray:
    N = oper.site_number
    mat = oper.get_data(0)
    for i in range(1, N-1):
        a, b, c  = mat.shape
        j, k, l, m = oper.get_dim(i)
        mat = np.reshape(np.einsum('abc,aklm->kblcm', mat, oper.get_data(i), optimize=True), (k, b*l, c*m))
    a, b, c = mat.shape
    d, e, f = oper.get_dim(N-1)
    mat = np.reshape(np.einsum('abc,aef->becf', mat, oper.get_data(N-1), optimize=True), (b*e, c*f))
    return mat

# get half chain entanglement entropy, based on 2
def getEntropy(mps: MPS, bond_site: int):
    mps.normalization()
    mps.normalize(bond_site)
    site_tensor = mps.get_data(bond_site)
    a, b, c = mps.get_dim(bond_site)
    site_matrix = np.reshape(site_tensor, (a*b, c))
    U, S, Vh = linalg.svd(site_matrix)
    entropy = 0
    for s in S:
        entropy += -(s**2)*np.log2(s**2)
    return entropy


