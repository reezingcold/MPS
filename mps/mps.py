'''
introducion: this package is created for MPS building, thus mainly including manipulation of tensors
ps: this package depends on numpy(https://numpy.org)
'''
import datetime
import pickle
import numpy as np
from numpy import linalg
from .tool_functions import special_diag

class MPS(object): # for open boundary condition only
    def __init__(self, N, bond_D = 4, phy_dims = 2, initial = 'rc'):
        '''
        @N: number of particles, i.e. length of list, need to be larger than 2
        @bond_D: upper limit of bond dimension, an positive integer
        @phy_dims: an integer(positive) or list, integer will be transferred to list automatically
        '''
        self.N = N
        self.bond_D = bond_D
        self.bond_dims = [bond_D]*(N-1)
        self.phy_dims = [phy_dims]*N if type(phy_dims) == type(1) else phy_dims
        if (len(self.phy_dims) != self.N) or (len(self.bond_dims) != self.N-1):
            raise Exception("dims length should be the same as N.")
        else:
            self.__mps = self.initialize(initial)
            # self.normalization()

    @property
    def site_number(self)->int:
        return self.N
    
    def dag(self):
        tensor_list = [each.conjugate() for each in self.get_data()]
        mps_dag = MPS(self.N, self.bond_D, self.phy_dims)
        for i in range(self.N):
            mps_dag.replace(tensor_list[i], i)
        return mps_dag

    def copy(self):
        tensor_list = self.get_data()
        mps_copy = MPS(self.N, self.bond_D, self.phy_dims)
        for i in range(self.N):
            mps_copy.replace(tensor_list[i], i)
        mps_copy.update()
        return mps_copy

    def get_data(self, site: int = "all"):
        if self.__mps == None:
            raise Exception("Initialization fisrt.")
        else:
            if site == "all":
                return self.__mps
            else:
                return self.__mps[site]
    
    def replace(self, new_tensor, site):
        self.__mps[site] = new_tensor

    def get_dim(self, site):
        if site < 0 or site >= self.N:
            raise Exception("site error, it maybe too large or too small.")
        else:
            pass
        return self.__mps[site].shape
    
    def update(self):
        self.bond_dims = [self.get_dim(i)[-1] for i in range(0,self.N-1)]

    def initialize(self, mold = 'rr'):
        '''
        @mold: create mps filled with 0('z'/"zeros") or random real numbers("rr"/"random real")
        or random complex numbers("rc"/"random complex")
        '''
        mps=[]
        if (mold == 'z') or (mold == "zeros"):
            mps.append(np.zeros((self.phy_dims[0], self.bond_dims[0])))
            for i in range(1, self.N-1):
                site_tensor = np.zeros((self.bond_dims[i-1], self.phy_dims[i], self.bond_dims[i-1]))
                mps.append(site_tensor)
            mps.append(np.zeros((self.bond_dims[self.N-2], self.phy_dims[self.N-1])))
        elif (mold == "rr") or (mold == "random real"):
            mps.append(np.random.rand(self.phy_dims[0], self.bond_dims[0]))
            for i in range(1, self.N-1):
                site_tensor = np.random.rand(self.bond_dims[i-1], self.phy_dims[i], self.bond_dims[i-1])
                mps.append(site_tensor)
            mps.append(np.random.rand(self.bond_dims[self.N-2], self.phy_dims[self.N-1]))
        elif (mold == "rc") or (mold == "random complex"):
            first_tensor = (1+0j)*np.random.rand(self.phy_dims[0], self.bond_dims[0])
            first_tensor += 1j*np.random.rand(self.phy_dims[0], self.bond_dims[0])
            mps.append(first_tensor)
            for i in range(1, self.N-1):
                site_tensor = (1+0j)*np.random.rand(self.bond_dims[i-1], self.phy_dims[i], self.bond_dims[i-1])/3
                site_tensor += 1j*np.random.rand(self.bond_dims[i-1], self.phy_dims[i], self.bond_dims[i-1])/38
                mps.append(site_tensor)
            last_tensor = (1+0j)*np.random.rand(self.bond_dims[self.N-2], self.phy_dims[self.N-1])
            last_tensor += 1j*np.random.rand(self.bond_dims[self.N-2], self.phy_dims[self.N-1])
            mps.append(last_tensor)
        else:
            raise Exception("Initialization error!")
        return mps
    
    # different from normalize(self)
    def normalization(self):
        a, b = self.get_dim(0)
        self.normalize(0)
        a, b = self.get_dim(0)
        normal_coeff = linalg.norm(np.reshape(self.get_data(0), (a*b, 1)))
        self.replace(self.get_data(0)/normal_coeff, 0)

    def isGaugeL(self, site):
        if site == self.N-1:
            raise Exception("the very right side matrix have no left canonical form.")
        else:
            pass
        M = self.__mps[site]
        if site == 0:
            a,c = M.shape
            A_i = M
        else:
            a,b,c = M.shape
            A_i = np.reshape(M, (a*b, c))
        P = A_i.transpose().conjugate()@A_i
        output = abs(linalg.norm(P-np.diag([1]*c))) <= 1e-5
        return output

    def normalize_L(self, site):
        if site < 1 :
            raise Exception("site set to be at least 1.")
        elif site > self.N-1:
            raise Exception("site need to be less than N-1.")
        else:
            pass
        for s in range(0, site):
            if self.isGaugeL(s):
                pass
            else:
                self.normalize_site_L(s)
    
    def isGaugeR(self, site):
        if site == 0:
            raise Exception("the very left side matrix have no right canonical form.")
        else:
            pass
        M = self.__mps[site]
        if site == self.N-1:
            a,c = M.shape
            A_i = M
        else:
            a,b,c = M.shape
            A_i = np.reshape(M, (a, b*c))
        P = A_i@A_i.transpose().conjugate()
        output = abs(linalg.norm(P-np.diag([1]*a))) <= 1e-7
        return output
    
    def normalize_R(self, site):
        if site < 0:
            raise Exception("site set to be larger than 0.")
        elif site > self.N-2 :
            raise Exception("site need to be less than N-1.")
        
        for s in range(self.N-1, site, -1):
            if self.isGaugeR(s):
                pass
            else:
                self.normalize_site_R(s)
    
    def normalize(self, site):
        if site < 0 or site >= self.N:
            raise Exception("site error, it maybe too large or too small.")
        else:
            pass
        if site == 0:
            self.normalize_R(0)
        elif site == self.N-1:
            self.normalize_L(self.N-1)
        else:
            self.normalize_L(site)
            self.normalize_R(site)
    
    def isGaugeCenter(self, site):
        result = 'unknow'
        if site == 0:
            result = all([self.isGaugeR(x) for x in range(1, self.N)])
        elif site == self.N-1:
            result = all([self.isGaugeL(x) for x in range(0, self.N-1)])
        else:
            result = all([self.isGaugeL(x) for x in range(0, site)]+[self.isGaugeR(y) for y in range(site+1, self.N)])
        return result

    # At site, turn site_tensor to GaugeL
    def normalize_site_L(self, site):
        if site < 0 or site >= self.N-1:
            raise Exception("site error, it maybe too large or too small.")
        else:
            pass
        psi = self.__mps
        bd = self.bond_dims
        pd = self.phy_dims
        D = self.bond_D
        if site == 0:
            M_aux = psi[0]@np.reshape(psi[1], (bd[0], pd[1]*bd[1]))
            U, S, Vh = linalg.svd(M_aux)
            if S.shape[0] <= D:
                psi[0] = U
                bd[0] = pd[0]
                psi[1] = np.reshape(special_diag(S, (pd[0], pd[1]*bd[1]))@Vh, (bd[0], pd[1], bd[1]))
            else:
                bd[0] = D
                psi[0] = U[:,:D]
                Sp = np.diag(S[:D])
                psi[1] = np.reshape(Sp@Vh[:D,:], (bd[0], pd[1], bd[1]))
        elif 1 <= site <= self.N-2:
            i = site
            if i == self.N-2: # dealing with the last vector(tensor)
                M_i = np.reshape(psi[i], (bd[i-1]*pd[i], bd[i]))
                M_i1 = psi[i+1]
                U, S, Vh = linalg.svd(M_i@M_i1)
                if len(S) <= D:
                    bd[i] = pd[i]*bd[i-1]
                    psi[i] = np.reshape(U, (bd[i-1], pd[i], bd[i]))
                    psi[i+1] = np.reshape(special_diag(S,(bd[i], pd[i+1]))@Vh,(bd[i],pd[i+1]))
                else: # i.e. len(S) > D
                    bd[i] = D
                    psi[i] = np.reshape(U[:,:D], (bd[i-1], pd[i], D))
                    Sp = np.diag(S[:D])
                    psi[i+1] = Sp@Vh[:D,:]
            else:
                M_i = np.reshape(psi[i], (bd[i-1]*pd[i], bd[i]))
                M_i1 = np.reshape(psi[i+1], (bd[i], pd[i+1]*bd[i+1]))
                U, S, Vh = linalg.svd(M_i@M_i1)
                if len(S) <= D:
                    bd[i] = pd[i]*bd[i-1]
                    psi[i] = np.reshape(U, (bd[i-1], pd[i], bd[i]))
                    psi[i+1] = np.reshape(special_diag(S,(bd[i], pd[i+1]*bd[i+1]))@Vh,(bd[i],pd[i+1],bd[i+1]))
                else: # i.e. len(S) > D
                    bd[i] = D
                    psi[i] = np.reshape(U[:,:D], (bd[i-1], pd[i], D))
                    Sp = S[:D]
                    psi[i+1] = np.reshape(np.diag(Sp)@Vh[:D,:], (D, pd[i+1], bd[i+1]))
        else:
            raise Exception("can't be left any more.")
    
    # At site, turn site_tensor to GaugeR
    def normalize_site_R(self, site):
        if site < 0 or site >= self.N:
            raise Exception("site error, it maybe too large or too small.")
        else:
            pass
        psi = self.__mps
        bd = self.bond_dims
        pd = self.phy_dims
        D = self.bond_D
        N = self.N
        if site == N-1:
            M_aux = np.reshape(psi[N-2], (bd[N-3]*pd[N-2],bd[N-2]))@psi[N-1]
            U, S, Vh = linalg.svd(M_aux)
            if S.shape[0] <= D:
                psi[N-1] = Vh
                bd[N-2] = pd[N-1]
                psi[N-2] = np.reshape(U@special_diag(S, (bd[N-3]*pd[N-2],bd[N-2])), (bd[N-3],pd[N-2],bd[N-2]))
            else:
                bd[N-2] = D
                psi[N-1] = Vh[:D,:]
                psi[N-2] = np.reshape(U[:,:D]@np.diag(S[:D]), (bd[N-3, pd[N-2], bd[N-2]]))
        elif 1 <= site <= N-2:
            i = site
            if i == 1: # dealing with the last vector(tensor)
                M_i = np.reshape(psi[1], (bd[0], bd[1]*pd[1]))
                M_i1 = psi[0]
                U, S, Vh = linalg.svd(M_i1@M_i)
                if len(S) <= D:
                    bd[0] = pd[1]*bd[1]
                    psi[1] = np.reshape(Vh, (bd[0], pd[1], bd[1]))
                    psi[0] = U@special_diag(S, (pd[0], bd[0]))
                else: # i.e. len(S) > D
                    print("hello")
                    bd[i-1] = D
                    psi[i] = np.reshape(Vh[:D,:], (D, pd[i], bd[i]))
                    Sp = np.diag(S[:D])
                    psi[i-1] = U[:,:D]@Sp
            else:
                M_i = np.reshape(psi[i], (bd[i-1], bd[i]*pd[i]))
                M_i1 = np.reshape(psi[i-1], (bd[i-2]*pd[i-1], bd[i-1]))
                U, S, Vh = linalg.svd(M_i1@M_i)
                if len(S) <= D:
                    bd[i-1] = pd[i]*bd[i]
                    psi[i] = np.reshape(Vh, (bd[i-1], pd[i], bd[i]))
                    psi[i-1] = np.reshape(U@special_diag(S, (pd[i-1]*bd[i-2],bd[i-1])), (bd[i-2],pd[i-1],bd[i-1]))
                else: # i.e. len(S) > D
                    bd[i-1] = D
                    psi[i] = np.reshape(Vh[:D,:], (D, pd[i], bd[i]))
                    Sp = S[:D]
                    psi[i-1] = np.reshape(U[:,:D]@np.diag(Sp), (bd[i-2],pd[i-1],D))
        else:
            raise Exception("can't be right any more.")
    
    # local inner product with self
    def local_inner_product(self, site, local_operator = None, eic = False):
        if site < 0 or site >= self.N:
            raise Exception("site error, it maybe too large or too small.")
        else:
            pass
        site_state = self.get_data(site)
        if site == 0:
            a, b = site_state.shape
            if type(local_operator) == type(None):
                result = site_state.conjugate().transpose()@site_state
            else:
                if (local_operator.shape != (a, a)) or (len(local_operator.shape) != 2):
                    raise Exception("local operator(tensor) shape error.")
                else:
                    result = site_state.conjugate().transpose()@local_operator@site_state
            if eic == True:
                result = np.reshape(result, (1, b*b))
            else:
                pass
        elif site == self.N-1:
            a, b = site_state.shape
            if type(local_operator) == type(None):
                result = site_state.conjugate()@site_state.transpose()
            else:
                if (local_operator.shape != (b, b)) or (len(local_operator.shape) != 2):
                    raise Exception("local operator(tensor) shape error.")
                else:
                    result = site_state.conjugate()@local_operator@site_state.transpose()
            if eic == True:
                result = np.reshape(result, (a*a, 1))
            else:
                pass
        else: # i.e. tensors in between
            a, b, c = site_state.shape
            site_state_up = np.reshape(site_state.transpose(0, 2, 1), (a*c, b)).conjugate()
            site_state_down = np.reshape(site_state.transpose(1, 0, 2), (b, a*c))
            if type(local_operator) == type(None):
                result = np.reshape(site_state_up@site_state_down, (a, c, a, c))
            else:
                if (local_operator.shape != (b, b)) or (len(local_operator.shape) != 2):
                    raise Exception("local operator(tensor) shape error.")
                else:
                    result = np.reshape(site_state_up@local_operator@site_state_down, (a, c, a, c))
            if eic == True:
                result = np.reshape(result.transpose(0, 2, 1, 3), (a*a, c*c))
            else:
                pass
        return result
    
    # inner product with self
    def inner_product(self, real = False):
        result = self.local_inner_product(0, eic = True)
        for i in range(1, self.N):
            result = result@self.local_inner_product(i, eic = True)
        result = linalg.norm(result[0][0]) if real else result[0][0]
        return result
    
    def expectation_value(self, oper_list, real = True):
        if len(oper_list) != self.N:
            raise Exception("length of operator list doesn't cooperate with the state")
        else:
            pass
        result = self.local_inner_product(0, local_operator = oper_list[0], eic = True)
        for i in range(1, self.N):
            result = result@self.local_inner_product(i, local_operator = oper_list[i], eic = True)
        result = linalg.norm(result[0][0]) if real else result[0][0]
        return result
    
    def transfer_matrix(self, site, local_operator, eic = True):
        '''
        @eic: external index contraction, 
         if True, return a matrix, 
         if given 'l' or 'left', return (1, n) tensor
         if given 'r' or 'right', return (n, 1) tensor
         if False, return (m, n) tensor.
        '''
        if site < 0 or site >= self.N:
            raise Exception("site error, it maybe too large or too small.")
        else:
            pass
        site_tensor = self.get_data(site)
        site_tensor_dag = site_tensor.conjugate()
        if site != 0 and site != self.N-1:
            a, b, c = site_tensor.shape
            i, j, k, l = local_operator.shape
            #transfer = np.einsum('abc,ijbk,dkf->aidcjf', site_tensor_dag, local_operator, site_tensor, optimize=True)
            temp_tensor = np.reshape(site_tensor_dag.transpose(0, 2, 1), (a*c, b))@np.reshape(local_operator.transpose(2, 0, 1, 3), (k, i*j*l))
            temp_tensor = np.reshape(temp_tensor, (a*c*i*j, l))@np.reshape(site_tensor.transpose(1, 0, 2), (b, a*c))
            transfer = np.reshape(temp_tensor, (a, c, i, j, a, c)).transpose(0, 2, 4, 1, 3, 5)
            if eic == True:
                transfer = np.reshape(transfer, (a*i*a, c*j*c))
            elif (eic == 'l') or (eic == 'left'):
                transfer = np.reshape(transfer, (a*i*a, c, j, c))
            elif (eic == 'r') or (eic == 'right'):
                transfer = np.reshape(transfer, (a, i, a, c*j*c))
            else:
                pass
        else:
            a, b = site_tensor.shape
            i, j, k = local_operator.shape
            if site == 0:
                transfer = np.einsum('ab,ica,cd->dib', site_tensor, local_operator, site_tensor_dag, optimize=True)
                if (eic == True) or (eic == 'r') or (eic == 'right'):
                    transfer = np.reshape(transfer, (1, b*i*b))
                else:
                    pass
            else:
                transfer = np.einsum('ab,idb,cd->cia', site_tensor, local_operator, site_tensor_dag, optimize=True)
                if (eic == True) or (eic == 'l') or (eic == 'left'):
                    transfer = np.reshape(transfer, (a*i*a, 1))
                else:
                    pass
        return transfer

# save mps information
def savemps(mps: MPS, fname:str='mps_data')->str:
    mps.update()
    with open(fname, 'wb') as f:
        pickle.dump(mps, f)
    print("Successfully saved in file "+fname)
    return fname

# load mps information
def loadmps(fname: str)->MPS:
    with open(fname, 'rb') as f:
        r = pickle.load(f)
    return r

# there are new functions for doing the following task in mps_functions
# this function is written to obtain the state after a mpo operate on a mps
# calculate O|psi>
"""
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
            a, b = site_tensor.shape
            i, j, k = site_operator.shape
            final_site_tensor = np.reshape(site_operator, (i*j, k))@site_tensor
            final_site_tensor = np.reshape(np.reshape(final_site_tensor, (i, j, b)).transpose(1, 0, 2), (j, i*b))
        elif s == N-1:
            a, b = site_tensor.shape
            i, j, k = site_operator.shape
            final_site_tensor = np.reshape(site_operator, (i*j, k))@site_tensor.transpose()
            final_site_tensor = np.reshape(np.reshape(final_site_tensor, (i, j, a)).transpose(0, 2, 1), (i*a, j))
        else: # those in between
            a, b, c = site_tensor.shape
            i, j, k, l = site_operator.shape
            final_site_tensor = np.reshape(site_operator, (i*j*k, l))@np.reshape(site_tensor.transpose(1, 0, 2), (b, a*c))
            final_site_tensor = np.reshape(final_site_tensor, (i, j, k, a, c)).transpose(0, 3, 2, 1, 4)
            final_site_tensor = np.reshape(final_site_tensor, (i*a, k, j*c))
        result_state.replace(final_site_tensor, s)
        result_state.update()
    return result_state
"""
# calculate <phi|O|psi>
"""
def multiply_former(phi_dag: MPS, psi: MPS, observable: MPO = None)->complex:
    '''
    @phi_dag: bra, <phi|
    @psi: ket, |psi>
    @observable: if None, return <phi|psi>, else return <phi|O|psi>
    '''
    if phi_dag.site_number == psi.site_number:
        N = phi_dag.site_number
    else:
        raise Exception("Two mps should have the same length.")
    matrix_list=[0]*N
    if observable == None:
        for s in range(0, N):
            site_psi = psi.get_data(s)
            site_phi = phi_dag.get_data(s)
            if s == 0:
                a, b = site_psi.shape
                d, e = site_phi.shape
                site_matrix = np.reshape(site_phi.transpose()@site_psi, (1, e*b))
            elif s == N-1:
                a, b = site_psi.shape
                d, e = site_phi.shape
                site_matrix = np.reshape(site_phi@site_psi.transpose(), (d*a, 1))
            else:
                a, b, c = site_psi.shape
                d, e, f = site_phi.shape
                site_phi_temp = np.reshape(site_phi.transpose(0, 2, 1), (d*f, e))
                site_psi_temp = np.reshape(site_psi.transpose(1, 0, 2), (b, a*c))
                site_matrix = np.reshape(np.reshape(site_phi_temp@site_psi_temp, (d, f, a, c)).transpose(0, 2, 1, 3), (d*a, f*c))
            matrix_list[s] = site_matrix
    elif observable.site_number == N:
        for s in range(0, N):
            site_psi = psi.get_data(s)
            site_phi = phi_dag.get_data(s)
            site_operator = observable.get_data(s)
            if s == 0:
                a, b = site_psi.shape
                d, e = site_phi.shape
                i, j, k = site_operator.shape
                site_temp = site_phi.transpose()@np.reshape(site_operator.transpose(1, 0, 2), (j, i*k))
                site_matrix = np.reshape(np.reshape(site_temp, (e, i*k))@site_psi, (1, e*i*b))
            elif s== N-1:
                a, b = site_psi.shape
                d, e = site_phi.shape
                i, j, k = site_operator.shape
                site_temp = np.reshape(site_phi@np.reshape(site_operator.transpose(1, 0, 2), (j, i*k)), (d*i, k))
                site_matrix = np.reshape(site_temp@site_psi.transpose(), (d*i*a, 1))
            else:
                a, b, c = site_psi.shape
                d, e, f = site_phi.shape
                i, j, k, l = site_operator.shape
                site_temp = np.reshape(site_phi.transpose(0, 2, 1), (d*f, e))@np.reshape(site_operator.transpose(2, 0, 1, 3), (k, i*j*l))
                site_temp = np.reshape(site_temp, (d*f*i*j, l))@np.reshape(site_psi.transpose(1, 0, 2), (b, a*c))
                site_matrix = np.reshape(np.reshape(site_temp, (d, f, i, j, a, c)).transpose(0, 2, 4, 1, 3, 5), (d*i*a, f*j*c))
            matrix_list[s] = site_matrix
    return matprod(matrix_list)[0][0]
"""
