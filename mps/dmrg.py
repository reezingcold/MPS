import numpy as np
from scipy import linalg
from .mps import MPS
from .mpo import MPO
from .tool_functions import matprod
from .lanczos import lanczos_gs

class oneDMRG(object):
    def __init__(self, mps: MPS, H: MPO, max_iter = 10, delta_E = 1e-5, use_lanczos = False):
        '''
        the one site DMRG algorithm
        @mps: the initial state, given in the form of mps
        @H: the hamiltonian, given in the form of mpo
        @max_iter: the upper limit of sweep times
        @delta_E: stop the iteration(sweep) if (E_before-E_after) < delta_E
        '''
        self.__mps = mps
        self.__H = H
        self.max_iter = max_iter
        self.delta_E = delta_E
        self.use_lanczos = use_lanczos
        self.lanczos_dim = 8
        self.cup = [0]*mps.N
        if mps.N == H.N:
            self.N = mps.N
        else:
            raise Exception("length doesn't cooperate.")
        self.gs_energy = None
        self.ground_state = None
        # initialize right environment and also left
        self.__mps.normalization()
        rightenv = [self.__mps.transfer_matrix(x, self.__H.get_data(x)) for x in range(1, self.N)]
        self.right = [0]*(self.N-1)
        self.right[self.N-2] = rightenv[self.N-2]
        current_right = self.right[self.N-2]
        for i in range(self.N-3, -1, -1):
            current_right = rightenv[i]@current_right
            self.right[i] = current_right
        self.left = [0]*(self.N-1)
    
    @property
    def mps(self):
        return self.__mps
    
    @property
    def ground_energy(self):
        if self.gs_energy == None:
            raise Exception("sweep first!")
        else:
            return self.gs_energy
    
    @property
    def result(self):
        self.sweep()
        if self.gs_energy == None:
            raise Exception("sweep first!")
        else:
            return self.gs_energy, self.ground_state
    
    @property
    def H(self):
        return self.__H
    
    def __local_gs(self, H, site, iter_max = 10, err = 1e-5):
        if self.use_lanczos:
            state = self.__mps
            vec0 = np.reshape(state.get_data(site), (np.prod(state.get_dim(site)), 1))
            local_gs = lanczos_gs(H, vec0, self.lanczos_dim, iter_max=iter_max, error=err)[1]
            local_gs = np.reshape(local_gs, state.get_dim(site))
        else:
            eigresult = linalg.eigh(H)
            local_gs = eigresult[1][:,0]
            local_gs = np.reshape(local_gs, self.__mps.get_dim(site))
        return local_gs

    def __optL(self):
        # optimize the very left side site tensor
        mps = self.__mps
        H = self.__H
        N = self.N
        if not mps.isGaugeCenter(0):
            mps.normalize(0)
        # obtain effective hamiltonian first
        right = self.right[0]
        a, b = mps.get_dim(0)
        i, j, k = H.get_dim(0)
        right = np.reshape(right, (b, i, b))
        Heff = np.einsum('ijk,eig->jekg', H.get_data(0), right, optimize=True)
        Heff = np.reshape(Heff, (a*b, a*b))
        # do eigen decomposition
        local_gs_mps = self.__local_gs(Heff, 0)
        mps.replace(local_gs_mps, 0)
        mps.normalize_site_L(0)
        # calculate left environment for following operations
        self.left[0] = mps.transfer_matrix(0, H.get_data(0))
    
    def __optLtoR(self, site):
        # optimize the site tensors in between from left to right
        mps = self.__mps
        H = self.__H
        N = self.N
        if not mps.isGaugeCenter(site):
            mps.normalize(site)
        left = self.left[site-1]
        right = self.right[site]
        # obtain effective hamiltonian first
        a, b, c = mps.get_dim(site)
        i, j, k, l = H.get_dim(site)
        left, right = np.reshape(left, (a, i, a)), np.reshape(right, (c, j, c))
        Heff = np.einsum('gim,ijkl,njq->gknmlq', left, H.get_data(site), right, optimize=True)
        Heff = np.reshape(Heff, (a*b*c, a*b*c))
        # do eigen decomposition
        local_gs_mps = self.__local_gs(Heff, site)
        mps.replace(local_gs_mps, site)
        mps.normalize_site_L(site)
        # calculate left environment for following operations
        self.left[site] = np.reshape(left, (1, a*i*a))@mps.transfer_matrix(site, H.get_data(site))
    
    def __optR(self):
        # optimize the site tensors in between from left to right
        mps = self.__mps
        H = self.__H
        N = self.N
        if not mps.isGaugeCenter(N-1):
            mps.normalize(N-1)
        left = self.left[N-2]
        # obtain effective hamiltonian first
        a, b = mps.get_dim(N-1)
        i, j, k = H.get_dim(N-1)
        left = np.reshape(left, (a, i, a))
        Heff = np.einsum('eig,ijk->ejgk', left, H.get_data(N-1), optimize=True)
        Heff = np.reshape(Heff, (a*b, a*b))
        # do eigen decomposition
        local_gs_mps = self.__local_gs(Heff, N-1)
        mps.replace(local_gs_mps, N-1)
        mps.normalize_site_R(N-1)
        # calculate right environment for following operations
        self.right[N-2] = mps.transfer_matrix(N-1, H.get_data(N-1))
    
    def __optRtoL(self, site):
        # optimize the site tensors in between from left to right
        mps = self.__mps
        H = self.__H
        N = self.N
        if not mps.isGaugeCenter(site):
            mps.normalize(site)
        left = self.left[site-1]
        right = self.right[site]
        # obtain effective hamiltonian first
        a, b, c = mps.get_dim(site)
        i, j, k, l = H.get_dim(site)
        left, right = np.reshape(left, (a, i, a)), np.reshape(right, (c, j, c))
        Heff = np.einsum('gim,ijkl,njq->gknmlq', left, H.get_data(site), right, optimize=True)
        Heff = np.reshape(Heff, (a*b*c, a*b*c))
        # do eigen decomposition
        local_gs_mps = self.__local_gs(Heff, site)
        mps.replace(local_gs_mps, site)
        mps.normalize_site_R(site)
        # calculate left environment for following operations
        self.right[site-1] = mps.transfer_matrix(site, H.get_data(site))@np.reshape(right, (c*j*c, 1))
    
    def all_contract(self, isReal = True):
        mps = self.__mps
        H = self.__H
        transfer_matrix_list = [mps.transfer_matrix(x, H.get_data(x)) for x in range(0, self.N)]
        result = matprod(transfer_matrix_list)[0][0]
        return result.real if isReal else result

    def sweep(self):
        mps = self.__mps
        iter = 0
        energy = [self.all_contract()/mps.inner_product(real = True)]
        while iter <= self.max_iter:
            self.__optL()
            for i in range(1, self.N-1):
                self.__optLtoR(i)
            self.__optR()
            for j in range(self.N-2, 0, -1):
                self.__optRtoL(j)
            energy.append(self.all_contract()/mps.inner_product(real = True))
            if abs(energy[iter+1]-energy[iter]) < self.delta_E:
                break
            else:
                iter += 1
        self.gs_energy = energy[-1].real
        mps.unit()
        self.ground_state = mps
        return energy

class twoDMRG(object):
    def __init__(self, ):
        """
        The implementation of the two site DMRG algorithm. 
        The two site DMRG can adaptively increase the bond dimension, 
        which at most times leads to more accurate result. 
        """