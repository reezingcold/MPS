'''
author: liu leiyinan
create date: 2022/6/6
last change: 
introducion: this package is created for calculating ground state and energy
             of a given Hamiltonian based on MPS and MPO. There are huge improvements 
             compared to "ground_energy.py"
'''

import numpy as np
from numpy import linalg
from .mps import MPS
from .mpo import MPO
from .tool_functions import *

class vFindGS(object):
    def __init__(self, mps: MPS, H: MPO, max_iter = 10, delta_E = 1e-3):
        '''
        @mps: the initial state, given in the form of mps
        @H: the hamiltonian, given in the form of mpo
        @max_iter: the upper limit of sweep times
        @delta_E: stop the iteration(sweep) if (E_before-E_after) < delta_E
        '''
        self.__mps = mps
        self.__H = H
        self.max_iter = max_iter
        self.delta_E = delta_E
        self.cup = [0]*mps.N
        if mps.N == H.N:
            self.N = mps.N
        else:
            raise Exception("length doesn' t cooperate.")
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
            return (self.gs_energy, self.ground_state)
    
    @property
    def H(self):
        return self.__H
    
    def optL(self):
        # optimize the very left side site tensor
        mps = self.__mps
        H = self.__H
        N = self.N
        if mps.isGaugeCenter(0) == True:
            pass
        else:
            mps.normalize(0)
        # obtain effective hamiltonian first
        right = self.right[0]
        a, b = mps.get_dim(0)
        i, j, k = H.get_dim(0)
        right = np.reshape(right, (b, i, b))
        Heff = np.einsum('ijk,eig->jekg', H.get_data(0), right)
        Heff = np.reshape(Heff, (a*b, a*b))
        # do eigen decomposition
        eigresult = linalg.eig(Heff)
        eigvals = [each.real for each in eigresult[0]]
        local_gs_site = eigvals.index(min(eigvals))
        local_gs = eigresult[1][:,local_gs_site]
        local_gs_mps = np.reshape(local_gs, (a, b))
        mps.replace(local_gs_mps, 0)
        mps.normalize_site_L(0)
        # calculate left environment for following operations
        self.left[0] = mps.transfer_matrix(0, H.get_data(0))
    
    def optLtoR(self, site):
        # optimize the site tensors in between from left to right
        mps = self.__mps
        H = self.__H
        N = self.N
        if mps.isGaugeCenter(site) == True:
            pass
        else:
            mps.normalize(site)
        left = self.left[site-1]
        right = self.right[site]
        # obtain effective hamiltonian first
        a, b, c = mps.get_dim(site)
        i, j, k, l = H.get_dim(site)
        left, right = np.reshape(left, (a, i, a)), np.reshape(right, (c, j, c))
        Heff = np.einsum('gim,ijkl,njq->gknmlq', left, H.get_data(site), right)
        Heff = np.reshape(Heff, (a*b*c, a*b*c))
        # do eigen decomposition
        eigresult = linalg.eig(Heff)
        eigvals = [each.real for each in eigresult[0]]
        local_gs_site = eigvals.index(min(eigvals))
        local_gs = eigresult[1][:,local_gs_site]
        local_gs_mps = np.reshape(local_gs, (a, b, c))
        mps.replace(local_gs_mps, site)
        mps.normalize_site_L(site)
        # calculate left environment for following operations
        self.left[site] = np.reshape(left, (1, a*i*a))@mps.transfer_matrix(site, H.get_data(site))
    
    def optR(self):
        # optimize the site tensors in between from left to right
        mps = self.__mps
        H = self.__H
        N = self.N
        if mps.isGaugeCenter(N-1) == True:
            pass
        else:
            mps.normalize(N-1)
        left = self.left[N-2]
        # obtain effective hamiltonian first
        a, b = mps.get_dim(N-1)
        i, j, k = H.get_dim(N-1)
        left = np.reshape(left, (a, i, a))
        Heff = np.einsum('eig,ijk->ejgk', left, H.get_data(N-1))
        Heff = np.reshape(Heff, (a*b, a*b))
        # do eigen decomposition
        eigresult = linalg.eig(Heff)
        eigvals = [each.real for each in eigresult[0]]
        local_gs_site = eigvals.index(min(eigvals))
        local_gs = eigresult[1][:,local_gs_site]
        local_gs_mps = np.reshape(local_gs, (a, b))
        mps.replace(local_gs_mps, N-1)
        mps.normalize_site_R(N-1)
        # calculate right environment for following operations
        self.right[N-2] = mps.transfer_matrix(N-1, H.get_data(N-1))
    
    def optRtoL(self, site):
        # optimize the site tensors in between from left to right
        mps = self.__mps
        H = self.__H
        N = self.N
        if mps.isGaugeCenter(site) == True:
            pass
        else:
            mps.normalize(site)
        left = self.left[site-1]
        right = self.right[site]
        # obtain effective hamiltonian first
        a, b, c = mps.get_dim(site)
        i, j, k, l = H.get_dim(site)
        left, right = np.reshape(left, (a, i, a)), np.reshape(right, (c, j, c))
        Heff = np.einsum('gim,ijkl,njq->gknmlq', left, H.get_data(site), right)
        Heff = np.reshape(Heff, (a*b*c, a*b*c))
        # do eigen decomposition
        eigresult = linalg.eig(Heff)
        eigvals = [each.real for each in eigresult[0]]
        local_gs_site = eigvals.index(min(eigvals))
        local_gs = eigresult[1][:,local_gs_site]
        local_gs_mps = np.reshape(local_gs, (a, b, c))
        mps.replace(local_gs_mps, site)
        mps.normalize_site_R(site)
        # calculate left environment for following operations
        self.right[site-1] = mps.transfer_matrix(site, H.get_data(site))@np.reshape(right, (c*j*c, 1))
    
    def all_contract(self, real = True):
        mps = self.__mps
        H = self.__H
        transfer_matrix_list = [mps.transfer_matrix(x, H.get_data(x)) for x in range(0, self.N)]
        result = matprod(transfer_matrix_list)[0][0].real
        return result

    def sweep(self):
        mps = self.__mps
        mps.initialize('rc')
        iter = 0
        energy = [self.all_contract()/mps.inner_product(real = True)]
        while iter <= self.max_iter:
            self.optL()
            for i in range(1, self.N-1):
                self.optLtoR(i)
            self.optR()
            for j in range(self.N-2, 0, -1):
                self.optRtoL(j)
            energy.append(self.all_contract()/mps.inner_product(real = True))
            if abs(energy[iter+1]-energy[iter]) < self.delta_E:
                break
            else:
                iter += 1
        self.gs_energy = energy[-1].real
        mps.normalization()
        self.ground_state = mps
        return energy