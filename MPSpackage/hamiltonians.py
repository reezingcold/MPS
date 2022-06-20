'''
author: liu leiyinan
create date: 2021/12/6
last change: 2021/12/21
introducion: this package is created for saving Hamiltonians 
            of popular models, such as Ising model, Heisenberg model, 
            Hubbard model, t-J model, etc.
'''

import numpy as np
from .mpo import *
from .tool_functions import *

class IsingH(MPO):
    # H_{Ising} = J\sum_{i}\sigma^z_i\sigma^z_{i+1}+\mu\sum_i\sigmax_i
    def __init__(self, N, J, mu):
        '''
        @N: number of particles, i.e., length of list
        @J: coupling strength, real number
        @mu: outfield strength, real number
        '''
        super(MPO, self).__init__()
        self.N = N
        self.J = J
        self.mu = mu
        self.mpo = self.initialize()

    def initialize(self):
        '''
        no extra parameters is needed.
        '''
        N = self.N
        J = self.J
        mu = self.mu
        H = []
        # first tensor is (I, J*Z, mu*X)
        H.append(np.array([eye(), sigmaz(J), sigmax(mu)]))
        # tensors in between are the same, which has the following form:
        # (I    J*Z    mu*X)
        # (O     O      Z  )
        # (O     O      I  )
        for i in range(1, N-1):
            local_H = np.array([[eye(), sigmaz(J), sigmax(mu)], 
                                [empty(), empty(), sigmaz()], 
                                [empty(), empty(), eye()]])
            H.append(local_H)
        # last tensor is (mu*X, Z, I)^T
        H.append(np.array([sigmax(mu), sigmaz(), eye()]))
        return H

class HeisenbergH(MPO):
    # H_{Heisenberg} = \sum_i J_x*\sigma^x_i*sigma^x_{i+1})+J_y*(sigma^y_i*sigma^y_{i+1})
    #                   +J_z*(sigma^z_i*sigma^z_{i+1}
    def __init__(self, N, Jx, Jy, Jz):
        '''
        @N: number of particles
        @Jx: coupling strength of sigmaxes
        @Jy: coupling strength of sigmayes
        @Jz: coupling strength of sigmazes
        '''
        super(MPO, self).__init__()
        self.N = N
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.mpo = self.initialize()

    def initialize(self):
        N = self.N
        Jx = self.Jx
        Jy = self.Jy
        Jz = self.Jz
        H = []
        # first tensor is (I, Jx*X, Jy*Y, Jz*Z, O)
        H.append(np.array([eye(), sigmax(Jx), sigmay(Jy), sigmaz(Jz), empty()]))
        # tensors in between are the same, which has the following form:
        # (I   Jx*X   Jy*Y   Jz*Z   O)
        # (O     O      O      O    X)
        # (O     O      O      O    Y)
        # (O     O      O      O    Z)
        # (O     O      O      O    I)
        for i in range(1, N-1):
            local_H = np.array([[eye(), sigmax(Jx), sigmay(Jy), sigmaz(Jz), empty()],
                                [empty(), empty(), empty(), empty(), sigmax()],
                                [empty(), empty(), empty(), empty(), sigmay()],
                                [empty(), empty(), empty(), empty(), sigmaz()],
                                [empty(), empty(), empty(), empty(), eye()]])
            H.append(local_H)
        # last tensor is (O, Jx, Jy, Jz, I)^T
        H.append(np.array([empty(), sigmax(), sigmay(), sigmaz(), eye()]))
        return H

class SuperConductingH(MPO):
    def __init__(self, N: int ,omega_B: float, omega: float, g: float, lamda: list, Nc):
        '''
        @N: number of qubit
        @omega_B: frequency of cavity
        @omega: frequency of qubit, same for all qubits
        @g: coupling strength between qubit and cavity
        @lamda: crosstalk strength
        @Nc: truncation number of cavity
        '''
        super(MPO, self).__init__()
        self.N = N
        self.omega_B = omega_B
        self.omega = omega
        self.g = g
        self.lamda = lamda
        self.Nc = Nc
        self.mpo = self.initialize()
    
    def initialize(self):
        N = self.N
        w_B = self.omega_B
        w_j = self.omega
        g = self.g
        lamda = self.lamda
        H = []
        # first tensor is  (I, ga, ga+, waa+)
        Nc = self.Nc
        H.append(np.array([eye(Nc+1), g*annihilate(Nc+1), g*create(Nc+1), w_B*create(Nc+1)@annihilate(Nc+1)]))
        # second tensor is
        # (I, lambda*sigma+, lambda*sigma-, O, O, wn)
        # (O, O, O, I, O, sigma+)
        # (O, O, O, O, I, sigma-)
        # (O, O, O, O, O, I)
        local_H = np.array([[eye(), lamda[0]*raising(), lamda[0]*lowering(), empty(), empty(), w_j*n()], 
                            [empty(), empty(), empty(), eye(), empty(), raising()], 
                            [empty(), empty(), empty(), empty(), eye(), lowering()],
                            [empty(), empty(), empty(), empty(), empty(), eye()]])
        H.append(local_H)
        # tensors in between are the same(except lamda), which has the following form:
        # (I, lambda*sigma+, lambda*sigma-, O, O, wn)
        # (O, O, O, O, O, sigma-)
        # (O, O, O, O, O, sigma+)
        # (O, O, O, I, O, sigma+)
        # (O, O, O, O, I, sigma-)
        # (O, O, O, O, O, I)
        for i in range(2, N):
            local_H = np.array([[eye(), lamda[i-1]*raising(), lamda[i-1]*lowering(), empty(), empty(), w_j*n()], 
                                [empty(), empty(), empty(), empty(), empty(), lowering()], 
                                [empty(), empty(), empty(), empty(), empty(), raising()],
                                [empty(), empty(), empty(), eye(), empty(), raising()], 
                                [empty(), empty(), empty(), empty(), eye(), lowering()],
                                [empty(), empty(), empty(), empty(), empty(), eye()]])
            H.append(local_H)
        # last tensor is (wn, sigma-, sigma+, sigma+, sigma-, I)^T
        H.append(np.array([w_j*n(), lowering(), raising(), raising(), lowering(), eye()]))
        return H

        
        

