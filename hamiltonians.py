'''
introducion: this package is created for saving Hamiltonians 
            of popular models, such as Ising model, Heisenberg model, 
            Hubbard model, t-J model, etc.
'''

import numpy as np
from .mpo import MPO
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
        for _ in range(1, N-1):
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
        for _ in range(1, N-1):
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
    def __init__(self, N: int ,omega_B: float, omega: float, g: list, lamda: list, Nc: int):
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
        self.mpo = self.__initialize()
    
    def __initialize(self):
        N = self.N
        w_B = self.omega_B
        w_j = self.omega
        g = self.g
        lamda = self.lamda
        H = []
        # first tensor is  (I, ga, ga+, waa+)
        Nc = self.Nc
        H.append(np.array([eye(Nc+1), annihilate(Nc+1), create(Nc+1), w_B*create(Nc+1)@annihilate(Nc+1)]))
        # second tensor is
        # (I, lambda*sigma+, lambda*sigma-, O, O, wn)
        # (O, O, O, I, O, sigma+)
        # (O, O, O, O, I, sigma-)
        # (O, O, O, O, O, I)
        local_H = np.array([[eye(), lamda[0]*raising(), lamda[0]*lowering(), empty(), empty(), w_j*n()], 
                            [empty(), empty(), empty(), eye(), empty(), g[0]*raising()], 
                            [empty(), empty(), empty(), empty(), eye(), g[0]*lowering()],
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
                                [empty(), empty(), empty(), eye(), empty(), g[i-1]*raising()], 
                                [empty(), empty(), empty(), empty(), eye(), g[i-1]*lowering()],
                                [empty(), empty(), empty(), empty(), empty(), eye()]])
            H.append(local_H)
        # last tensor is (wn, sigma-, sigma+, sigma+, sigma-, I)^T
        H.append(np.array([w_j*n(), lowering(), raising(), g[N-1]*raising(), g[N-1]*lowering(), eye()]))
        return H

# Rydberg system Hamiltonian
class RydbergH(MPO):
    # H_{Rydberg} = \Omega(t)/2\sum_i \sigma_x^i-\sum_i \Delta_i(t) n_i
    # +\sum_{i<j} V/|i-j|^6 n_i n_j
    def __init__(self, N: int, Omega: float, Delta: float, V: float, delta: list = 0):
        '''
        @N: number of atoms
        @Omega: effective coupling strength of two photon transition
        @Delta: detuning
        @delta: light shift, a list
        @V: interaction strength of two Rydberg atoms on neighboring sites
        '''
        super(MPO, self).__init__()
        self.N = N
        self.Omega = Omega
        self.Delta = Delta
        self.V = V
        self.delta = [0]*N if delta == 0 else delta
        self.chi = [0.041288, 31.45, 51.968377, 3.7e-8]
        self.lamda = [0.3071715, 0.0109042, 0.0123995, 0.065121788]
        self.mpo = self.__initialize()
    
    def __initialize(self):
        '''
        no extra parameters is needed.
        '''
        N = self.N
        Omega = self.Omega
        Delta = self.Delta
        delta = self.delta
        V = self.V
        chi = self.chi
        lamda = self.lamda
        one_body_oper = [sigmax(Omega/2)-(Delta+delta[i])*n() for i in range(N)]
        H = []
        # first tensor
        H.append(np.array([eye(), V*chi[0]*n(), V*chi[1]*n(), V*chi[2]*n(), V*chi[3]*n(), one_body_oper[0]]))
        # tensors in between
        for k in range(1, N-1):
            local_H = np.array([[eye(), V*chi[0]*n(), V*chi[1]*n(), V*chi[2]*n(), V*chi[3]*n(), one_body_oper[k]], 
                                [empty(), lamda[0]*eye(), empty(), empty(), empty(), lamda[0]*n()], 
                                [empty(), empty(), lamda[1]*eye(), empty(), empty(), lamda[1]*n()], 
                                [empty(), empty(), empty(), lamda[2]*eye(), empty(), lamda[2]*n()], 
                                [empty(), empty(), empty(), empty(), lamda[3]*eye(), lamda[3]*n()], 
                                [empty(), empty(), empty(), empty(), empty(), eye()]])
            H.append(local_H)
        # last tensor
        H.append(np.array([one_body_oper[N-1], lamda[0]*n(), lamda[1]*n(), lamda[2]*n(), lamda[3]*n(), eye()]))
        return H

# Rydberg system Hamiltonian with two different type atom
class Rydberg2H(MPO):
    # H_{Rydberg2} = \Omega(t)/2\sum_i \sigma_x^i - \sum_i \Delta_i(t) n_i
    # + \sum_{i<j, aa type} Vaa/|i-j|^6 n_i n_j + \sum_{i<j, bb type} Vbb/|i-j|^6 n_i n_j
    # + \sum_{i<j, ab type} Vab/|i-j|^6 n_i n_j
    def __init__(self, N: int, Omega: float, Delta: float, Vaa: float, Vbb: float, 
                 Vab: float, delta: list=0, atom0: str='a'):
        """
        @N: number of atoms
        @Omega: effective coupling strength of two photon transition
        @Delta: detuning
        @Vaa: interaction strength of two Rydberg atoms between type a atoms
        @Vbb: interaction strength of two Rydberg atoms between type b atoms
        @Vab: interaction strength of two Rydberg atoms between type a atom and type b atom
        @delta: light shift, a list
        @atom0: type of first atom, 'a' or 'b'

        Here, we can choose the type of the first atom.
        Once the type of the first atom is chosen, the construction 
        of the Hamiltonian follows the rule below:
        a--b--a--b--a--b--a--b--... if first atom is of type a
        b--a--b--a--b--a--b--a--... if first atom is of type b
        N (number of atoms) will decide the type of the last atom automatically.
        """
        super(MPO, self).__init__()
        self.N = N
        self.Omega = Omega
        self.Delta = Delta
        self.Vaa, self.Vbb, self.Vab = Vaa, Vbb, Vab
        self.delta = [0]*N if delta == 0 else delta
        #self.chi = [0.048026252570467734, 31.435899181574143, 51.74232772401499, 4.048868348332189e-08]
        #self.lamda = [0.28778490730578843, 0.01096325868136576, 0.01240008732747094, -0.03374130145699987]
        self.chi = [0.041288, 31.45, 51.968377, 3.7e-8]
        self.lamda = [0.3071715, 0.0109042, 0.0123995, 0.065121788]
        self.mpo = self.__initialize(atom0)
    
    def __initialize(self, atom0='a'):
        '''
        no extra parameters is needed.
        '''
        N = self.N
        Omega = self.Omega
        Delta = self.Delta
        delta = self.delta
        if atom0 == 'a':
            Vaa, Vbb, Vab = self.Vaa, self.Vbb, self.Vab
        elif atom0 == 'b':
            Vaa, Vbb, Vab = self.Vbb, self.Vaa, self.Vab
        else:
            raise Exception('atom type error')
        V1, V2 = Vab/Vaa, Vab/Vbb
        chi = self.chi
        lamda = self.lamda
        one_body_oper = [sigmax(Omega/2)-(Delta+delta[i])*n() for i in range(N)]
        H = []
        # first tensor
        H.append(np.array([eye(), Vaa*chi[0]*n(), Vaa*chi[1]*n(), Vaa*chi[2]*n(), Vaa*chi[3]*n(), one_body_oper[0]]))
        # second tensor
        H.append(np.array([[eye(), empty(), empty(), empty(), empty(), Vbb*chi[0]*n(), Vbb*chi[1]*n(), Vbb*chi[2]*n(), Vbb*chi[3]*n(), one_body_oper[1]], 
                           [empty(), lamda[0]*eye(), empty(), empty(), empty(), empty(), empty(), empty(), empty(), V1*lamda[0]*n()],
                           [empty(), empty(), lamda[1]*eye(), empty(), empty(), empty(), empty(), empty(), empty(), V1*lamda[1]*n()], 
                           [empty(), empty(), empty(), lamda[2]*eye(), empty(), empty(), empty(), empty(), empty(), V1*lamda[2]*n()], 
                           [empty(), empty(), empty(), empty(), lamda[3]*eye(), empty(), empty(), empty(), empty(), V1*lamda[3]*n()], 
                           [empty(), empty(), empty(), empty(), empty(), empty(), empty(), empty(), empty(), eye()]]))
        # tensor in between
        for k in range(2, N-1):
            if k%2 == 0:
                local_H = np.array([[eye(), Vaa*chi[0]*n(), Vaa*chi[1]*n(), Vaa*chi[2]*n(), Vaa*chi[3]*n(), empty(), empty(), empty(), empty(), one_body_oper[k]], 
                                    [empty(), lamda[0]*eye(), empty(), empty(), empty(), empty(), empty(), empty(), empty(), lamda[0]*n()],
                                    [empty(), empty(), lamda[1]*eye(), empty(), empty(), empty(), empty(), empty(), empty(), lamda[1]*n()], 
                                    [empty(), empty(), empty(), lamda[2]*eye(), empty(), empty(), empty(), empty(), empty(), lamda[2]*n()], 
                                    [empty(), empty(), empty(), empty(), lamda[3]*eye(), empty(), empty(), empty(), empty(), lamda[3]*n()], 
                                    [empty(), empty(), empty(), empty(), empty(), lamda[0]*eye(), empty(), empty(), empty(), V2*lamda[0]*n()], 
                                    [empty(), empty(), empty(), empty(), empty(), empty(), lamda[1]*eye(), empty(), empty(), V2*lamda[1]*n()], 
                                    [empty(), empty(), empty(), empty(), empty(), empty(), empty(), lamda[2]*eye(), empty(), V2*lamda[2]*n()], 
                                    [empty(), empty(), empty(), empty(), empty(), empty(), empty(), empty(), lamda[3]*eye(), V2*lamda[3]*n()], 
                                    [empty(), empty(), empty(), empty(), empty(), empty(), empty(), empty(), empty(), eye()]])
                H.append(local_H)
            elif k%2 == 1:
                local_H = np.array([[eye(), empty(), empty(), empty(), empty(), Vbb*chi[0]*n(), Vbb*chi[1]*n(), Vbb*chi[2]*n(), Vbb*chi[3]*n(), one_body_oper[k]], 
                                    [empty(), lamda[0]*eye(), empty(), empty(), empty(), empty(), empty(), empty(), empty(), V1*lamda[0]*n()], 
                                    [empty(), empty(), lamda[1]*eye(), empty(), empty(), empty(), empty(), empty(), empty(), V1*lamda[1]*n()], 
                                    [empty(), empty(), empty(), lamda[2]*eye(), empty(), empty(), empty(), empty(), empty(), V1*lamda[2]*n()], 
                                    [empty(), empty(), empty(), empty(), lamda[3]*eye(), empty(), empty(), empty(), empty(), V1*lamda[3]*n()], 
                                    [empty(), empty(), empty(), empty(), empty(), lamda[0]*eye(), empty(), empty(), empty(), lamda[0]*n()], 
                                    [empty(), empty(), empty(), empty(), empty(), empty(), lamda[1]*eye(), empty(), empty(), lamda[1]*n()], 
                                    [empty(), empty(), empty(), empty(), empty(), empty(), empty(), lamda[2]*eye(), empty(), lamda[2]*n()], 
                                    [empty(), empty(), empty(), empty(), empty(), empty(), empty(), empty(), lamda[3]*eye(), lamda[3]*n()], 
                                    [empty(), empty(), empty(), empty(), empty(), empty(), empty(), empty(), empty(), eye()]])
                H.append(local_H)
            else:
                raise Exception("index error!")
        
        if N%2 == 0:
            H.append(np.array([one_body_oper[N-1], V1*lamda[0]*n(), V1*lamda[1]*n(), V1*lamda[2]*n(), V1*lamda[3]*n(), 
                               lamda[0]*n(), lamda[1]*n(), lamda[2]*n(), lamda[3]*n(), eye()]))
        elif N%2 == 1:
            H.append(np.array([one_body_oper[N-1], lamda[0]*n(), lamda[1]*n(), lamda[2]*n(), lamda[3]*n(), 
                               V2*lamda[0]*n(), V2*lamda[1]*n(), V2*lamda[2]*n(), V2*lamda[3]*n(), eye()]))
        else:
            raise Exception("last index error!")
        return H

class PXPH(MPO):
    # H_{PXP} = \sum_i P_{i-1}X_{i}P_{i+1}
    def __init__(self, N):
        '''
        @N: number of particles, i.e., length of list
        '''
        super(MPO, self).__init__()
        self.N = N
        self.mpo = self.__initialize()

    def __initialize(self):
        '''
        no extra parameters is needed.
        '''
        N = self.N
        # the projector to ground state, (I-Z)/2
        P = (eye() - sigmaz())/2
        H = []
        # first tensor is (I, P, X)
        H.append(np.array([eye(), P, sigmax()]))
        # second tensor is 
        # (I, P, O, O)
        # (O, O, X, O)
        # (O, O, O, P)
        H.append(np.array([[eye(), P, empty(), empty()], 
                           [empty(), empty(), sigmax(), empty()], 
                           [empty(), empty(), empty(), P]]))
        # tensors in between are
        # (I, P, O, O)
        # (O, O, X, O)
        # (O, O, O, P)
        # (O, O, O, I)
        for _ in range(2, N-2):
            local_H = np.array([[eye(), P, empty(), empty()], 
                                [empty(), empty(), sigmax(), empty()], 
                                [empty(), empty(), empty(), P], 
                                [empty(), empty(), empty(), eye()]])
            H.append(local_H)
        # second last tensor is 
        # (P, O, O)
        # (O, X, O)
        # (O, O, P)
        # (O, O, I)
        H.append(np.array([[P, empty(), empty()], 
                           [empty(), sigmax(), empty()], 
                           [empty(), empty(), P], 
                           [empty(), empty(), eye()]]))
        # last tensor id (X, P, I)^T
        H.append(np.array([sigmax(), P, eye()]))
        return H

class PIXIPH(MPO):
    # H_{PIXIP} = \sum_i P_{i-2}I_{i-1}X_{i}I_{i+1}P_{i+2}
    def __init__(self, N: int, Omega: float = 2):
        """
        @N: number of particles(sites)
        """
        super(MPO, self).__init__()
        self.N = N
        self.Omega = Omega
        self.mpo = self.__initialize()
    
    def __initialize(self):
        '''
        no extra parameters is needed.
        '''
        N = self.N
        # projector to ground state, (I-Z)/2
        P = (eye() - sigmaz())/2
        I, X, O = eye(), sigmax(self.Omega/2), empty()
        H = []
        # first tensor is (I, P, I, X)
        H.append(np.array([I, P, I, X]))
        # second tensor
        H.append(np.array([[I, P, O, O, O], 
                           [O, O, I, O, O], 
                           [O, O, O, X, O], 
                           [O, O, O, O, I]]))
        # third tensor
        H.append(np.array([[I, P, O, O, O, O], 
                            [O, O, I, O, O, O], 
                            [O, O, O, X, O, O], 
                            [O, O, O, O, I, O], 
                            [O, O, O, O, O, P]]))
        # tensors in between
        for _ in range(3, N-3):
            local_H = np.array([[I, P, O, O, O, O], 
                                [O, O, I, O, O, O], 
                                [O, O, O, X, O, O], 
                                [O, O, O, O, I, O], 
                                [O, O, O, O, O, P], 
                                [O, O, O, O, O, I]])
            H.append(local_H)
        # last three tensor 
        H.append(np.array([[P, O, O, O, O], 
                           [O, I, O, O, O], 
                           [O, O, X, O, O], 
                           [O, O, O, I, O], 
                           [O, O, O, O, P], 
                           [O, O, O, O, I]]))
        H.append(np.array([[I, O, O, O], 
                           [O, X, O, O], 
                           [O, O, I, O], 
                           [O, O, O, P], 
                           [O, O, O, I]]))
        H.append(np.array([X, I, P, I]))
        return H
        



