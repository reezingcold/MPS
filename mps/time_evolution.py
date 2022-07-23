'''
introducion: this package is created for calculate the time evolution of a given initial state
             (MPS form) under a given Hamiltonian (MPO form).
'''

from .mps import *
from .mps_functions import *
from .tool_functions import *
from .mpo import *

class TimeEvolution(object):
    def __init__(self, initial_state: MPS, H: MPO, T: float, n: int, bond_D: int):
        '''
        @initial_state: initial state in the form of mps
        @H: a function which returns Hamiltonian over time
        @T: total evolution time
        @n: number of time steps
        @bond_D: upper limit of bond dimension
        '''
        self.initial_state = initial_state
        self.H = H
        self.T = T
        self.step_num = n
        self.bond_D = bond_D
        self.N = initial_state.site_number
        self.current_state = initial_state.copy()
        self.current_time = 0
        if self.N == H(0).site_number:
            pass
        else:
            raise Exception("state and Hamiltonian should have the same length.")
    
    # evolve a step, i.e., |psi(t)> --> |psi(t+dt)>
    def evolve_a_step(self):
        t = self.current_time
        dt = self.T/self.step_num
        current_state = self.current_state
        # zero order
        zero_order = current_state
        zero_order.update()
        # 1st order
        c_1 = -1j*dt
        c_2 = -0.5*dt**2
        c_3 = (-1j*dt)**3/6
        first_order = operate(self.H(t), zero_order)
        first_order.update()
        second_order = operate(self.H(t), first_order)
        second_order.update()
        third_order = operate(self.H(t), second_order)
        third_order.update()

        first_order.normalize_R(0)
        first_order.replace(first_order.get_data(0)*c_1, 0)
        second_order.normalize_R(0)
        second_order.replace(second_order.get_data(0)*c_2, 0)
        third_order.normalize_R(0)
        third_order.replace(third_order.get_data(0)*c_3, 0)

        new_state = accurate_plus(accurate_plus(accurate_plus(zero_order, first_order), second_order), third_order)
        new_state.update()
        new_state.bond_D = self.bond_D
        new_state.normalization()

        #new_state = accurate_plus(new_state, third_order)
        #new_state.bond_D = self.bond_D
        #new_state.normalization()

        self.current_time = t+dt
        self.current_state = new_state
        
