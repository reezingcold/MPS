# MPS package
python package for doing 1D quantum (spin) system calculations via tensor network state (matrix product states)

# Language and dependencies
This package is purely written in [python](https://www.python.org) and it depends on the following packages:
* [numpy](https://numpy.org)
* [scipy](https://scipy.org)

It is also recommended to install [qutip](https://qutip.org) and [matplotlib](https://matplotlib.org) for future convenience.

# This package can do...
This package contains the implementation of various algorithms based on matrix product states(MPS).
## Ground state search(vMPS, one site DMRG)
One of the most important properties of a quantum system is its ground state. 
Hamiltonians of most quantum systems are very local, such as Ising model, Heisenberg model, etc. (only nearest neighbouring terms). 
Therefore, in these models, ground states are not highly entangled, which implies they can be 
efficiently saved and calculated via MPS. 

We first write the Hamitonian in the form of matrix product operators(MPO), after which the ground state search algorithm 
can be applied. Here is a simple example. 
```python
from mps.mps import MPS
from mps.hamiltonians import HeisenbergH
from mps.ground_search import vFindGS

N = 10 # site number or qubit number
Jx, Jy, Jz = 2, -8, 2 # parameters in Hamiltonian
bond_dim_max = 8 # bond dimension constraint
gs_search = vFindGS(MPS(N, bond_dim_max), HeisenbergH(N, Jx, Jy, Jz), max_iter = 20, delta_E = 1e-4)
gs_search.sweep()
gs_energy, gs = gs_search.result
```
The above code is to find the ground state and ground energy of the Heisenberg Hamiltonian:
$$\hat{H} = J_x\sum_{i=1}^N\sigma^x_i\sigma^x_{i+1}+J_y\sum_{i=1}^N\sigma^y_i\sigma^y_{i+1}+J_z\sum_{i=1}^N\sigma^z_i\sigma^z_{i+1}$$
where $J_x=2, J_y=-8, J_z=2$, and the qubit(site) number $N = 10$.

The `gs_energy` is the ground state energy and `gs` is the ground state in the form of MPS.
## Time evolution
Dynamics of a quantum system is also where we are interested. However, it is usually very difficult 
to solve the time evolution of the quantum system, even in the language of tensor network state, especially 
in the case of a quantum quench. 

Here, we realize two algorithms of evolving a quantum system. 
### MPO time evolution
The idea of this method is rather naive. We just do the what the following equation tells
$$|\psi(t+\delta t)\rangle = e^{-iH\delta t}|\psi(t)\rangle\approx(1-iH\delta t)|\psi(t)\rangle$$
The basic procedure is exactly what you are thinking.

![MPOtimeevo](https://github.com/reezingcold/MPS/blob/main/pics/MPOtimeevo.png)

* Apply MPO on the MPS with being multiplied by $-i\delta t$
* Add two MPS $|\psi(t)\rangle+(-i\delta t)H|\psi\rangle$
* Do SVD to reduce the bond dimension. (neglecting small singular values)

The cost of the simplicity of this method is its accuracy. The MPO time evolution method needs very small 
time segment $\delta t$ to reach high accuracy, which is not efficient in most cases.

### TDVP method
<img src="https://github.com/reezingcold/MPS/blob/main/pics/tdvp.png" width="50%">
TDVP stands for time dependent variational principle. The idea of this method is 
to evolve the given MPS with its bond dimension fixed. This method straightfowardly proposes the 
time evolution equation of a MPS when bond dimension is fixed.

$$i\frac{\mathrm{d}|\psi\rangle}{\mathrm{d}t} = \mathcal{P}\hat{H}|\psi\rangle$$
The above equation is the TDVP equation. $\mathcal{P}$ is the projector to the manifold of fixed bond dimension.

The above TDVP method is often called as one site TDVP, because it can't change bond dimension dynamically. 
If you start at an MPS with very small bond dimension, the result may be very inaccurate.
So we usually have to increase bond dimension in the begining, where the MPO time evolution method 
plays an important role. 
However, there is another better way to solve this problem, the two site TDVP method.
The two site TDVP can dynamically increase the bond dimension because it evolves two 
neighbouring site tensors together. 
In most cases, the two site TDVP method can give more accurate result comparing to 
MPO time evolution method before the one site TDVP is implemented.

Here is a simple example of the time evolution of Heisenberg system.
```python
import numpy as np
import qutip as qt
from mps.mps import MPS
from mps.mps_functions import multiply
from mps.hamiltonians import HeisenbergH
import matplotlib.pyplot as plt
from mps.tdvp import oneTDVP, twoTDVP

T = 10
n = 1000
dt = T/n

# Heisenberg model
N = 8
Jx = 0.9
Jy = -0.5
Jz = 0.6

# Hamiltonian, matrix form
Hz = 0
for i in range(N-1):
    cup=[qt.qeye(2)]*N
    cup[i],cup[i+1]=qt.sigmaz(),qt.sigmaz()
    Hz+=qt.tensor(cup)

Hx = 0
for j in range(N-1):
    cup=[qt.qeye(2)]*N
    cup[j],cup[j+1]=qt.sigmax(),qt.sigmax()
    Hx+=qt.tensor(cup)

Hy = 0
for k in range(N-1):
    cup=[qt.qeye(2)]*N
    cup[k],cup[k+1]=qt.sigmay(),qt.sigmay()
    Hy+=qt.tensor(cup)

H = Jx*Hx+Jy*Hy+Jz*Hz

# initial state
zero = qt.basis(2, 0)
one = qt.basis(2, 1)

initial_state = qt.tensor([zero]*N)
ed_result = []
state = initial_state
ed_result.append(abs((initial_state.dag()*state)[0][0][0])**2)
# time evolution for state vector
for i in range(0, n):
    state = (-1j*H*dt).expm()*state
    ed_result.append(abs((initial_state.dag()*state)[0][0][0])**2)
    print(i)

plt.plot(np.linspace(0,T,n+1), ed_result, color = 'orangered', label = "exact diagonalization result")

# create an initial state in mps form
mps_init = MPS(N, bond_D = 8)
mps_init.replace(np.array([[1],[0]]), 0)
for i in range(1, N-1):
    mps_init.replace(np.array([[[1],[0]]]), i)
mps_init.replace(np.array([[1, 0]]) ,N-1)
mps_init.update()

# mpo Hamiltonian of Heisenberg model
def Ham(t):
    return HeisenbergH(N, Jx, Jy, Jz)

mps_result = []

# time evolution for mps
timeevo = twoTDVP(mps_init, Ham, 0, dt, 32)
for t in range(0, 4):
    mps_result.append(abs(multiply(mps_init.dag(), timeevo.current_state))**2)
    timeevo.evolve_a_step()
timeevo2 = oneTDVP(timeevo.current_state, Ham, 3*dt, dt)
for tt in range(4, n+1):
    mps_result.append(abs(multiply(mps_init.dag(), timeevo2.current_state))**2)
    timeevo2.evolve_a_step()
    print(tt)

plt.scatter(np.linspace(0, T, n+1), mps_result, label = "TDVP result", marker='.')
plt.ylabel(r"$|\langle \psi(0)|\psi(t)\rangle|^2$")
plt.xlabel(r"time")
plt.legend()
plt.show()
```
The above code calculates the time evolution of a quantum Heisenberg system with initial state being $|0\rangle^{\otimes 8}$. 
The result is shown below.
![tdvptimeevoresult](https://github.com/reezingcold/MPS/blob/main/pics/tdvp_result.png)
# Reference
* The first paper about DMRG algorithm by S. R. White: [Density matrix formulation for quantum renormalization groups](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.69.2863)
* Matrix product states paper by G. Vidal: [Efficient classical simulation of slightly entangled quantum computations](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.91.147902)
* TDVP time evolution paper by Jutho Haegeman, et al.: [Time-dependent variational principle for quantum lattices](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.070601)
* Detailed one/two site TDVP algorithm paper by Jutho Haegeman, et al.: [Unifying time evolution and optimization with matrix product states](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.165116)
* Review of matrix product states and relative algorithms by Ulrich Schollwock: [The density-matrix renormalization group in the age of matrix product states](https://www.sciencedirect.com/science/article/abs/pii/S0003491610001752?via%3Dihub)
