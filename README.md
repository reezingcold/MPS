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
from mps.dmrg import oneDMRG

N = 20 # site number or qubit number
Jx, Jy, Jz = 1, -1, 1 # parameters in Hamiltonian
bond_dim_max = 8 # bond dimension constraint
predict_state = MPS(N, bond_dim_max) # initial state
H = HeisenbergH(N, Jx, Jy, Jz) # model Hamiltonian
gs_search = oneDMRG(predict_state, H, max_iter = 20, delta_E = 1e-4)
gs_search.sweep() # run DMRG
gs_energy, gs = gs_search.result # get result
print(gs_energy)
```
The above code is to find the ground state and ground energy of the Heisenberg Hamiltonian:
$$\hat{H} = J_x\sum_{i=1}^N\sigma^x_i\sigma^x_{i+1}+J_y\sum_{i=1}^N\sigma^y_i\sigma^y_{i+1}+J_z\sum_{i=1}^N\sigma^z_i\sigma^z_{i+1}$$
where $J_x=1, J_y=-1, J_z=1$, and the qubit(site) number $N = 20$.

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

Here is a simple example of the time evolution of the PXP model.
The PXP model Hamiltonian reads

$$ \hat{H} = \sum_{i}\hat{P}_{i-1}\hat{X}_i \hat{P}_{i+1} $$

where $\hat{P} = |0\rangle\langle 0|$ is a projector and $\hat{X} = |0\rangle\langle 1|+|1\rangle\langle0|$ is simply 
the Pauli X matrix.
The PXP model reveals quantum many-body scars, which can be clearly seen in the revival of Loschmidt echo defined by
$$g(t) = |\langle\mathbb{Z}_2|e^{-iHt}|\mathbb{Z}_2\rangle|^2$$
where $|\mathbb{Z}_2\rangle$ is the Neel state:
$$|\mathbb{Z}_2\rangle = |10101010\cdots\rangle$$
```python
from mps.mps import MPS
from mps.hamiltonians import PXPH
from mps.mps_functions import multiply, getEntropy
from mps.tdvp import tdvp
import numpy as np
import matplotlib.pyplot as plt

def create_mps(initial_state: str)->MPS:
    temp = 'cb' + ''.join(['1' if x == '0' else '0' for x in initial_state])
    return MPS(len(initial_state), initial=temp)

N = 16
T, Tn = 30, 300
dt = T/Tn
t = np.linspace(0, T, Tn+1)
bonddim = 32

mpsi0 = create_mps('10'*(N//2))
mpoH = PXPH(N)

evo_params = {"t0": 0, "dt": 0.1, "use_lanczos": True, "to_normalize": False, 
              "bond_D": bonddim, "cut_off": 1.e-10, "switch_step": 10, "lanczos_dim": 4}

tdvp_engine = tdvp(mpsi0, lambda x: mpoH, evo_params, True)
echolst = [1]*(Tn+1)
entrolst = [0]*(Tn+1)
for i in range(1, Tn+1):
    tdvp_engine.run()
    echolst[i] = abs(multiply(mpsi0.dag(), tdvp_engine.current_state))**2
    entrolst[i] = getEntropy(tdvp_engine.current_state, N//2-1)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

fig, ax1 = plt.subplots(figsize = (7, 5))
ax1.set_xlabel(r"$\mathrm{time}$", fontsize = 15)
ax1.set_ylabel(r'$\mathrm{Echo}$', color = 'tab:blue', fontsize = 15)
ax1.plot(t, echolst, label = 'echo')
ax2 = ax1.twinx()
ax2.set_ylabel(r'$\mathrm{Entropy}$', color = 'tab:orange', fontsize = 15)
ax2.plot(t, entrolst, label = 'entropy', color = 'tab:orange')
ax1.set_ylim(0, 1)
ax1.set_xlim(0, T)
ax2.set_ylim(0, 3)
ax1.grid(ls = '--')
ax1.tick_params(labelsize = 15)
ax2.tick_params(labelsize = 15)
plt.savefig("PXP16.pdf")
plt.show()
```
The above code calculates the time evolution of the PXP model with initial state being $|1010\cdots\rangle$ with 
16 qubits in total.
The result is shown below.
![tdvptimeevoresult](https://github.com/reezingcold/MPS/blob/main/pics/pxp16.png)
# Reference
* The first paper about DMRG algorithm by S. R. White: [Density matrix formulation for quantum renormalization groups](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.69.2863)
* Matrix product states paper by G. Vidal: [Efficient classical simulation of slightly entangled quantum computations](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.91.147902)
* TDVP time evolution paper by Jutho Haegeman, et al.: [Time-dependent variational principle for quantum lattices](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.070601)
* Detailed one/two site TDVP algorithm paper by Jutho Haegeman, et al.: [Unifying time evolution and optimization with matrix product states](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.165116)
* Review of matrix product states and relative algorithms by Ulrich Schollwock: [The density-matrix renormalization group in the age of matrix product states](https://www.sciencedirect.com/science/article/abs/pii/S0003491610001752?via%3Dihub)
* PXP model introduction: [Quantum scarred eigenstates in a Rydberg atom chain: entanglement, breakdown of thermalization, and stability to perturbations](https://arxiv.org/pdf/1806.10933.pdf)
