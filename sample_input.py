import numpy as np
from rlci.solvers import RL

np.random.seed(1)

# generate random symmetric matrix
NDIM = 100
A = np.random.rand(NDIM, NDIM)
A = A + A.T

# Or, load a sample Hamiltonian
#A = np.loadtxt('full_hamiltonians/h6_1p00_ring.txt')

k = 40  # sparsity
print("k:    ", k)
print("NDet: ", len(A))

E_exact, s = RL(A, k, mode='full')
print("exact:  ", E_exact)

E_apsci, s = RL(A, k, mode='apsci')
print("ap-sCI: ", E_apsci)

E_greedy, s = RL(A, k, mode='greedy')
print("greedy: ", E_greedy)

E_rl, s = RL(A, k, mode='rl', max_pick=50)
print("RLCI:   ", E_rl)
