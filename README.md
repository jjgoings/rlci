[![Build Status](https://travis-ci.com/jjgoings/rlci.svg?branch=master)](https://travis-ci.com/jjgoings/rlci) 

# Reinforcement Learning Configuration Interaction

<p align="center">
<img src="/assets/rlci.png" width="400">
</p>

Here are some routines to do reinforcement learning on the FCI Hamiltonian in 
order to obtain near-optimal k-sparse approximations to symmetric matrices.

Basically, we are looking for the k by k submatrix that, when diagonalized, 
yields the best approximation to the lowest eigenvalue of the full matrix. This 
k-sparse approximation is obtained from the k rows and columns of the original
matrix. Since the original matrix should be Hermitian, the "best" approximate 
solution will be the submatrix with the lowest eigenvalue (e.g., variational 
theorem). The lower the minimum eigenvalue, the better the approximation.

This work is based on the pre-prints

```
Goings, Joshua, Hang Hu, Chao Yang, and Xiaosong Li. "Reinforcement Learning Configuration Interaction." ChemRxiv, 2021. doi:10.26434/chemrxiv.14342234.v2. 
```

and

```
Li Zhou, Lihao Yan, Mark A. Caprio, Weiguo Gao, Chao Yang. "Solving the k-sparse Eigenvalue Problem with Reinforcement Learning" arXiv, 2020. https://arxiv.org/abs/2009.04414.
```

as always, this is research code, so use at your own risk :)

## Installation

Once you've cloned the directory, you can just

```
python setup.py install
```

### Dependencies
You'll need `numpy` and `tqdm` (for the nice progess bar in RLCI). Easiest way is just with `pip` 

```
pip install numpy tqdm
```

### Testing
You can test the install with `nosetests`. In the head directory, just do

```
nosetests tests
```

it should take a few seconds, but everything should pass. You can uncomment out
the RLCI tests in the test directory, but because RLCI has stochastic components
it may not converge to exactly the same value on all machines / seeds.

## Running
Once you've installed, you can try running the input script `sample_input.py`:

```
python sample_input.py
```

which looks like:

```
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
```

This does a few methods, including RLCI, on a "fake" symmetric matrix. You can
also try loading in the sample full CI Hamiltonians from the `full_hamiltonians`
directory. You should see something like:

```
k:     40
NDet:  100
exact:   -8.077612478401324
ap-sCI:  -7.140567021103666
greedy:  -7.012956319532492
RLCI:    -7.17615434373654
```

The exact numbers will vary depending on your input matrix, obviously. 

`k` is the number of rows/columns you wish to retain to form the submatrix. 
`NDet` is the dimension of the full, original matrix (number of determinants, in
configuration interaction lingo).

The "exact" value should be the lowest, followed by ap-sCI or greedy values
(depending on the problem, greedy or ap-sCI will be lower). RLCI is initialized 
by the "greedy" algorithm, so the RL value should be lower than the greedy value.

Since ap-sCI is obtained by taking the exact solution and taking the `k` largest
magnitude components of its corresponding eigenvector, it serves as a measure of
"how good" we are doing. If we obtain a lower value than ap-sCI, the RLCI and/or
greedy algorithm are performing well.

