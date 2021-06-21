import numpy as np
from numpy.testing import assert_allclose 
from rlci.solvers import RL 

np.random.seed(1)

def test_r1p0():
    M = np.loadtxt('full_hamiltonians/h6_1p00_ring.txt')
    k = 40

    # full diagonalization 
    E_exact, _ = RL(M,k=k,mode='full') 
    assert_allclose(-3.2587720432125162,E_exact)

    # a-posteriori selected CI
    E_apsCI, _ = RL(M,k=k,mode='apsci') 
    assert_allclose(-3.2576584886428814,E_apsCI)

    # greedy selected CI
    E_greedy, _ = RL(M,k=k,mode='greedy') 
    assert_allclose(-3.2576363073108983,E_greedy)

    # RLCI -- depends on random seed, so criteria is "looser"
    E_rl, s = RL(M, k, mode='rl', max_pick=50, silent=True)
    assert E_rl <= E_greedy

    
def test_r1p5():
    M = np.loadtxt('full_hamiltonians/h6_1p50_ring.txt')
    k = 40

    # full diagonalization 
    E_exact, _ = RL(M,k=k,mode='full') 
    assert_allclose(-3.0384283093874647,E_exact)

    # a-posteriori selected CI
    E_apsCI, _ = RL(M,k=k,mode='apsci') 
    assert_allclose(-3.0308182115807174,E_apsCI)

    # greedy selected CI
    E_greedy, _ = RL(M,k=k,mode='greedy') 
    assert_allclose(-3.0306888446843683,E_greedy)

    # RLCI -- depends on random seed, so criteria is "looser"
    E_rl, s = RL(M, k, mode='rl', max_pick=50, silent=True)
    assert E_rl <= E_greedy
    

def test_r2p0():
    M = np.loadtxt('full_hamiltonians/h6_2p00_ring.txt')
    k = 40

    # full diagonalization 
    E_exact, _ = RL(M,k=k,mode='full') 
    assert_allclose(-2.8784268631037997,E_exact)

    # a-posteriori selected CI
    E_apsCI, _ = RL(M,k=k,mode='apsci') 
    assert_allclose(-2.8679644866850493,E_apsCI)

    # greedy selected CI
    E_greedy, _ = RL(M,k=k,mode='greedy') 
    assert_allclose(-2.867964486685051,E_greedy)

    # RLCI -- depends on random seed, so criteria is "looser"
    E_rl, s = RL(M, k, mode='rl', max_pick=50, silent=True)
    assert E_rl <= E_greedy
   

 
