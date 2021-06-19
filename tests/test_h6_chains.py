import numpy as np
from numpy.testing import assert_allclose 
from rlci.solvers import RL 

np.random.seed(1)

def test_r1p0():
    M = np.loadtxt('full_hamiltonians/h6_1p00_chain.txt')
    k = 40

    # full diagonalization 
    E_exact, _ = RL(M,k=k,mode='full') 
    assert_allclose(-3.2576068322409553,E_exact)

    # a-posteriori selected CI
    E_apsCI, _ = RL(M,k=k,mode='apsci') 
    assert_allclose(-3.2514253974982923,E_apsCI)

    # greedy selected CI
    E_greedy, _ = RL(M,k=k,mode='greedy') 
    assert_allclose(-3.2514253974982923,E_greedy)

    # reinforcement learning CI -- depends on random seed, so leave out for now
    #E_rl, s = RL(M, k, mode='rl', max_pick=50)
    #assert_allclose(-3.2515672420550255,E_rl)
    
def test_r1p5():
    M = np.loadtxt('full_hamiltonians/h6_1p50_chain.txt')
    k = 40

    # full diagonalization 
    E_exact, _ = RL(M,k=k,mode='full') 
    assert_allclose(-3.020198096930829,E_exact)

    # a-posteriori selected CI
    E_apsCI, _ = RL(M,k=k,mode='apsci') 
    assert_allclose(-2.995049575625855,E_apsCI)

    # greedy selected CI
    E_greedy, _ = RL(M,k=k,mode='greedy') 
    assert_allclose(-2.994664375794827,E_greedy)

    # reinforcement learning CI -- depends on random seed, so leave out for now
    #E_rl, s = RL(M, k, mode='rl', max_pick=50)
    #assert_allclose(-2.9975365981935687,E_rl)
    
def test_r2p0():
    M = np.loadtxt('full_hamiltonians/h6_2p00_chain.txt')
    k = 40

    # full diagonalization 
    E_exact, _ = RL(M,k=k,mode='full') 
    assert_allclose(-2.8740730709371056,E_exact)

    # a-posteriori selected CI
    E_apsCI, _ = RL(M,k=k,mode='apsci') 
    assert_allclose(-2.8571403691550183,E_apsCI)

    # greedy selected CI
    E_greedy, _ = RL(M,k=k,mode='greedy') 
    assert_allclose(-2.857843869693096,E_greedy)

    # reinforcement learning CI -- depends on random seed, so leave out for now
    #E_rl, s = RL(M, k, mode='rl', max_pick=50)
    #assert_allclose(-2.857935385969051,E_rl)
    
