import numpy as np
import itertools
import matplotlib.pyplot as plt
from numba import jit, njit, prange

@njit
def distance(p1, p2):
    """
        Calculate the distance between two points, p1 and p2, using numpy.
    """
    return np.sqrt(np.sum(np.power(p1 - p2, 2)))

@njit
def setCities(n, area=(1,1)):
    """
        This Function distribute points uniformly in a square space
    """
    pos = np.random.rand(n, 2)
    pos[:,0] *= area[0]
    pos[:,1] *= area[1]
    return pos

@njit
def setCitiesCirc(n, area=(1,1)):
    """
        Puts points in a circular grid
    """
    rand = np.random.choice(np.arange(0, n), n, replace=False)
    arange = np.arange(n)
    x = area[0]*np.sin(2*np.pi * arange/n)
    y = area[0]*np.sin(2*np.pi * arange/n)
    pos = np.column_stack((x,y))
    return pos[rand]
    
@njit
def getInitialState(n, grid='random'):
    """
        For a given points distribution this function find a random initial system state
    """
    s = np.random.choice(np.arange(n), n, replace=False)
    if grid == 'random':
        pos = setCities(n, area=(1,1))
    elif grid == 'circ':
        pos = setCitiesCirc(n, area=(1,1)) 
    return s, pos

@njit
def Hamiltonian(pos):
    """
        Calculate the system energy
    """
    f = 0
    l = len(pos) - 1
    for i in range(l):
        f += distance(pos[i], pos[i+1])
    f += distance(pos[0], pos[-1])
    return f

@jit
def permutate(s, nCities, nPermutations=2):   
    """
        Perform n permutations in the traveler trajectory 
    """
    if nPermutations % 2 != 0:
        raise ValueError('nPermutations mus be pair')

    s0 = np.copy(s)
    halfLen = int(nPermutations/2)
    p = np.random.choice(np.arange(1, nCities), nPermutations, replace=False)
    for i in prange(halfLen):
        aux = s[p[i]].copy()
        s[p[i]] = s[p[i+halfLen]]
        s[p[i+halfLen]] = aux

    return s0, s

@jit
def newState(s, pos, nCities, T, nPermutations):
    """
        Try a new system trajectory 
    """
    s0, s = permutate(s, nCities, nPermutations)
    e0 = Hamiltonian(pos[s0])
    e = Hamiltonian(pos[s])

    if e < e0:
        return s, e
    else:
        Tss0 = np.exp(-(e - e0)/T)
        if np.random.rand() < Tss0:
            return s, e
        else: return s0, e0

@jit
def simulate(nCities, nPermutations, Ti, DeltaT, steps, grid):
    """
        Performs the Metropolis algorithm
    """
    s, pos = getInitialState(nCities, grid)
    s0 = s.copy()
    e = Hamiltonian(pos)
    eList = np.zeros(steps)
    smallest = np.array([e, s])
    acc = 0
    for t in prange(steps):
        e0 = e
        s, e = newState(s=s, pos=pos, nCities=nCities, T=Ti, nPermutations=nPermutations)
        if e < smallest[0]: 
            smallest[0] = e
            smallest[1] = s.copy()
        if t % 10000 == 0: Ti *= DeltaT
        if t % 1000 == 0: print(t, Ti, e)
        eList[t] = e
    return eList, smallest, s0, pos#, sList, posList 


def findAllStates(nCities, s, pos):
    """
        Finds all possible trajectories for a given cities distribution 
        Pls, do not use large values of cities. I recommend a maximum of ten 

    """
    states = list(itertools.permutations(s[1:]))
    l = len(states)
    states = np.column_stack((np.zeros(len(states), dtype=np.int), states))
    elist = np.zeros(l)
    return elist, states
    for s in prange(l):
        elist[s] = Hamiltonian(pos[states[s]])
        print(s)
        if s % 1000 == 0: print(l-s)
    return elist, states

