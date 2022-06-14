import numpy as np

from numba import njit, vectorize
from Sampler  import run_sampler
from Particle import Particle

from PyGFun import to_pg, from_pg, WIDTH, HEIGHT
global WIDTH, HEIGHT

@vectorize
def Ve(r):
    return np.sqrt(2) * ( 1 + r**2)**(-1/4)

def Compute_IC(N_0, seed = None):
    
    if seed != None:
        np.random.seed(seed)
        samples = run_sampler(0.5, 2*N_0+1000, 1000, 0.1, seed)
    else:
        samples = run_sampler(0.5, 2*N_0+1000, 1000, 0.1)
    
    R = 100
    Mtot = 100
    Energy = -(3 * np.pi / 64) * Mtot**2 / R
    
    to_len = 3 * np.pi * Mtot**2 / ( 64 * np.abs(Energy)) 
    to_vel = 64 * np.sqrt(np.abs(Energy) / Mtot) / (3 * np.pi)
    
    masses = np.random.uniform(size = N_0)
    radius = ( masses**(-2/3) - 1 )**(-1/2)
    angs = np.random.uniform(size = N_0)
    
    x      = np.sqrt( radius ) * np.cos(2*np.pi* angs)
    y      = np.sqrt( radius ) * np.sin(2*np.pi* angs)
    
    #x[:len(x)//2] = x[:len(x)//2] + 5
    #x[len(x)//2:] = x[len(x)//2:] - 5
    
    V = samples[:N_0] * Ve(radius)
    angs = np.random.uniform(size = N_0) 
    vx      = np.sqrt( V ) * np.cos(2*np.pi* angs)
    vy      = np.sqrt( V ) * np.sin(2*np.pi* angs)
    
    particles = []
    for i in range(N_0):
        pos = to_pg( (x[i] *to_len, y[i] *to_len) )
        v   = [vx[i]*to_vel, vy[i]*to_vel]
        particles.append(Particle(pos, v, masses[i]*Mtot, _color = (255,255,255), _tolen = to_len))
    
    ### TEST CONFIGURATION
    #particles = []
    #particles.append(Particle( to_pg((0,0)), [0,0], 100, _color = (255,255,255), _tolen = 1))
    #particles.append(Particle( to_pg((100,0)), [0,-.5], 100, _color = (255,255,255), _tolen = 1))
    
    return particles