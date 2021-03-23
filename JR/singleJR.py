# All necessary functions to describe a single Jansen and Rit cortical column model
import numpy as np; from numba import njit
from matfuns import S
usefastmath = True

def unpacking(params):
    '''Returns the values of the parameters in the dictionary params in a tuple so that we can work with numba'''
    A, B, v0, a, b, e0, pbar = params['A'],params['B'],params['v0'],params['a'], params['b'], params['e0'], params['pbar']
    delta, f, C, C1, C2 = params['delta'], params['f'], params['C'], params['C1'], params['C2']
    C3, C4, r = params['C3'], params['C4'], params['r']

    return (A, B, v0, a, b, e0 , pbar, delta, f, C, C1, C2, C3, C4, r )

@njit(fastmath = usefastmath)
def derivatives(x, t, paramtup):
    '''
    Function that returns the derivatives of the 6-dimensional ODE system.
    Inputs:
    x:          6-element vector: x = (z0, y0, z1, y1, z2, y2). Numpy array
    t:          Value of time
    paramtup:   Tuple with all the parameters inside.
    '''
    #Unpacking of values.
    A, B, v0, a, b, e0, pbar, delta, f, C, C1, C2, C3, C4, r = paramtup
    # Assigning variables value for each input.
    z0 = x[0]
    y0 = x[1]
    z1 = x[2]
    y1 = x[3]
    z2 = x[4]
    y2 = x[5]
    # Derivatives of each variable.
    dz0 = A*a*S(y1-y2,e0,r,v0) - 2*a*z0 - a**2*y0
    dy0 = z0
    dz1 = A*a*(pbar + C2*S(C1*y0,e0,r,v0) + delta*np.sin(2*np.pi*f*t)) - a**2*y1 - 2*a*z1
    dy1 = z1
    dz2 = B*b*(C4*S(C3*y0,e0,r,v0)) - 2*b*z2 - b**2*y2
    dy2 = z2
    return np.array([dz0, dy0, dz1, dy1, dz2, dy2])

@njit(fastmath = usefastmath)
def Heun_nonoise(x0, tspan, tstep, fun, funparams):
    '''
    Heun method of integrating equations, this time without noise simulation.
    Iputs:
    x0:         Vector of initial conditions.
    tspan:      Tuple: (t0, tf).
    tstep:      Timestep.
    fun:        Function f from dy/dt = f(t,y).
    funparams:  Parameters that the function f needs.

    Outputs:
    x:      Array of shape (nsteps,6) containing the evolution of the 6 variables.
    tvec:   Vector with time values. Size of nsteps.
    '''
    t0 = tspan[0]
    tf = tspan[1]
    nsteps = int((tf-t0)/tstep) # Number of total steps that will be taken.
    dim = x0.size # Number of odes to integrate.

    tvec = np.linspace(t0,tf,nsteps)    
    x = np.zeros((nsteps,dim)) # Matrix of outputs
    x[0,:] = x0
    
    # Loop. Main algorithm of the Heun Method.
    for n in range(nsteps-1):
        # We first obtain the intermediate value (aux) with an Euler algorithm.
        t1 = tvec[n]
        f1 = fun(x[n,:], t1, funparams)
        
        aux = x[n,:] + tstep*f1
        
        # Then we obtain the final approximation at the next integration point.
        t2 = tvec[n+1]
        f2 = fun(aux, t2, funparams)
        
        x[n+1,:] = x[n,:] + 0.5*tstep*(f1 + f2)
        
    return x, tvec 


def obtaindynamics(params, tspan, tstep):
    ''' 
    Returns the evolution over time of the PSP of the pyramidal population.
    Inputs:
    params: Dictionary of parameters
    tspan:  Tuple of the type (t_0, t_f)
    tstep:  Timestep
    '''
    funparams = unpacking(params)
    x0=10*np.random.normal(size=6) # starting from random normally distributed IC 
    
    x1,t1 = Heun_nonoise(x0, tspan, tstep, derivatives, funparams) # We want to get rid of the transitory in the ode.
    x, t = x1[10000:], t1[10000:] # First ten seconds omitted, enough for it to converge.
    # Select from the 6 ODEs those corresponding to the excitatory and inhibitory PSPs
    y1 = x[:,3]
    y2 = x[:,5]
    return y1-y2,t