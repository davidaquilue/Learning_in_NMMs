import numpy as np; from numba import njit
from matfuns import S, networkmatrix
usefastmath = True

# Version 1. One excit/inhib parameter per column. In version 2, every connexion in the matrix will have a different value
def unpackingNET_V1(params):
    '''Returns the values of the parameters in the dictionary params in a tuple so that we can work with numba'''
    A, B, v0, a, b, e0, pbar = params['A'],params['B'],params['v0'],params['a'], params['b'], params['e0'], params['pbar']
    delta, f, C, C1, C2 = params['delta'], params['f'], params['C'], params['C1'], params['C2']
    C3, C4, r = params['C3'], params['C4'], params['r'] # JR Model Params

    # Network architecture params
    matrix = params['matrix']
    Nnodes = params['Nnodes']
    tuplenetwork = params['tuplenetwork']
    forcednode = params['forcednode']
    stimulation_mode = params['stimulation_mode']
    individual = params['individual']
    return (A, B, v0, a, b, e0 , pbar, delta, f, C, C1, C2, C3, C4, r, individual, matrix, Nnodes, tuplenetwork, forcednode, stimulation_mode)
    

@njit(fastmath = usefastmath)
def derivativesNET_V1(inp, t, paramtup):
    ''' Returns derivatives of the 6 variables of the model for each node
    Inputs:
    inp:    A (N_nodes,6) matrix
    t:      step of time for which the values are those of inp
    Output:
    dz:     A (N_nodes,6) matrix, containing all the derivatives of the variables.
    '''
    A, B, v0, a, b, e0 , pbar, delta, f, C, C1, C2, C3, C4, r, individual, matrix, Nnodes, tuplenetwork, forcednode, stimulation_mode = paramtup

    # Now the input will be a matrix where each row i corresponds to the variables (z0,y0,z1,y1,z2,y2) of each node i.
    dz = np.zeros_like(inp)
    # Now we obtain the derivatives of every variable for every node.
    for nn in range(Nnodes):
        x = inp[nn] # This will extract the row corresponding to each node.
        z0 = x[0]
        y0 = x[1]
        z1 = x[2]
        y1 = x[3]
        z2 = x[4]
        y2 = x[5]
        
        # Coupled intensities, we obtain them from a function.
        pa, pb = couplingval_V1(inp, matrix[nn], individual, C3, e0, r, v0, nn, tuplenetwork, Nnodes)

        # If the node is the forced node, there is periodic driving, if not, the stimulation comes from constant and other nodes.
        delta1 = 0
        if nn == forcednode:
            delta1 = delta
        # Derivatives of each variable.
        dz0 = A*a*S(y1-y2,e0,r,v0) - 2*a*z0 - a**2*y0
        dy0 = z0
        dz1 = A*a*(pbar + C2*S(C1*y0,e0,r,v0) + delta1*np.sin(2*np.pi*f*t)) - a**2*y1 - 2*a*z1 + pa
        dy1 = z1
        dz2 = B*b*(C4*S(C3*y0,e0,r,v0)) - 2*b*z2 - b**2*y2 + pb
        dy2 = z2
        dz[nn] = np.array([dz0, dy0, dz1, dy1, dz2, dy2])
    return dz


@njit(fastmath = usefastmath)
def couplingval_V1(inp, connections, individual, C3, e0, r, v0, nn, tuplenetwork, Nnodes):
    '''
    Obtains the effects of the coupling for each node. One excitatory/inhibitory value constant per cortical column.
    Main approach used in the first computations of the Genetic Algorithm'''
    Suma = 0
    Sumb = 0
    # We obtain the contribution of each node.
    for node,value in enumerate(connections):
        if value == 1:  #by this we have acces to the nodes to which the current node is linked to
            alpha = individual[node]
            beta = individual[-(node+1)]
            Suma = Suma + alpha*S(inp[node,3]-inp[node,5], e0, r, v0)
            Sumb = Sumb + beta*S(C3*inp[node, 1], e0, r, v0)
    if nn < tuplenetwork[0]: # First layer.
        Ni = 1
        Nj = 3
    else:
        Ni = 3
        Nj = 3
    pa = Suma/np.sqrt(Ni*Nj)
    pb = Sumb/np.sqrt(Ni*Nj)
    return pa,pb



# Version 2. Each connection consists of different excitatory and inhibitory weights
def individual_to_weights(individual, matrix):
    '''Individual is transformed into two weight arrays. Individual should have size of 2*np.count_nonzero(matrix)'''
    individual = np.array(individual)
    halves = np.split(individual, 2)
    weights_exc = np.zeros_like(matrix)
    weights_inh = np.zeros_like(matrix)

    indices = np.nonzero(matrix)

    weights_exc[indices] = halves[0]
    weights_inh[indices] = halves[1]

    return weights_exc, weights_inh

def unpackingNET_V2(params):
    '''Returns the values of the parameters in the dictionary params in a tuple so that we can work with numba'''
    A, B, v0, a, b, e0, pbar = params['A'],params['B'],params['v0'],params['a'], params['b'], params['e0'], params['pbar']
    delta, f, C, C1, C2 = params['delta'], params['f'], params['C'], params['C1'], params['C2']
    C3, C4, r = params['C3'], params['C4'], params['r'] # JR Model Params

    # Network architecture params
    matrix = params['matrix']
    Nnodes = params['Nnodes']
    tuplenetwork = params['tuplenetwork']
    forcednode = params['forcednode']
    stimulation_mode = params['stimulation_mode']
    individual = params['individual']
    weights_exc, weights_inh = individual_to_weights(individual, matrix)

    return (A, B, v0, a, b, e0 , pbar, delta, f, C, C1, C2, C3, C4, r, weights_exc, weights_inh, Nnodes, tuplenetwork, forcednode, stimulation_mode)

@njit(fastmath = usefastmath)
def derivativesNET_V2(inp, t, paramtup):
    ''' Returns derivatives of the 6 variables of the model for each node
    Inputs:
    inp:    A (N_nodes,6) matrix
    t:      step of time for which the values are those of inp
    Output:
    dz:     A (N_nodes,6) matrix, containing all the derivatives of the variables.
    '''
    A, B, v0, a, b, e0 , pbar, delta, f, C, C1, C2, C3, C4, r, weights_exc, weights_inh, Nnodes, tuplenetwork, forcednode, stimulation_mode = paramtup

    # Now the input will be a matrix where each row i corresponds to the variables (z0,y0,z1,y1,z2,y2) of each node i.
    dz = np.zeros_like(inp)
    # Now we obtain the derivatives of every variable for every node.
    for nn in range(Nnodes):
        x = inp[nn] # This will extract the row corresponding to each node.
        z0 = x[0]
        y0 = x[1]
        z1 = x[2]
        y1 = x[3]
        z2 = x[4]
        y2 = x[5]
        
        # Coupled intensities, we obtain them from a function.
        pa, pb = couplingval_V2(inp, weights_exc[nn], weights_inh[nn], C3, e0, r, v0, nn, tuplenetwork, Nnodes)

        # If the node is the forced node, there is periodic driving, if not, the stimulation comes from constant and other nodes.
        delta1 = 0
        if nn == forcednode:
            delta1 = delta
        # Derivatives of each variable.
        dz0 = A*a*S(y1-y2,e0,r,v0) - 2*a*z0 - a**2*y0
        dy0 = z0
        dz1 = A*a*(pbar + C2*S(C1*y0,e0,r,v0) + delta1*np.sin(2*np.pi*f*t)) - a**2*y1 - 2*a*z1 + pa
        dy1 = z1
        dz2 = B*b*(C4*S(C3*y0,e0,r,v0)) - 2*b*z2 - b**2*y2 + pb
        dy2 = z2
        dz[nn] = np.array([dz0, dy0, dz1, dy1, dz2, dy2])
    return dz

@njit(fastmath = usefastmath)
def couplingval_V2(inp, row_weights_exc, row_weights_inh, C3, e0, r, v0, nn, tuplenetwork, Nnodes):
    '''
    Obtains the effects of the coupling for each node. One excitatory/inhibitory value constant per cortical column.
    Main approach used in the first computations of the Genetic Algorithm'''
    pa = 0
    pb = 0
    # We obtain the contribution of each node.
    for node,value in enumerate(row_weights_exc):
        if value != 0:  #by this we have acces to the nodes to which the current node is linked to
            pa = pa + value*S(inp[node,3]-inp[node,5], e0, r, v0)
        elif row_weights_inh[node] != 0:
            pb = pb + row_weights_inh[node]*S(C3*inp[node, 1], e0, r, v0)
    return pa,pb

# I think that the other functions can be used aswell.

@njit(fastmath = usefastmath)
def HeunNET(x0, tspan, tstep, fun, funparams):
    '''
    Heun method of integrating equations, this time without noise simulation. 
    Inputs:
    x0:         Matrix of initial conditions. (N,6)
    tspan:      Tuple: (t0, tf).
    tstep:      Timestep taken.
    fun:        Function f from dy/dt = f(t,y).
    funparams:  Parameters that the function f needs.
    '''
    t0 = tspan[0]
    tf = tspan[1]
    
    nsteps = int((tf-t0)/tstep) # Number of total steps that will be taken.
    
    tvec = np.linspace(t0,tf,nsteps)
    nnodes = x0.shape[0]
    nvars = x0.shape[1]
    x = np.zeros((nnodes,nsteps,nvars)) # Matrix of outputs, now it's 3D
    x[:,0,:] = x0
    # Loop. Main algorithm of the Heun Method.
    for n in range(nsteps-1):
        t1 = tvec[n]
        f1 = fun(x[:,n,:], t1, funparams)
        
        aux = x[:,n,:] + tstep*f1
        
        t2 = tvec[n+1]
        f2 = fun(aux, t2, funparams)
        x[:,n+1,:] = x[:,n,:] + 0.5*tstep*(f1 + f2)
        
    return x, tvec 
import matplotlib.pyplot as plt

def obtaindynamicsNET(params, tspan, tstep, v):
    ''' 
    Returns the evolution over time of the PSP of the pyramidal population.
    Inputs:
    params: Dictionary of parameters
    tspan:  Tuple of the type (t_0, t_f)
    tstep:  Timestep
    v:      Whether version 1 (1 exc/inh coef/node) or version 2 (1 exc/inh coef/connection)

    Outputs:
    y1-y2:   Matrix of the kind (N_nodes, tsteps) of the PSPs of pyramidal populations of every node in the network
    t:      Time vector
    '''
    Nnodes, matrix = networkmatrix(params['tuplenetwork'], params['recurrent'])
    params['Nnodes'] = Nnodes
    params['matrix'] = matrix

    x0=10*np.random.normal(size=(Nnodes,6)) # Random normally distributed IC
    if v == 1:
        funparams = unpackingNET_V1(params)
        x1,t1 = HeunNET(x0, tspan, tstep, derivativesNET_V1, funparams) 
    elif v==2:
        funparams = unpackingNET_V2(params)
        x1,t1 = HeunNET(x0, tspan, tstep, derivativesNET_V2, funparams)
    else:
        print('No version has been selected. Dynamics not obtained.')

    # The output matrix x1 is of the type (Nnodes, timevalues, Nvariables)
    x, t = x1[:,10000:,:], t1[10000:] # We get values after 10 seconds. Enough to get rid of the transitory

    y1 = x[:,:,3]
    y2 = x[:,:,5]
    return y1-y2,t