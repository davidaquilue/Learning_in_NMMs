''' Necessary functions to obtain the dynamics of a network of Jansen and Rit models of cortical columns.

Functions included: unpackingNET_V1, derivativesNET_V1, couplingval_V1, individual_to_weights, unpackingNET_V2, derivativesNET_V2, 
couplingval_V2, unpackingNET_V3, derivativesNET_V3, HeunNet, obtaindynamicsNET '''

from numba import njit
from matfuns import S, networkmatrix, findlayer
import numpy as np
usefastmath = True

# VERSION 1. One excit/inhib parameter per column. In version 2, every connexion in the matrix will have a different value
def unpackingNET_V1(params):
    '''Returns the values of the parameters in the dictionary params in a tuple so that we can work with numba'''
    A, B, v0, a, b, e0, pbar = params['A'],params['B'],params['v0'],params['a'], params['b'], params['e0'], params['pbar']
    delta, f, C1, C2 = params['delta'], params['f'], params['C1'], params['C2']
    C3, C4, r = params['C3'], params['C4'], params['r'] # JR Model Params

    # Network architecture params
    matrix = params['matrix']
    Nnodes = params['Nnodes']
    tuplenetwork = params['tuplenetwork']
    forcednode = params['forcednode']
    individual = params['individual']
    return A, B, v0, a, b, e0, pbar, delta, f, C1, C2, C3, C4, r, individual, matrix, Nnodes, tuplenetwork, forcednode
    

@njit(fastmath = usefastmath)
def derivativesNET_V1(inp, t, paramtup, n):
    ''' Returns derivatives of the 6 variables of the model for each node
    Inputs:
    inp:    A (N_nodes,6) matrix
    t:      step of time for which the values are those of inp. t = tvec[n]
    n:      Step of the iteration
    Output:
    dz:     A (N_nodes,6) matrix, containing all the derivatives of the variables.
    '''
    A, B, v0, a, b, e0, pbar, delta, f, C1, C2, C3, C4, r, individual, matrix, Nnodes, tuplenetwork, forcednode = paramtup

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


# VERSION 2. Each connection consists of different excitatory and inhibitory weights
def individual_to_weights(individual, matrix_exc, matrix_inh):
    """Individual is transformed into two weight arrays. Individual should have size of 2*np.count_nonzero(matrix)"""
    individual = np.array(individual)
    ind_exc_weights = individual[0:len(np.nonzero(matrix_exc)[0])]
    ind_inh_weights = individual[len(np.nonzero(matrix_exc)[0]):]

    weights_exc = np.zeros_like(matrix_exc)
    weights_inh = np.zeros_like(matrix_inh)

    indices_exc = np.nonzero(matrix_exc)
    indices_inh = np.nonzero(matrix_inh)

    weights_exc[indices_exc] = ind_exc_weights
    weights_inh[indices_inh] = ind_inh_weights

    return weights_exc, weights_inh


def unpackingNET_V2(params):
    """Returns the values of the parameters in the dictionary params in a tuple so that we can work with numba"""
    A, B, v0, a, b, e0, pbar = params['A'],params['B'],params['v0'],params['a'], params['b'], params['e0'], params['pbar']
    delta, f, C1, C2 = params['delta'], params['f'], params['C1'], params['C2']
    C3, C4, r = params['C3'], params['C4'], params['r'] # JR Model Params

    # Network architecture params
    Nnodes = params['Nnodes']
    tuplenetwork = params['tuplenetwork']
    forcednode = params['forcednode']
    individual = params['individual']
    weights_exc, weights_inh = individual_to_weights(individual, params['matrix_exc'], params['matrix_inh'])

    return A, B, v0, a, b, e0 , pbar, delta, f, C1, C2, C3, C4, r, weights_exc, weights_inh, Nnodes, tuplenetwork, forcednode

@njit(fastmath = usefastmath)
def derivativesNET_V2(inp, t, paramtup, n):
    ''' Returns derivatives of the 6 variables of the model for each node
    Inputs:
    inp:    A (N_nodes,6) matrix
    t:      step of time for which the values are those of inp. t = tvec[n]
    n:      Step of the iteration
    Output:
    dz:     A (N_nodes,6) matrix, containing all the derivatives of the variables.
    '''
    A, B, v0, a, b, e0 , pbar, delta, f, C1, C2, C3, C4, r, weights_exc, weights_inh, Nnodes, tuplenetwork, forcednode = paramtup

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
        pbar = np.random.uniform(120,360)
        # Coupled intensities, we obtain them from a function.
        pa, pb = couplingval_V2(inp, weights_exc[nn], weights_inh[nn], C3, e0, r, v0, nn, tuplenetwork, Nnodes)

        # If the node is the forced node, there is periodic driving, if not, the stimulation comes from constant and other nodes.
        delta1 = 0
        if nn == forcednode:
            delta1 = delta
        # Derivatives of each variable.
        dz0 = A*a*S(y1-y2,e0,r,v0) - 2*a*z0 - a**2*y0
        dy0 = z0
        dz1 = A*a*(pbar + C2*S(C1*y0,e0,r,v0) + delta1*np.sin(2*np.pi*f*t) + pa) - a**2*y1 - 2*a*z1
        dy1 = z1
        dz2 = B*b*(C4*S(C3*y0,e0,r,v0) + pb) - 2*b*z2 - b**2*y2
        dy2 = z2
        dz[nn] = np.array([dz0, dy0, dz1, dy1, dz2, dy2])
    return dz


@njit(fastmath=usefastmath)
def couplingval_V2(inp, row_weights_exc, row_weights_inh, C3, e0, r, v0, nn, tuplenetwork, Nnodes):
    """Obtains the effects of the coupling for each node. One excitatory/inhibitory value constant per cortical column.
    Main approach used in the first computations of the Genetic Algorithm"""
    pa = 0
    pb = 0
    # We obtain the contribution of each node.
    for node, (value_ex, value_inh) in enumerate(zip(row_weights_exc, row_weights_inh)):
        if value_ex != 0:  # by this we have acces to the nodes to which the current node is linked to
            pa = pa + value_ex*S(inp[node, 3]-inp[node, 5], e0, r, v0)
        if value_inh != 0:
            pb = pb + value_inh*S(C3*inp[node, 1], e0, r, v0)
    return pa, pb


# VERSION 3. The excitation comes from a time dependent signal, with a pair of coupling values per connection (as in version 2), thus couplingval_V2 will be used.
def unpackingNET_V3(params):
    """Returns the values of the parameters in the dictionary params in a tuple so that we can work with numba"""
    A, B, v0, a, b, e0, pbar = params['A'],params['B'],params['v0'],params['a'], params['b'], params['e0'], params['pbar']
    C1, C2 = params['C1'], params['C2']
    C3, C4, r = params['C3'], params['C4'], params['r']  # JR Model Params

    # Network architecture params
    Nnodes = params['Nnodes'] # Total number of nodes
    tuplenetwork = params['tuplenetwork'] # A tuple containing the architecture of the network, in our case: (3,3,3)
    forcednodes = params['forcednodes'] # A tuple containing the indexes of the input layer, in our case: (0,1,2)
    individual = params['individual']   # Vector containing the values of the coupling values, through the following line it is converted to two weight matrices.
    weights_exc, weights_inh = individual_to_weights(individual, params['matrix_exc'], params['matrix_inh']) # Connectivity matrices, if matrix[i,j] = 0 nodes i,j are 
    # not connected, if matrix[i,j] means that node j sends an input to node i.

    # Signals, that will be built outside the dynamics functions and will have to be the same length as will be the resulting vectors.
    signals = params['signals']# shape of (nodesfirstlayer, nsteps)

    return A, B, v0, a, b, e0 , pbar, C1, C2, C3, C4, r, weights_exc, weights_inh, Nnodes, tuplenetwork, forcednodes, signals


@njit(fastmath=usefastmath)
def derivativesNET_V3(inp, t, paramtup, n):
    """ Returns derivatives of the 6 variables of the model for each node
    Inputs:
    inp:    A (N_nodes,6) matrix
    t:      step of time for which the values are those of inp. t = tvec[n]
    n:      Step of the iteration
    Output:
    dz:     A (N_nodes,6) matrix, containing all the derivatives of the variables.
    """
    A, B, v0, a, b, e0, pbar, C1, C2, C3, C4, r, weights_exc, weights_inh, Nnodes, tuplenetwork, forcednodes, signals = paramtup
    # Now the input will be a matrix where each row i corresponds to the variables (z0,y0,z1,y1,z2,y2) of each node i.
    dz = np.zeros_like(inp)
    # Now we obtain the derivatives of every variable for every node.
    for nn in range(Nnodes):
        x = inp[nn]  # This will extract the row corresponding to each node.
        z0 = x[0]
        y0 = x[1]
        z1 = x[2]
        y1 = x[3]
        z2 = x[4]
        y2 = x[5]
        pbar = np.random.uniform(120, 240)
        """
        # The following lines of code are just a test to see if it is possible
        # for one of the layers to oscillate at a different rhythm
        a = paramtup[3]
        b = paramtup[4]
        C1 = paramtup[7]
        C2 = 0.8*C1
        C3 = 0.25*C1
        C4 = 0.25*C1
        if findlayer(nn, tuplenetwork) == 2:
            a = 190
            b = 190
            C1 = 300
            C2 = 0.8*C1
            C3 = 0.25*C1
            C4 = 0.25*C1
        """
        # Coupled intensities, we obtain them from a function.
        pa, pb = couplingval_V2(inp, weights_exc[nn], weights_inh[nn], C3, e0, r, v0, nn, tuplenetwork, Nnodes)

        if nn in forcednodes:
            pbar = signals[nn, n]
        # Derivatives of each variable.
        dz0 = A*a*S(y1-y2, e0, r, v0) - 2*a*z0 - a**2*y0
        dy0 = z0
        dz1 = A*a*(pbar + C2*S(C1*y0, e0, r, v0) + pa) - a**2*y1 - 2*a*z1
        dy1 = z1
        dz2 = B*b*(C4*S(C3*y0, e0, r, v0) + pb) - 2*b*z2 - b**2*y2
        dy2 = z2
        dz[nn] = np.array([dz0, dy0, dz1, dy1, dz2, dy2])
    return dz


# VERSION 4. ALL CONNECTIONS HAVE A CERTAIN DELAY, FOR RECURRENCE IN GA WIHT TIMESHIFTED INPUTS.
# It would be interesting for the GA to also determine the amount of delay that it will apply. 
def unpackingNET_V4(params):
    """Returns the values of the parameters in the dictionary params in a tuple so that we can work with numba"""
    A, B, v0, a, b, e0, pbar = params['A'],params['B'],params['v0'],params['a'], params['b'], params['e0'], params['pbar']
    C1, C2 = params['C1'], params['C2']
    C3, C4, r = params['C3'], params['C4'], params['r']  # JR Model Params

    # Network architecture params
    Nnodes = params['Nnodes'] # Total number of nodes
    tuplenetwork = params['tuplenetwork'] # A tuple containing the architecture of the network, in our case: (3,3,3)
    forcednodes = params['forcednodes'] # A tuple containing the indexes of the input layer, in our case: (0,1,2)
    individual = params['individual']   # Vector containing the values of the coupling values, through the following line it is converted to two weight matrices.
    weights_exc, weights_inh = individual_to_weights(individual, params['matrix_exc'], params['matrix_inh']) # Connectivity matrices, if matrix[i,j] = 0 nodes i,j are 
    # not connected, if matrix[i,j] means that node j sends an input to node i.

    delaysteps = params['delaysteps']

    # Signals, that will be built outside the dynamics functions and will have to be the same length as will be the resulting vectors.
    signals = params['signals']# shape of (nodesfirstlayer, nsteps)

    return A, B, v0, a, b, e0 , pbar, C1, C2, C3, C4, r, weights_exc, weights_inh, Nnodes, tuplenetwork, forcednodes, signals, delaysteps


@njit(fastmath=usefastmath)
def derivativesNET_V4(inp, t, paramtup, n, inpdelay):
    """ Returns derivatives of the 6 variables of the model for each node
    Inputs:
    inp:    A (N_nodes,6) matrix
    t:      step of time for which the values are those of inp. t = tvec[n]
    n:      Step of the iteration
    inp:    A (N_nodes, 6) matrix of delaysteps timesteps before.
    Output:
    dz:     A (N_nodes,6) matrix, containing all the derivatives of the variables.
    """
    A, B, v0, a, b, e0, pbar, C1, C2, C3, C4, r, weights_exc, weights_inh, Nnodes, tuplenetwork, forcednodes, signals, delaysteps = paramtup
    # Now the input will be a 3D matrix where for inp[]each row i corresponds to the variables (z0,y0,z1,y1,z2,y2) of each node i.
    dz = np.zeros_like(inp)
    # Now we obtain the derivatives of every variable for every node.
    for nn in range(Nnodes):
        x = inp[nn]  # This will extract the row corresponding to each node.
        z0 = x[0]
        y0 = x[1]
        z1 = x[2]
        y1 = x[3]
        z2 = x[4]
        y2 = x[5]
        pbar = np.random.uniform(120, 240)
        # Coupled intensities, we obtain them from a function.
        pa, pb = couplingval_V4(inpdelay, weights_exc[nn], weights_inh[nn], C3, e0, r, v0, nn, tuplenetwork, Nnodes)

        if nn in forcednodes:
            pbar = signals[nn, n]
        # Derivatives of each variable.
        dz0 = A*a*S(y1-y2, e0, r, v0) - 2*a*z0 - a**2*y0
        dy0 = z0
        dz1 = A*a*(pbar + C2*S(C1*y0, e0, r, v0) + pa) - a**2*y1 - 2*a*z1
        dy1 = z1
        dz2 = B*b*(C4*S(C3*y0, e0, r, v0) + pb) - 2*b*z2 - b**2*y2
        dy2 = z2
        dz[nn] = np.array([dz0, dy0, dz1, dy1, dz2, dy2])
    return dz

@njit(fastmath=usefastmath)
def couplingval_V4(inp, row_weights_exc, row_weights_inh, C3, e0, r, v0, nn, tuplenetwork, Nnodes):
    """Obtains the effects of the coupling for each node. One excitatory/inhibitory value constant per cortical column.
    Main approach used in the first computations of the Genetic Algorithm"""
    pa = 0
    pb = 0
    # We obtain the contribution of each node.
    for node, (value_ex, value_inh) in enumerate(zip(row_weights_exc, row_weights_inh)):
        if value_ex != 0:  # by this we have acces to the nodes to which the current node is linked to
            pa = pa + value_ex*S(inp[node, 3]-inp[node, 5], e0, r, v0)
        if value_inh != 0:
            pb = pb + value_inh*S(C3*inp[node, 1], e0, r, v0)
    return pa, pb

# OBTAINING DYNAMICS FUNCTIONS
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
        f1 = fun(x[:,n,:], t1, funparams, n)
        
        aux = x[:,n,:] + tstep*f1
        
        t2 = tvec[n+1]
        f2 = fun(aux, t2, funparams, n)
        x[:,n+1,:] = x[:,n,:] + 0.5*tstep*(f1 + f2)
        
    return x, tvec


@njit(fastmath = usefastmath)
def HeunNET_V4(x0, tspan, tstep, fun, funparams, delaysteps):
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
        if n <= delaysteps:
            inpdelay = x[:, 0, :]
        else:
            inpdelay = x[:, n-delaysteps, :]

        f1 = fun(x[:,n,:], t1, funparams, n, inpdelay)
        
        aux = x[:,n,:] + tstep*f1
        
        t2 = tvec[n+1]
        f2 = fun(aux, t2, funparams, n, inpdelay)
        x[:,n+1,:] = x[:,n,:] + 0.5*tstep*(f1 + f2)
        
    return x, tvec


def obtaindynamicsNET(params, tspan, tstep, v):
    ''' 
    Returns the evolution over time of the PSP of the pyramidal population.
    Inputs:
    params: Dictionary of parameters
    tspan:  Tuple of the type (t_0, t_f)
    tstep:  Timestep
    v:      Whether version 1 (1 exc/inh coef/node), version 2 (1 exc/inh coef/connection) or version 3 (1 exc/inh coef/connection and squared input signals)

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
    elif v == 2:
        funparams = unpackingNET_V2(params)
        x1,t1 = HeunNET(x0, tspan, tstep, derivativesNET_V2, funparams)
    elif v == 3 and not params['recurrent']:
        funparams = unpackingNET_V3(params)
        x1, t1 = HeunNET(x0, tspan, tstep, derivativesNET_V3, funparams)
    elif v == 3 and params['recurrent']:
        funparams = unpackingNET_V4(params)
        delaysteps = params['delaysteps']
        x1, t1 = HeunNET_V4(x0, tspan, tstep, derivativesNET_V4, funparams, delaysteps)

    else:
        print('No version has been selected. Dynamics not obtained.')

    # The output matrix x1 is of the type (Nnodes, timevalues, Nvariables)
    x, t = x1[:,10000:,:], t1[10000:] # We get values after 10 seconds. Enough to get rid of the transitory

    y1 = x[:,:,3]
    y2 = x[:,:,5]
    return y1-y2,t
