# NMMs Network with weight behaviour governed by Hebbian Learning
import numpy as np; from numba import njit
import matplotlib.pyplot as plt
usefastmath = True

# Luego habria que duplicar esto, pero con pesos inhibidores en la segunda capa. Y además habria que conseguir que cada nodo i de la primera capa se lo 
# mande al nodo i de la segunda.

# Let us begin by developing the first layer and trying to achieve an autoassociative memory. That is that weights are capable of reproducing 
# the same patterns that are produced when

# matrix = NxM array
# Weights matrix: (N*M)x(N*M) array.

# IMPORTANT PARAMETERS FOR THE FIRST LAYER.
# We can work with shaping and reshaping the columns into a vector, allowing us to work like before with a 3D array [column, odevariables, tstep]
# or we can work with a matrix of columns, which corresponds to a 4D array [i_col, j_col, odevariables, tstep]
 

@njit(fastmath = usefastmath)
def S(m,e0,r,v0):
    '''Transformation corresponding to the transformation in the somas of the net average PSP into an average action potential density.'''
    return 2*e0/(1+np.exp(r*(v0 - m)))

def unpackingHEB(params):
    '''Returns the values of the parameters in the dictionary params in a tuple so that we can work with numba'''
    A, B, v0, a, b, e0 = params['A'],params['B'],params['v0'],params['a'], params['b'], params['e0']
    C1, C2, C3, C4, r = params['C1'], params['C2'], params['C3'], params['C4'], params['r']# JR Model Params
    # Network architecture params
    Totalcols = params['Totalcols']

    return (A, B, v0, a, b, e0 , C1, C2, C3, C4, r, Totalcols)

@njit(fastmath = usefastmath)
def derivativesHEB(inp, t, paramtup, W, stimnodes):
    ''' Returns derivatives of the 6 variables of the model for each node
    Inputs:
    inp:        A (N_nodes,6) matrix
    t:          step of time for which the values are those of inp
    paramtup:   Tuple containing all the parameters
    W:          First layers excitatory weights matrix
    stimpat:    Which pattern is being stimulated at the moment: 1,2 or 3

    Output:
    dz:     A (N_nodes,6) matrix, containing all the derivatives of the variables.
    '''
    A, B, v0, a, b, e0 , C1, C2, C3, C4, r, Totalcols = paramtup

    # Now the input will be a matrix where each row i corresponds to the variables (z0,y0,z1,y1,z2,y2) of each node i.
    dz = np.zeros_like(inp)
    # Now we obtain the derivatives of every variable for every node.
    for nn in range(Totalcols):
        x = inp[nn] # This will extract the row corresponding to each node.
        z0 = x[0]
        y0 = x[1]
        z1 = x[2]
        y1 = x[3]
        z2 = x[4]
        y2 = x[5]

        if nn in stimnodes: # Or maybe do sth like patterns = (pattern1, pattern2, pattern3) and then stimpat = patterns[0] and input 0, 1 or 2.
            pbar = np.random.normal(1000,50)
        else:
            pbar = np.random.uniform(120,360)
        # Coupled intensities, we obtain them from a function.
        # Again, we have to take into account that a row in the W matrix contains the parameters of the nodes that arrive to the row's node.
        pa = couplingvalHEBV1(inp, W[nn], C3, e0, r, v0, nn)

        dz0 = A*a*S(y1-y2,e0,r,v0) - 2*a*z0 - a**2*y0
        dy0 = z0
        dz1 = A*a*(pbar + C2*S(C1*y0,e0,r,v0)) - a**2*y1 - 2*a*z1 + pa
        dy1 = z1
        dz2 = B*b*(C4*S(C3*y0,e0,r,v0)) - 2*b*z2 - b**2*y2
        dy2 = z2
        dz[nn] = np.array([dz0, dy0, dz1, dy1, dz2, dy2])
    return dz

@njit(fastmath = usefastmath)
def couplingvalHEBV1(inp, Wrow, C3, e0, r, v0, nn):
    '''
    Obtains the effects of the coupling for each node. First layer of Hebbian learning only coupling is excitatory'''
    pa = 0
    # We obtain the contribution of each node.
    for node,weightval in enumerate(Wrow):
        if weightval != 0:  #by this we have acces to the nodes to which the current node is linked to
            pa += weightval*S(inp[node,3]-inp[node,5], e0, r, v0)
    return pa

@njit(fastmath = usefastmath)
def HeunHEB(x0, tspan, tstep, fun, funparams, stim_info, W, learnparams, trainingphase):
    '''
    Heun method of integrating equations, this time without noise simulation. 
    Inputs:
    x0:          Matrix of initial conditions. (N,6)
    tspan:       Tuple: (t0, tf).
    tstep:       Timestep taken.
    fun:         Function f from dy/dt = f(t,y).
    funparams:   Parameters that the function f needs.
    stim_info:   A tuple containing the times and patterns that will be stimulated [((t0,t1),(t1,t2),(),..), ((No pattern)(pattern1), (pattern2),...)]
    W:           The weights matrix that we will update after every step with the Hebbian learning algorithm
    learnparams: Parameters that will be used in the learning algorithm
    '''
    t0 = tspan[0]
    tf = tspan[1]
    _, _, v0, _, _, e0 ,  _,  _,  _,  _, r, _ = funparams
    nsteps = int((tf-t0)/tstep) # Number of total steps that will be taken.

    tvec = np.linspace(t0,tf,nsteps)
    nnodes = x0.shape[0]
    nvars = x0.shape[1]
    x = np.zeros((nnodes,nsteps,nvars)) # Matrix of outputs
    x[:,0,:] = x0

    stimtimes = stim_info[0]
    stimpatts = stim_info[1]
    gammaW, vL, Wmax, Tpre = learnparams

    # Loop. Main algorithm of the Heun Method.
    for n in range(nsteps-1):
        t1 = tvec[n]
        # We select which stimulation pattern is going to be run. I don't really know if it is the fastest approach.
        for ii,(t0,tf) in enumerate(stimtimes):
            if t0 <= t1 and t1 < tf:
                stimnodes = stimpatts[ii]
            else:
                stimnodes = stimpatts[0]

        f1 = derivativesHEB(x[:,n,:], t1, funparams, W, stimnodes)
        aux = x[:,n,:] + tstep*f1
        
        t2 = tvec[n+1]
        f2 = derivativesHEB(aux, t2, funparams, W, stimnodes)
        x[:,n+1,:] = x[:,n,:] + 0.5*tstep*(f1 + f2)

        # After each step we update the weights
        if trainingphase:
            for jj in range(W.shape[1]): # This is assuming that we are using the vector type of description
                # We firs iterate over the columns, since in a column, the presynaptic node is always the same, and thus we have
                # to compute less moving averages
                m = np.mean(S(x[jj,-Tpre:,3]-x[jj,-Tpre:,5],e0,r,v0)) # Presynaptic running average activity
                for ii in range(W.shape[0]):
                    if ii != jj:
                        zii = S(x[ii,-1,3]-x[ii,-1,5],e0,r,v0)# Postsynaptic activty 
                        if zii > vL and m > vL:
                            W[ii,jj] = W[ii,jj] + gammaW*(zii - vL)*(m-vL)*(Wmax-W[ii,jj])    
                
    return x, tvec, W

def obtaindynamicsHEB(params, tspan, tstep, stim_info, learnparams, trainingphase, W = None):
    ''' 
    Returns the evolution over time of the PSP of the pyramidal population.
    Inputs:
    params: Dictionary of parameters
    tspan:  Tuple of the type (t_0, t_f)
    tstep:  Timestep

    Outputs:
    y1-y2:   Matrix of the kind (N_nodes, tsteps) of the PSPs of pyramidal populations of every node in the network
    t:      Time vector
    '''
    funparams = unpackingHEB(params)
    Totalcols = funparams[-1]
    x0=np.random.normal(10,5,size=(Totalcols,6)) # Random normally distributed IC
    if trainingphase:
        W = np.zeros((Totalcols, Totalcols))
    if (not trainingphase and W == None):
        print('ERROR: When simulating a weight matrix should be provided')
    
    x,t,W = HeunHEB(x0, tspan, tstep, derivativesHEB, funparams, stim_info, W, learnparams, trainingphase)
    # The output matrix x is of the type (Totalcols, timevalues, Nvariables)
    y1 = x[:,:,3]
    y2 = x[:,:,5]
    return y1-y2,t, W

def same_size_patterns(patterns):
    maxl = 0
    for pattern in patterns:
        if len(pattern)>maxl:
            maxl = len(pattern)
    # Now we have the maximal length
    a = int(len(patterns))
    newpatterns = -np.ones(shape= (a, maxl))
    for ii,pattern in enumerate(patterns):
        for jj, value in enumerate(pattern):
            newpatterns[ii,jj] = value

    return tuple(map(tuple, newpatterns))

if __name__ == '__main__':
    # We define the parameters of the JR model
    params = {'A': 3.25, 'B': 22.0, 'v0': 6.0} 
    params['a'], params['b'], params['e0'] = 100.0, 50.0, 2.5
    C = 133.5
    params['C'], params['C1'], params['C2'], params['C3'], params['C4'] = C, C, 0.8*C, 0.25*C, 0.25*C # All dimensionless
    params['r'] = 0.56 #mV^(-1)

    # Now the system's parameters
    Shape = (5,5)
    Totalcols = Shape[0]*Shape[1]
    params['Totalcols'] = Totalcols
    Indexes_matrix = np.linspace(0,Totalcols-1, Totalcols).reshape(Shape)

    # Finally the learning parameters:
    gammaW = 0.1; vL = 2; Wmax = 10; Tpre = 20
    learnparams = (gammaW, vL, Wmax, Tpre)

    # We are going to train 3 easy patterns
    pattern0 = (-2,-1) # No stimulation but we have to write something so that numba does not explode
    pattern1 = (0,1,2,5,6,7)
    pattern2 = (3,4,8,9,13,14,18,19,23,24)
    pattern3 = (10,11,12,15,16,17,20,21,22)
    patterns = same_size_patterns((pattern0, pattern1, pattern2, pattern3))
    # We are going to leave some time for the dynamics to converge and then we start the training:
    pair0 = (0,5)   # We leave 5s so that the ode converges to the dynamics
    pair1 = (5,15)  # 10 Seconds to train the first pattern
    pair2 = (15,25) # 10 more seconds to train 2nd pattern
    pair3 = (25,35) # 10 more seconds to trian 3rd pattern
    stim_info = ((pair0,pair1,pair2,pair3),patterns)

    tspan = (0,50)
    tstep = 0.001
    y,t, W = obtaindynamicsHEB(params, tspan, tstep, stim_info, learnparams, True)

    plt.imshow(W, vmin = 0, vmax = 10)
    plt.show()
    print(np.amax(W))
    plt.plot(t[:17000],y[1,:17000])
    plt.show()


