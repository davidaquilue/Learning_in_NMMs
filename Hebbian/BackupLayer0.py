# HEBBIAN LEARNING V1. In this module, everything related to the first layer works correctly. Backup file in case the main one starts to go wrong. 9/4/2021

# NMMs Network with weight behaviour governed by Hebbian Learning
import numpy as np; from numba import njit
import matplotlib.pyplot as plt
from matfuns import psd
from random import sample
usefastmath = True

# Luego habria que duplicar esto, pero con pesos inhibidores en la segunda capa. Y además habria que conseguir que cada nodo i de la primera capa se lo 
# mande al nodo i de la segunda.

# Let us begin by developing the first layer and trying to achieve an autoassociative memory. That is that weights are capable of reproducing 
# the same patterns that are produced when

# matrix = NxM array
# Weights matrix: (N*M)x(N*M) array.

 

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
            pbar = np.random.normal(400,50)
        else:
            pbar = np.random.normal(50,10)
        # Coupled intensities, we obtain them from a function.
        # Again, we have to take into account that a row in the W matrix contains the parameters of the nodes that arrive to the row's node.
        pa = couplingvalHEBV1(inp, W[nn,:], C3, e0, r, v0)

        dz0 = A*a*S(y1-y2,e0,r,v0) - 2*a*z0 - a**2*y0
        dy0 = z0
        dz1 = A*a*(pbar + C2*S(C1*y0,e0,r,v0) + pa) - a**2*y1 - 2*a*z1
        dy1 = z1
        dz2 = B*b*(C4*S(C3*y0,e0,r,v0)) - 2*b*z2 - b**2*y2
        dy2 = z2
        dz[nn] = np.array([dz0, dy0, dz1, dy1, dz2, dy2])
    return dz

@njit(fastmath = usefastmath)
def couplingvalHEBV1(inp, Wrow, C3, e0, r, v0):
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

    # Another matrix of outputs will have to be built. We will see how to correctly describe all this.
    x[:,0,:] = x0

    stimtimes = stim_info[0]
    stimpatts = stim_info[1]
    gammaW, vL, Wmax, Tpre = learnparams # Parameters used to train the connection weights.

    tracking = np.array([0.0])

    # Loop. Main algorithm of the Heun Method.
    for n in range(nsteps-1):
        t1 = tvec[n]
        marker = 0 # Used to select which of the patterns will be stimulated, depending on the time.
        for ii,(ti_pat,tf_pat) in enumerate(stimtimes):
            if ti_pat <= t1 and t1 < tf_pat:
                marker = ii
        stimnodes = stimpatts[marker]

        f1 = derivativesHEB(x[:,n,:], t1, funparams, W, stimnodes)
        aux = x[:,n,:] + tstep*f1
        
        t2 = tvec[n+1]
        f2 = derivativesHEB(aux, t2, funparams, W, stimnodes)
        x[:,n+1,:] = x[:,n,:] + 0.5*tstep*(f1 + f2)

        # After each step we update the weights
        if trainingphase:
            if n > Tpre+5: # Some steps need to have passed, we cannot start calculating running averages when no time has run
                # We perform the sweep taking columns as the main dimension and iterating over every element in the column. This way less running avgs have to be computed.
                for jj in range(W.shape[1]):
                    m = np.mean(S(x[jj,n-Tpre:n,3]-x[jj,n-Tpre:n,5],e0,r,v0)) # Presynaptic running average activity
                    for ii in range(W.shape[0]): # Iterating over every element of the column
                        if ii != jj:
                            zii = S(x[ii,n,3]-x[ii,n,5],e0,r,v0)    # Postsynaptic activty 
                            if zii > vL and m > vL: # Only if the pre and postsynaptic activity are over a certain threshold the weights are updated.
                                W[ii,jj] = W[ii,jj] + gammaW*(zii - vL)*(m-vL)*(Wmax-W[ii,jj])
        
        tracking = np.append(tracking,W[9,13]) # A simple way of tracking how one of the weights evolves.
                
    return x, tvec, W, tracking

def obtaindynamicsHEB(params, tspan, tstep, stim_info, learnparams, trainingphase, W = np.array(0)):
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
    if not trainingphase and W.size < 2:
        print('ERROR: When simulating a weight matrix should be provided')
    
    x,t,W, tracking = HeunHEB(x0, tspan, tstep, derivativesHEB, funparams, stim_info, W, learnparams, trainingphase)
    # The output matrix x is of the type (Totalcols, timevalues, Nvariables)
    y1 = x[:,:,3]
    y2 = x[:,:,5]
    return y1-y2,t, W, tracking

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
    gammaW = 0.3; vL = 4.5; Wmax = 10; Tpre = 20
    learnparams = (gammaW, vL, Wmax, Tpre)

    # We are going to train 3 easy patterns
    pattern0 = (-2,-1);                         pair0 = (0,10)  # No stimulation but we have to write something so that numba does not explode
    pattern1 = (0,1,2,5,6,7);                   pair1 = (10,25) 
    pattern2 = (3,4,8,9,13,14,18,19,23,24);     pair2 = (25,40)
    pattern3 = (10,11,12,15,16,17,20,21,22);    pair3 = (40,55)
    patterns = same_size_patterns((pattern0, pattern1, pattern2, pattern3)) # This function makes all the tuples the same size. Necessary for numba.

    stim_info = ((pair0,pair1,pair2,pair3),patterns)

    tspan = (0,55)
    tstep = 0.001
    # Training:
    y,t, W, tracking = obtaindynamicsHEB(params, tspan, tstep, stim_info, learnparams, True)
    print('Training Finished')

    plt.imshow(W, vmin = 0, vmax = Wmax)
    plt.colorbar()
    plt.title('Resulting weight values after training')
    plt.xlabel('Nodes')
    plt.ylabel('Nodes')
    plt.show()

    # Simulating. We will stimulate half of the nodes in each pattern, during different dynamics, and see how the mean of the signals evolve
    print('Starting stimulation')
    print('First pattern')
    pattern0 = (-2,-1); pair0 = (0,5)
    pattern1_sim = tuple(sample(pattern1, k = int(0.7*len(pattern1)))) # Get random selection of nodes in the pattern, 70%
    pair1 = (5,20)
    patterns = same_size_patterns((pattern0,pattern1_sim))

    tspan = (0,20)
    y, t, _, _ = obtaindynamicsHEB(params, tspan, tstep, ((pair0,pair1), patterns), learnparams, False, W)
    dyns = S(y[:,5000:], params['e0'], params['r'], params['v0'])
    dyns = y[:,5000:]
    t = t[5000:]
    vectormeans1 = np.zeros(Totalcols)
    for ii,dyn in enumerate(dyns):
        if ii in pattern1:
            vectormeans1[ii] = np.mean(dyn)
        else:
            vectormeans1[ii] = np.mean(dyn)
    vectormeans1 = vectormeans1.reshape(Shape)
    plt.subplot(131)
    plt.imshow(vectormeans1, vmin = np.amin(vectormeans1), vmax = np.amax(vectormeans1))
    
    print('Second pattern')
    pattern2_sim = tuple(sample(pattern2, k = int(0.7*len(pattern2))))  
    pair2 = (5,20)
    patterns = same_size_patterns((pattern0,pattern2_sim))
    y, t, _, _ = obtaindynamicsHEB(params, tspan, tstep, ((pair0,pair2), patterns), learnparams, False, W)
    dyns = S(y[:,5000:], params['e0'], params['r'], params['v0'])
    dyns = y[:,5000:]
    t = t[5000:]
    vectormeans2 = np.zeros(Totalcols)
    for ii,dyn in enumerate(dyns):
        if ii in pattern1:
            vectormeans2[ii] = np.mean(dyn)
        else:
            vectormeans2[ii] = np.mean(dyn)
    vectormeans2 = vectormeans2.reshape(Shape)
    plt.subplot(132)
    plt.imshow(vectormeans2, vmin = np.amin(vectormeans1), vmax = np.amax(vectormeans1)) 

    print('Third pattern')
    pattern3_sim = tuple(sample(pattern3, k = int(0.7*len(pattern3))))   
    pair3 = (5,20)
    patterns = same_size_patterns((pattern0,pattern3_sim))
    y, t, _, _ = obtaindynamicsHEB(params, tspan, tstep, ((pair0,pair3), patterns), learnparams, False, W)
    dyns = S(y[:,5000:], params['e0'], params['r'], params['v0'])
    dyns = y[:,5000:]
    t = t[5000:]
    vectormeans3 = np.zeros(Totalcols)
    for ii,dyn in enumerate(dyns):
        if ii in pattern1:
            vectormeans3[ii] = np.mean(dyn)
        else:
            vectormeans3[ii] = np.mean(dyn)
    vectormeans3 = vectormeans3.reshape(Shape)
    plt.subplot(133)
    im = plt.imshow(vectormeans3, vmin = np.amin(vectormeans1), vmax = np.amax(vectormeans1))
    plt.colorbar(im)
    plt.show()

    # We want to observe a node that is not in the stimulated pattern:
    ind = 0; n = 0
    while ind != 1:
        if pattern3[n] in pattern3_sim: n += 1
        else: ind = 1
    plt.plot(t, dyns[pattern3_sim[0]], label = 'Stimulated')
    plt.plot(t, dyns[pattern3[n]], label = 'non-Stimulated')
    plt.plot(t, dyns[0],  label = 'Pattern 1 ')
    plt.plot(t, dyns[13], label = 'Pattern 2')
    plt.title('Dynamics of pattern 3')
    plt.legend(loc = 'best')
    plt.show()
    f, psds = psd(dyns[17], tstep)
    plt.plot(f, psds)
    plt.xlim((0,30))
    plt.show()

    f, psds = psd(dyns[0], tstep)
    plt.plot(f, psds)
    plt.xlim((0,30))
    plt.show()


