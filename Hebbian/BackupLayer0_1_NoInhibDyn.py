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
    WL0L1 = params['WL0L1']
    KL0L1 = params['KL0L1']

    return (A, B, v0, a, b, e0 , C1, C2, C3, C4, r, Totalcols, WL0L1, KL0L1)

@njit(fastmath = usefastmath)
def derivativesHEB(inp0, inp1, t, paramtup, W, K, O, stimnodes):
    ''' Returns derivatives of the 6 variables of the model for each node
    Inputs:
    inps:       A (N_nodes,6) matrices. inp0 correspods to the nodes of the first layer while inp1 corresponds to the nodes in the second layer.
    t:          step of time for which the values are those of inp
    paramtup:   Tuple containing all the parameters
    W:          First layers excitatory weights matrix. We will have to also input the second layer weight matrix.
    stimpat:    Which pattern is being stimulated at the moment: 1,2 or 3

    Output:
    dz:     A (N_nodes,6) matrix, containing all the derivatives of the variables.
    '''
    A, B, v0, a, b, e0 , C1, C2, C3, C4, r, Totalcols, WL0L1, KL0L1 = paramtup

    # Now the input will be a matrix where each row i corresponds to the variables (z0,y0,z1,y1,z2,y2) of each node i.
    dinp0 = np.zeros_like(inp0)
    dinp1 = np.zeros_like(inp1)
    # Now we obtain the derivatives of every variable for every node.
    for ii in range(2):
        if ii == 0: inp = inp0
        elif ii == 1: inp = inp1    
        for nn in range(Totalcols):
            x = inp[nn] # This will extract the row corresponding to each node.
            z0 = x[0]
            y0 = x[1]
            z1 = x[2]
            y1 = x[3]
            z2 = x[4]
            y2 = x[5]

            if (ii == 0) and (nn in stimnodes): pbar = np.random.normal(500,50)# Stimulated nodes are only those of the first layer
            else:   pbar = np.random.normal(50,10)
            # Coupled intensities, we obtain them from a function.
            # Again, we have to take into account that a row in the W matrix contains the parameters of the columns that arrive to the node.
            # I will have to decide if I define a function for every layer. I guess so because layer 2 contains inhibitory responses aswell, meaning another matrix.
            if ii == 0:
                pa = coupval_layer0(inp0, W[nn,:], C3, e0, r, v0)
            elif ii == 1:
                pa, pb = coupval_layer1(inp0[nn],inp1, WL0L1, KL0L1, K[nn,:], O[nn,:], C3, e0, r, v0) # K and D are inhibitory weights that will have to be trained in the Heun method.
            

            dz0 = A*a*S(y1-y2,e0,r,v0) - 2*a*z0 - a**2*y0
            dy0 = z0
            dz1 = A*a*(pbar + C2*S(C1*y0,e0,r,v0) + pa) - a**2*y1 - 2*a*z1
            dy1 = z1
            dz2 = B*b*(C4*S(C3*y0,e0,r,v0) + pb) - 2*b*z2 - b**2*y2
            dy2 = z2
            if ii == 0:
                dinp0[nn] = np.array([dz0, dy0, dz1, dy1, dz2, dy2])
            elif ii == 1:
                dinp1[nn] = np.array([dz0, dy0, dz1, dy1, dz2, dy2])
    return dinp0, dinp1

@njit(fastmath = usefastmath)
def coupval_layer0(inp0, Wrow, C3, e0, r, v0):
    ''' Obtains the effects of the coupling for each node. First layer of Hebbian learning only coupling is excitatory'''
    pa = 0
    # We obtain the contribution of each node.
    for node,weightval in enumerate(Wrow):
        if weightval != 0:  #by this we have acces to the nodes to which the current node is linked to
            pa += weightval*S(inp0[node,3]-inp0[node,5], e0, r, v0)
    return pa

@njit(fastmath = usefastmath)
def coupval_layer1(inp0samenode, inp1, WL0L1, KL0L1, Krow, Orow, C3, e0, r, v0):
    '''Obtains the effects of the coupling for each node. Second layer excitatory/inhibitory coupling from first layer and inhibitory coupling within second layer.'''
    pa = WL0L1 * S(inp0samenode[3]-inp0samenode[5], e0, r, v0) # WL0L1 * y pyramidal same node
    pb = KL0L1 * S(inp0samenode[3]-inp0samenode[5], e0, r, v0) # KL0L1 * y pyramidal same node
    for node, (Kval, Oval) in enumerate(zip(Krow, Orow)):
        if Kval != 0 or Oval != 0: 
            pb += Kval*S(inp1[node,3]-inp1[node,5], e0, r, v0) + Oval*S(inp1[node,3]-inp1[node,5], e0, r, v0)
    return pa, pb


@njit(fastmath = usefastmath)
def HeunHEB(tspan, tstep, fun, funparams, stim_info, W, K, O, learnparams, trainingphase):
    '''
    Heun method of integrating equations, this time without noise simulation. 
    Inputs:
    tspan:       Tuple: (t0, tf).
    tstep:       Timestep taken.
    fun:         Function f from dy/dt = f(t,y).
    funparams:   Parameters that the function f needs.
    stim_info:   A tuple containing the times and patterns that will be stimulated [((t0,t1),(t1,t2),(),..), ((No pattern)(pattern1), (pattern2),...)]
    W:           The weights matrix that we will update after every step with the Hebbian learning algorithm
    learnparams: Parameters that will be used in the learning algorithm
    '''
    x0_IC = np.random.normal(10,5,size=(Totalcols,6))
    x1_IC = np.random.normal(10,5,size=(Totalcols,6)) # Random normally distributed IC
    t0 = tspan[0]
    tf = tspan[1]
    nsteps = int((tf-t0)/tstep) # Number of total steps that will be taken.

    tvec = np.linspace(t0,tf,nsteps)
    nnodes = x0_IC.shape[0]
    nvars = x0_IC.shape[1]
    x0 = np.zeros((nnodes,nsteps,nvars)) 
    x1 = np.zeros((nnodes,nsteps,nvars)) # Matrix of outputs. Luckily they are identical.

    x0[:,0,:] = x0_IC
    x1[:,0,:] = x1_IC

    stimtimes = stim_info[0]
    stimpatts = stim_info[1]
    gammaW, gammaK, gammaO, vL, vU, Wmax, Kmax, Omax, Tpre = learnparams # Parameters used to train the connection weights.

    tracking = np.array([0.0])

    # Loop. Main algorithm of the Heun Method.
    for n in range(nsteps-1):
        t1 = tvec[n]
        marker = 0 # Used to select which of the patterns will be stimulated, depending on the time.
        for ii,(ti_pat,tf_pat) in enumerate(stimtimes):
            if ti_pat <= t1 and t1 < tf_pat:
                marker = ii
        stimnodes = stimpatts[marker]

        # Two matrices thus two arrays to be updated every timestep. First subindex is the Heun subindex. Second subindex is the layer indicator.
        f1_0, f1_1 = derivativesHEB(x0[:,n,:], x1[:,n,:], t1, funparams, W, K, O, stimnodes)
        aux0 = x0[:,n,:] + tstep*f1_0
        aux1 = x1[:,n,:] + tstep*f1_1
        
        t2 = tvec[n+1]
        f2_0, f2_1 = derivativesHEB(aux0, aux1, t2, funparams, W, K, O, stimnodes)
        x0[:,n+1,:] = x0[:,n,:] + 0.5*tstep*(f1_0 + f2_0)
        x1[:,n+1,:] = x1[:,n,:] + 0.5*tstep*(f1_1 + f2_1)

        # Hebbian training:
        _, _, v0, _, _, e0 , _,  _, C3, _, r, _, _, _ = funparams 
        if trainingphase:
            if n > Tpre+5: # Some steps need to have passed, we cannot start calculating running averages when no time has run
                # We first start by updating the values of the W matrix, corresponding to layer 0:
                for jj in range(W.shape[1]):# We iterate first over columns. This will allow us to compute way less running averages.
                    m = np.mean(S(x0[jj,n-Tpre:n,3]-x0[jj,n-Tpre:n,5],e0,r,v0)) # Presynaptic running average activity
                    for ii in range(W.shape[0]): # Iterating over every element of the column
                        if ii != jj:
                            zii = S(x0[ii,n,3]-x0[ii,n,5],e0,r,v0)    # Postsynaptic activity of pyramidal neurons
                            if zii > vL and m > vL: # Only if the pre and postsynaptic activity are over a certain threshold the weights are updated.
                                W[ii,jj] = W[ii,jj] + gammaW*(zii - vL)*(m-vL)*(Wmax-W[ii,jj])
                
                # Then we update the values of the K matrix, now the variables are those of layer 1: 
                ## Less for loops could be used, if the algorithm turns very slow it can be optimized but it will be harder to read ##
                for jj in range(K.shape[1]):
                    m = np.mean(S(x1[jj,n-Tpre:n,3]-x1[jj,n-Tpre:n,5],e0,r,v0)) # Presynaptic running average activity of pyramidal neurons
                    for ii in range(K.shape[0]):
                        if ii != jj:
                            zii = S(C3*x1[ii,n,1],e0,r,v0)    # Postsynaptic activity of inhibitory neurons
                            if zii > vL and m > vL: 
                                K[ii,jj] = K[ii,jj] + gammaK*(zii - vL)*(m-vL)*(Kmax-K[ii,jj])
                # K should look the same way than W, but taking action in layer 1.

                # Finally we update the values of the O matrix, following AntiHebbian rules. The function of this matrix is that when a pattern is active,
                # it inhibits all the others.
                for jj in range(O.shape[1]):
                    mp = np.mean(S(x1[jj,n-Tpre:n,3]-x1[jj,n-Tpre:n,5],e0,r,v0)) # Presynaptic running average activity of pyramidal neurons
                    for ii in range(O.shape[0]):
                        if ii != jj:
                            mi = np.mean(S(C3*x1[ii,n-Tpre:n,1],e0,r,v0))   # Postsynaptic running average of inhibitory so that columns in the same pattern don't inhibit each other.
                            if mi < vU and mp > vL:  # AntiHebbian rule
                                O[ii,jj] = O[ii,jj] + gammaO*(vU - mi)*(mp - vL)*(Omax-O[ii,jj])


        
        tracking = np.append(tracking,W[9,13]) # A simple way of tracking how one of the weights evolves.
                
    return x0, x1, tvec, W, K, O, tracking

def obtaindynamicsHEB(params, tspan, tstep, stim_info, learnparams, trainingphase, W = np.array(0), K = np.array(0), O = np.array(0)):
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
    Totalcols = params['Totalcols']
    if trainingphase:
        W = np.zeros((Totalcols, Totalcols))
        K = np.zeros_like(W)
        O = np.zeros_like(W)
    if not trainingphase and (W.size < 2 or K.size < 2 or O.size < 2):
        print('ERROR: When simulating a weight matrix should be provided')
    
    x0, x1, t, W, K, O, tracking = HeunHEB(tspan, tstep, derivativesHEB, funparams, stim_info, W, K, O, learnparams, trainingphase)
    # The output matrix x is of the type (Totalcols, timevalues, Nvariables)
    y0_1 = x0[:,:,3]
    y0_2 = x0[:,:,5] # We extract the pyramidal behavior of nodes in first layer by doing this

    y1_1 = x1[:,:,3]
    y1_2 = x1[:,:,5] # We extract the pyramidal behavior of nodes in second layer by doing this
    return y0_1 - y0_2, y1_1 - y1_2, t, W, K, O, tracking

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
    gammaW = 0.3; gammaK = 0.3; gammaO =  0.3; vL = 4.5; vU = 0.01; Wmax = 10; Kmax = 1; Omax = 1; Tpre = 20 
    learnparams = gammaW, gammaK, gammaO, vL, vU, Wmax, Kmax, Omax, Tpre
    WL0L1 = 100;    params['WL0L1'] = WL0L1
    KL0L1 = 1;    params['KL0L1'] = KL0L1


    # We are going to train 3 easy patterns
    pattern0 = (-2,-1);                         pair0 = (0,5)  # No stimulation but we have to write something so that numba does not explode
    pattern1 = (0,1,2,5,6,7);                   pair1 = (5,15) 
    pattern2 = (3,4,8,9,13,14,18,19,23,24);     pair2 = (15,25)
    pattern3 = (10,11,12,15,16,17,20,21,22);    pair3 = (25,35)
    patterns = same_size_patterns((pattern0, pattern1, pattern2, pattern3)) # This function makes all the tuples the same size. Necessary for numba.

    stim_info = ((pair0,pair1,pair2,pair3),patterns)

    tspan = (0,35)
    tstep = 0.001
    # Training:
    y0, y1, t, W, K, O, tracking = obtaindynamicsHEB(params, tspan, tstep, stim_info, learnparams, True)
    print('Training Finished')
    plt.imshow(W)
    plt.title('W matrix')
    plt.colorbar()
    plt.show()

    plt.imshow(K)
    plt.title('K matrix')
    plt.colorbar()
    plt.show()

    plt.imshow(O)
    plt.title('O matrix')
    plt.colorbar()
    plt.show()

    print('Third pattern')
    pattern3_sim = tuple(sample(pattern3, k = int(0.7*len(pattern3))))   
    pair3 = (5,20)
    patterns = same_size_patterns((pattern0,pattern3_sim))

    tspan = (0,10)
    y0, y1, t, _, _, _, _ = obtaindynamicsHEB(params, tspan, tstep, ((pair0,pair3), patterns), learnparams, False, W, K, O)
    dyns0 = y0[:,5000:]
    dyns1 = y1[:,5000:]
    t = t[5000:]
    vectormeans30 = np.zeros(Totalcols)
    vectormeans31 = np.zeros(Totalcols)
    for ii in range(Totalcols):
        vectormeans30[ii] = np.mean(dyns0[ii])
        vectormeans31[ii] = np.mean(dyns1[ii])
    vectormeans30 = vectormeans30.reshape(Shape)
    vectormeans31 = vectormeans31.reshape(Shape)
    plt.subplot(121)
    im1 = plt.imshow(vectormeans30, vmin = np.amin(vectormeans30), vmax = np.amax(vectormeans30))
    plt.colorbar(im1)
    plt.subplot(122)
    im = plt.imshow(vectormeans31, vmin = np.amin(vectormeans31), vmax = np.amax(vectormeans31))
    plt.colorbar(im)
    plt.show()
    
    # We want to observe a node that is not in the stimulated pattern:
    ind = 0; n = 0
    while ind != 1:
        if pattern3[n] in pattern3_sim: n += 1
        else: ind = 1
    plt.plot(t, dyns0[pattern3_sim[0]], label = 'Stimulated')
    plt.plot(t, dyns0[pattern3[n]], label = 'non-Stimulated')
    plt.plot(t, dyns1[pattern3_sim[0]], label = 'Layer 1, stimulated')
    plt.plot(t, dyns1[pattern3[n]], label = 'Layer1, non-stimulated')
    plt.plot(t, dyns1[0], label = 'Layer 1. Pattern 1')
    plt.plot(t, dyns0[0],  label = 'Pattern 1 ')
    plt.plot(t, dyns0[13], label = 'Pattern 2')
    plt.title('Dynamics of pattern 3')
    plt.legend(loc = 'best')
    plt.show()
    


