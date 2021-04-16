'''Collection of fitness functions that can be used in the Genetic Algorithm.

Functions included: fitness_function_cross_V2, fitness_function_reg, fitness_functions_amps, 
fitness_functions_cross, fitness_functions_psds and complementary functions'''
from matfuns import S, psd, normalize, regularity, networkmatrix, crosscorrelation, maxcrosscorrelation
from plotfuns import plot3x3, plotanynet, plotcouplings3x3
from networkJR import obtaindynamicsNET
import numpy as np; from numba import njit
from itertools import combinations

# MAIN FITNESS FUNCTION THAT WILL BE USED DURING THE DEVELOPMENT OF THE THESIS
def fitness_function_cross_V2(params, individual):
    '''This fitness function is multiobjective'''
    params['individual'] = np.array(individual) # Set of weights
    nodes_in_lastlayer = params['tuplenetwork'][-1]; idx_nodes_lastlayer = params['Nnodes'] - nodes_in_lastlayer # Indexes for correlation computing
    fit0 = 0; fit1 = 0; fit2 = 0
    iters = 5
    for it in range(iters):
        for ii, (pair,unsync) in enumerate(zip(params['pairs'], params['unsync'])): # Iterating over everyone of the three different combinations

            params['signals'] = params['All_signals'][ii] # Change between the three different sets of signals available. Have to be obtained from outside because of their random description
            y, _ = obtaindynamicsNET(params, params['tspan'], params['tstep'], v = 3)

            idxcomp1 = idx_nodes_lastlayer + pair[0]; idxcomp2 = idx_nodes_lastlayer + pair[0]; idxunsync = idx_nodes_lastlayer + unsync # More index algebra

            # Fitness computed for every situation
            if ii == 0:
                fit0 += maxcrosscorrelation(y[idxcomp1], y[idxcomp2]) - (maxcrosscorrelation(y[idxcomp1], y[idxunsync]) + maxcrosscorrelation(y[idxcomp2], y[idxunsync]))/2
            elif ii == 1:
                fit1 += maxcrosscorrelation(y[idxcomp1], y[idxcomp2]) - (maxcrosscorrelation(y[idxcomp1], y[idxunsync]) + maxcrosscorrelation(y[idxcomp2], y[idxunsync]))/2
            elif ii == 2:
                fit2 += maxcrosscorrelation(y[idxcomp1], y[idxcomp2]) - (maxcrosscorrelation(y[idxcomp1], y[idxunsync]) + maxcrosscorrelation(y[idxcomp2], y[idxunsync]))/2

    return fit0/iters, fit1/iters, fit2/iters


# OTHER FITNESS FUNCTIONS USED DURING THE EXPLORATION OF POSSIBILITIES (can be ignored for the moment)

# USING REGULARITIES TO ENCODE INFORMATION 
def fitness_function_reg(params, individual):
    # At the moment, individual are lists.
    individual = np.array(individual)
    params['individual'] = individual

    nodes0layer = params['tuplenetwork'][0]; nodeslastlayer = params['tuplenetwork'][-1]
    fit = 0
    regs = np.zeros((3,3)); arraybits = np.zeros((3,3))
    itermax = 10
    # We try to improve reliability of our results averaging over itermax different iterations.
    for it in range(itermax):
        # Three different simulations. In each one, we drive a different node from the first layer. 
        for ii in range(nodes0layer):
            params['forcednode'] = ii
            y , _ = obtaindynamicsNET(params, params['tspan'], params['tstep'],2)
            # For each node of the first layer driven, we store the regularities of the last layer in a different row of regs.
            for ll,yy in enumerate(y[-nodeslastlayer:]):
                regs[ii,ll] += regularity(yy)
      
    arraybits[np.where(regs>0.9*itermax)] = 1 # A matrix where each row i contains the 1 if very regular, 0 if not, of the last three nodes when forcing node i. i = 0,1,2

    # Now we want to maximize the differences between the results in order to observe a clear code. This can be done in many different ways:

    # 1. Calculating the determinant on the bits, if they repeat it will be 0
    #det = np.linalg.det(arraybits)
    # And as a first assumption we will want the most regularity possible
    #fit = (np.abs(det)+0.1)*np.sum(arraybits)

    # 2. Trying to maximize the differences in regularities of the last layer when forcing different nodes
    #fit = np.sum((regs[0]-regs[1])**2 + (regs[1]- regs[2])**2 + (regs[2] - regs[0])**2)

    # 3. Trying to maximize the differences between pairs of nodes.
    #fit = np.sum((regs[0,0:2]-regs[1,0:2])**2 + (regs[0,1:3]-regs[1,1:3])**2 + (regs[1,0:2]-regs[2,0:2])**2 + (regs[1,1:3]-regs[2,1:3])**2 + (regs[0,0:2]-regs[2,0:2])**2 + (regs[0,1:3]-regs[2,1:3])**2)

    # 4. Transform the bit-type rows to binary and compute the differences between the integers obtained...
    # Adding as well some kind of penalization can also help

    # 5. Maximize regularity of the node parallel to the driven node in the first layer. Only when first and last layer have the same number of nodes
    regs = regs/itermax
    reg_wanted_to1 = regs[0,0] + regs[1,1] + regs[2,2]
    # And maximal difference between the othes
    diffs = (regs[0,0]-regs[0,1])**2 + (regs[0,0]-regs[0,2])**2 + (regs[1,1]-regs[1,0])**2 + (regs[1,1]-regs[1,2])**2 + (regs[2,2]-regs[2,0])**2 + (regs[2,2]-regs[2,1])**2
    fit = reg_wanted_to1 + diffs

    if fit <= 0:
        return 0,
    else:
        return fit, # For the DEAP algorithm we have to output a tuple of values thus the comma.


# USING AMPLITUDES TO ENCODE INFORMATION
def fitness_function_amps(params, individual):
    params['individual'] = np.array(individual)
    itermax = 5
    nodes0layer = params['tuplenetwork'][0]
    nodeslastlayer = params['tuplenetwork'][-1]
    fit = 0
    store_amplitudes = np.zeros((nodes0layer,nodeslastlayer)) # In this matrix we will store the characteristic of the amplitude that we want to obtain from a node.
    for it in range(itermax):
        for ii in range(nodes0layer):
            params['forcednode'] = ii
            y, _ = obtaindynamicsNET(params, params['tspan'], params['tstep'], v = 2)
            for jj in range(nodeslastlayer):
              nodelastlayer = params['Nnodes'] - nodeslastlayer + jj
              # We can store different behaviours of the dynamics of the nodes

              #1. Mean of signal
              store_amplitudes[ii,jj] += np.mean(y[nodelastlayer])
              #2. Span of signal
              #store_amplitudes[ii,jj] = np.amax(y[nodelastlayer]) - np.amin(y[nodelastlayer])
              #3. Max of signal
              #store_amplitudes[ii,jj] = np.amax(y[nodelastlayer])
              #4. Min of signal 
              #store_amplitudes[ii,jj] += np.amin(y[nodelastlayer])

        # And now we have to define how we want the amplitudes to be
        # 1. We want that for each forced node, the differences between the means of the signal are as large as possible
        #fit += diff_between_driven(store_amplitudes) # Maybe divide by itermax

        # 2. Maybe maximize the difference between the nodes for a same forced node:
        fit += diff_between_same(store_amplitudes)

    # 3. Or other alternatives like combining 1 and 2, coding in binary and trying to obtain different numbers, etc..
    return fit/itermax,


# USING CROSS-CORRELATIONS TO ENCODE INFORMATION
def fitness_function_cross(params, individual):
    params['individual'] = np.array(individual)
    itermax = 5
    fit = 0

    #store_crosscorrelations = np.zeros(nodeslastlayer) # Depending on the algorithm we use we might need a vector/array to store
    for it in range(itermax):
        y, _ = obtaindynamicsNET(params, params['tspan'], params['tstep'], v = 3)
        # Again different behaviours can be stored.
        # For the first two assumptions we will need that the first layer and last layer have the same amount of nodes:

        # Testing if GA can synchronize two while they are desynchronized with the other
        fit += maxcrosscorrelation(y[6], y[8]) - (maxcrosscorrelation(y[6], y[7]) + maxcrosscorrelation(y[7], y[8]))/2

        # 2. We want to achieve correlation between the node that has been driven with its parallel in the last layer
        #for jj in range(nodeslastlayer):
            #if ii == jj:
                #fit += maxcrosscorrelation(y[ii], y[jj])

        # Or other behaviors that I will think about later on...

    return fit/(itermax),

# USING PSDs TO ENCODE INFORMATION:
def fitness_function_psds(params, individual):
    params['individual'] = np.array(individual)
    itermax = 5
    nodes0layer = params['tuplenetwork'][0]
    nodeslastlayer = params['tuplenetwork'][-1]

    maxFs = np.zeros((nodes0layer,nodeslastlayer)) # In this matrix we will store the characteristic of the amplitude that we want to obtain from a node.
    for it in range(itermax):
        for ii in range(nodes0layer):
            params['forcednode'] = ii
            y,_ = obtaindynamicsNET(params, params['tspan'], params['tstep'], v = 2)
            for jj in range(nodeslastlayer):
              nodelastlayer = params['Nnodes'] - nodeslastlayer + jj
              yy = y[nodelastlayer]
              f, PSD = psd(yy,params['tstep'])
              # We obtain the frequency at which the spectral power density is maximal
              maxFs[ii,jj] = np.abs(f[np.argmax(PSD)])
    # Then we can proceed to the same evaluation af we have done previously, whether differences between nodes or other
    # types of differences. I should think more deeply about these functions  


# COMPLEMENTARY FUNCTIONS FOR FITNESS-FUNCTION EVALUATIONS

# Matrix should be a (nodesfirstlayer,nodeslastlayer)
def diff_between_driven(matrix):
  '''The fitness is the accumulated difference between behaviors when driving different nodes'''
  fit = 0
  aux = np.arange(0, matrix.shape[0])
  for idx1, idx2 in combinations(aux,2):
    fit += np.sum((matrix[idx1]-matrix[idx2])**2)
  return fit

def diff_between_same(matrix):
  '''The fitness is the accumulated difference between last layer's nodes when driving one node of the first layer'''
  fit = 0
  aux = np.arange(0, matrix.shape[1])
  for ii in range(matrix.shape[0]):
    row = matrix[ii]
    for idx1, idx2 in combinations(aux,2):
      fit += (row[idx1]-row[idx2])**2
  return fit

# This method might be really susceptible to local minima since differences get quite averaged. Plus, they could be behaving in a same manner for each driven node.
# Maybe a combination of the two would be a good idea.