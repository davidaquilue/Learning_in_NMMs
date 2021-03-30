from matfuns import S, psd, normalize, regularity, networkmatrix, crosscorrelation, maxcrosscorrelation
from plotfuns import plot3x3, plotanynet, plotcouplings3x3
from networkJR import obtaindynamicsNET
import numpy as np; from numba import njit

# Collection of different Fitness Functions used to evaluate the individuals of the Genetic Algorithm.

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
    for iter in range(itermax):
        # Three different simulations. In each one, we drive a different node from the first layer. 
        for ii in range(nodes0layer):
            params['forcednode'] = ii
            y , t = obtaindynamicsNET(params, params['tspan'], params['tstep'],2)
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
def fitness_function_amps(individual, params):
    # Primer de tot el que farem és que siguin les màximes diferències
    # What do we want to take into account? Maybe the span of y, maybe the mean of the amplitude, maybe the max or the min?
    # Explore later on, when the 
    params['individual'] = np.array(individual)
    itermax = 5
    nodes0layer = params['tuplenetwork'][0]
    nodeslastlayer = params['tuplenetwork'][-1]
    for it in range(itermax):
        for ii in range(nodes0layer):
            params['forcednode'] = ii
            y,t = obtaindynamicsNET(params, params['tspan'], params['tstep'])

    return fit,

def fitness_function_cross(params, individual):
    # Primer de tot el que farem és que siguin les màximes diferències
    # In the case of amplitudes it will be interesting to see if the GA goes weights with really unplausible values
    # What do we want to take into account? Maybe the span of y, maybe the mean of the amplitude, maybe the max or the min?
    params['individual'] = np.array(individual)
    itermax = 5
    fit = 0
    nodes0layer = params['tuplenetwork'][0]
    nodeslastlayer = params['tuplenetwork'][-1]

    #store_crosscorrelations = np.zeros(nodeslastlayer) # Depending on the algorithm we use we might need a vector/array to store
    for it in range(itermax):
        for ii in range(nodes0layer):
            params['forcednode'] = ii
            y, = obtaindynamicsNET(params, params['tspan'], params['tstep'], v = 2)
            # Again different behaviours can be stored.
            # For the first two assumptions we will need that the first layer and last layer have the same amount of nodes:

            # 1. We want to achieve correlation between the two nodes that are not the node parallel to the one that's been driven
            idx_for_crossc = []
            for jj in range(nodeslastlayer):
                nodelastlayer = params['Nnodes'] - nodeslastlayer + jj
                if ii == jj:
                    continue
                else:
                    idx_for_crossc.append(nodelastlayer)
            fit += maxcrosscorrelation(y[idx_for_crossc[0]], y[idx_for_crossc[1]])

            # 2. We want to achieve correlation between the node that has been driven with its parallel in the last layer
            #for jj in range(nodeslastlayer):
                #if ii == jj:
                    #fit += maxcrosscorrelation(y[ii], y[jj])

            # Or other behaviors that I will think about later on...