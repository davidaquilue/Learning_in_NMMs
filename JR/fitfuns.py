from matfuns import S, psd, normalize, regularity, networkmatrix, crosscorrelation
from plotfuns import plot3x3, plotanynet, plotcouplings3x3
from networkJR import obtaindynamicsNET_V1
import numpy as np; from numba import njit

# Collection of different Fitness Functions used to evaluate the individuals of the Genetic Algorithm.

# USING REGULARITIES TO ENCODE INFORMATION 
def fitness_function_reg(individual, params):
    # At the moment, individual are lists.
    individual = np.array(individual)
    params['individual'] = individual

    nodes0layer = tuplenetwork[0]; nodeslastlayer = tuplenetwork[-1]
    fit = 0
    regs = np.zeros((3,3)); arraybits = np.zeros((3,3))
    itermax = 10
    # We try to improve reliability of our results averaging over itermax different iterations.
    for iter in range(itermax):
        # Three different simulations. In each one, we drive a different node from the first layer. 
        for ii in range(nodes0layer):
            params['forcednode'] = ii
            y , t = obtaindynamicsNET_V1(params, tspan, tstep)
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
    reg_wanted_to1 = regs[0,0] + regs[1,1] + regs[2,2]
    # And maximal difference between the othes
    diffs = (regs[0,0]-regs[0,1])**2 + (regs[0,0]-regs[0,2]) + (regs[1,1]-regs[1,0])**2 + (regs[1,1]-regs[1,2])**2 + (regs[2,2]-regs[2,0])**2 + (regs[2,2]-regs[2,1])**2
    fit = reg_wanted_to1 + fit

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
    nodes0layer = tuplenetwork[0]
    nodeslastlayer = tuplenetwork[-1]
    for it in range(itermax):
        for ii in range(nodes0layer):
            params['forcednode'] = ii
            y,t = obtaindynamicsNETGA(params, tspan, tstep)

    return fit,