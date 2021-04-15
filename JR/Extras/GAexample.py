# Main example of Genetic Algorithm use:
import numpy as np; import matplotlib.pyplot as plt
from plotfuns import plot3x3
from networkJR import obtaindynamicsNET
from matfuns import networkmatrix
from fitfuns import fitness_function_reg
from multiprocessing import Pool
import GAlgs

# We first determine the paramaters of the JR model
params = {'A': 3.25, 'B': 22.0, 'v0': 6.0} 
params['a'], params['b'], params['e0'], params['pbar'], params['delta'], params['f'] = 100.0, 50.0, 2.5, 155.0, 65.0, 8.5
C = 133.5
params['C'], params['C1'], params['C2'], params['C3'], params['C4'] = C, C, 0.8*C, 0.25*C, 0.25*C # All dimensionless

params['r'] = 0.56 #mV^(-1)

params['delta'] = 72.09
params['f'] = 8.6
params['stimulation_mode'] = 1

# Now we define the network architecture. Always feedforward and we can decide wether we want recurrencies or not
params['tuplenetwork'] = (3,3,3)
params['recurrent'] = False
params['forcednode'] = 1
Nnodes, matrix =  networkmatrix(params['tuplenetwork'], params['recurrent'])
indivsize = 2*np.count_nonzero(matrix)
params['Nnodes'] = Nnodes
params['matrix'] = matrix
params['tstep'] = 0.001
params['tspan'] = (0,40)

# One has to make sure that the initiation of the genetic algorithm is done before if main...
toolbox, creator = GAlgs.initiate_DEAP(fitness_function_reg, params, generange = (0,100*C), indsize = indivsize)

if __name__ == '__main__':
    with Pool(processes = 4) as piscina:
        num_generations = 5; popsize = 4; mutindprob = 0.2; coprob = 0.5; indsize = indivsize
        maxfits, avgfits, bestsols = GAlgs.main_DEAP(num_generations, popsize, mutindprob, coprob, indsize, toolbox, creator, 1, piscina)
        gens = np.linspace(1, 5, 5)
        plt.plot(gens, maxfits)
        plt.plot(gens, avgfits)
        plt.show()
