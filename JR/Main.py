# Running Genetic Algorithm and Saving Figures in Cluster.
import os
path = os.getcwd()
#print(path) # First time to see how the format of the directory is
import numpy as np; import matplotlib.pyplot as plt
from plotfuns import plot3x3, plotcouplings3x3V2
from networkJR import obtaindynamicsNET
from matfuns import networkmatrix, regularity, maxcrosscorrelation
from multiprocessing import Pool, cpu_count
import GAlgs
from fitfuns import fitness_function_cross

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
params['recurrent'] = True
params['forcednode'] = 1
Nnodes, matrix =  networkmatrix(params['tuplenetwork'], params['recurrent'])
indivsize = 2*np.count_nonzero(matrix)
params['Nnodes'] = Nnodes
params['matrix'] = matrix
params['tstep'] = 0.001
params['tspan'] = (0,40)

# This has to go before the if name = main and before running the program.
toolbox, creator = GAlgs.initiate_DEAP(fitness_function_cross, params, generange = (0,10*C), indsize = indivsize)
    
if __name__ == '__main__':
    last_test = int(os.listdir(path + '/Resultats')[-1])
    print('Last Test: ' + str(last_test))
    testn = last_test + 1
    newfolder = path + '/Resultats/' + str(testn)
    os.makedirs(newfolder)

    print('Actual test: ' + str(testn))
    with Pool(processes= 4) as piscina:
        num_generations = 40; popsize = 20; mutindprob = 0.4; coprob = 0.5; indsize = indivsize
        maxfits, avgfits, bestsols = GAlgs.main_DEAP(num_generations, popsize, mutindprob, coprob, indsize, toolbox, creator, 1, piscina)
        # Plot the maximum fitnesses and average fitnesses of each generation
        gens = np.linspace(1,num_generations, num_generations)
        plt.plot(gens, maxfits)
        plt.plot(gens, avgfits)
        plt.title('Evolution of fitness')
        plt.savefig(newfolder + "/fitness.jpg")
        #plt.show()
        # Obtain the best solution of the population
        solution = bestsols[np.argmax(maxfits)]


        fig = plotcouplings3x3V2(solution, matrix, (0,10*C))
        fig.savefig(newfolder + "/bestweights.jpg")
        np.save(newfolder + '/best_ind', solution)

        norms = np.array([np.sum(weight**2) for weight in bestsols])
        plt.plot(gens, norms)
        plt.title('Evolution of the norm of the max fitness weights')
        plt.savefig(newfolder + "/normevol.jpg")