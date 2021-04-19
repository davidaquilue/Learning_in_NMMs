'''Main Script. It runs the Genetic Algorithm and Saves the results. Modifications should be
performed in this module's code.'''

from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt

from matfuns import networkmatrix, maxcrosscorrelation, creating_signals # network_matrix_exc_inh # This last function will be used later on
from plotfuns import plotcouplings3x3V2, plot3x3_signals
from networkJR import obtaindynamicsNET
from fitfuns import fitness_function_cross_V2
from filefuns import check_create_results_folder, test_folder
import galgs

# We first determine the paramaters of the JR model
params = {'A': 3.25, 'B': 22.0, 'v0': 6.0} 
params['a'], params['b'], params['e0'] = 100.0, 50.0, 2.5
params['pbar'], params['delta'], params['f'] = 155.0, 65.0, 8.5

params['C'] = C = 133.5
params['C1'], params['C2'], params['C3'], params['C4'] = C, 0.8*C, 0.25*C, 0.25*C # Dimensionless

params['r'] = 0.56 #mV^(-1)

params['delta'] = 72.09
params['f'] = 8.6
params['stimulation_mode'] = 1

# Now we define the network architecture. Always feedforward and we can decide wether we want
# recurrencies or not
params['tuplenetwork'] = (3, 3, 3)
params['recurrent'] = False
params['forcednodes'] = (0, 1, 2)
Nnodes, matrix = networkmatrix(params['tuplenetwork'], params['recurrent'])

# For further tests
# Nnodes, matrix_exc, matrix_inh = networkmatrix_exc_inh(params['tuplenetwork'], params['recurrent'], v = 0)
# indivsize = np.count_nonzero(matrix_exc) + np.count_nonzero(matrix_inh)

indivsize = 2*np.count_nonzero(matrix)
params['Nnodes'] = Nnodes
params['matrix'] = matrix
params['tstep'] = 0.001
params['tspan'] = (0,40)

# Now we define the settings for the construction of the input signals.
t = np.linspace(params['tspan'][0], params['tspan'][1], int((params['tspan'][1] - params['tspan'][0])/params['tstep']))
amp = 250
dc = 120
f = 2
params['pairs'] = ((0, 1), (0,2), (1, 2))
params['unsync'] = (2, 1, 0)
# For each pair corresponds a matrix of signals. In total: a matrix o matrices
All_signals = np.zeros(shape = (3, 3, t.size))
for ii, pair in enumerate(params['pairs']):
    All_signals[ii] = creating_signals(t, amp, dc, f, pair)
params['All_signals'] = All_signals


############################ GENETIC ALGORITHM PARAMETER SETUP ####################################
num_generations = 10
popsize = 10        # Population size
mutindprob = 0.2    # Probability that an individual undergoes mutation
coprob = 0.5        # Crossover probability
maxgene = 0.4*C     # Maximum coupling value of a connection
mingene = 0         # Minimum coupling value of a connection
par_processes = 4   # How many cores will be used in order to parallelize the GA.

# Initialization of the necessary GA functions:
# this has to go before the if name = main and before running the algorithm.
toolbox, creator = galgs.initiate_DEAP(fitness_function_cross_V2, params, generange = (mingene, maxgene), indsize = indivsize, v = 2)
    
if __name__ == '__main__':
    # Folder management
    results_dir = check_create_results_folder()
    newfolder = test_folder(results_dir)

    with Pool(processes= par_processes) as piscina:
        # Running GA
        maxfits, avgfits, bestsols = galgs.main_DEAP(num_generations, popsize, mutindprob, coprob, indivsize, toolbox, creator, 1, piscina, v = 2)

        # Plot the maximum fitnesses and average fitnesses of each generation
        # ESCRIURE FUNCIO A PLOTFUNS D'AIXO
        gens = np.linspace(1,num_generations, num_generations)
        plt.plot(gens, maxfits[:,0], label = 'fit0')
        plt.plot(gens, maxfits[:,1], label = 'fit1')
        plt.plot(gens, maxfits[:,2], label = 'fit2')
        plt.plot(gens, avgfits, label = 'avg')
        plt.title('Evolution of fitness')
        plt.legend(loc = 'best')
        plt.savefig(newfolder + "/fitness.jpg")
        plt.show()

        # Obtain the best solution of the population
        maxfits_avg = (maxfits[:,0] + maxfits[:,1] + maxfits[:,2])/3
        solution = bestsols[np.argmax(maxfits_avg)]

        # Show the coupling matrices
        fig = plotcouplings3x3V2(solution, matrix, (mingene, maxgene))
        fig.savefig(newfolder + "/bestweights.jpg")
        plt.show()
        np.save(newfolder + '/best_ind', solution)
        np.save(newfolder + '/signals', params['All_signals']) # In case we wanted to repeat the dynamics computation

        # Show the evolution of the norm of the best solutions as a function of generations
        norms = np.array([np.sum(weight**2) for weight in bestsols])
        plt.plot(gens, norms)
        plt.title('Evolution of the norm of the max fitness weights')
        plt.savefig(newfolder + "/normevol.jpg")
        plt.show()

        # And let's observe the resulting dynamics of the best individual
        solution = np.array(solution)
        params['individual'] = solution
        for ii in range(3):
            params['signals'] = params['All_signals'][ii]
            pair = params['pairs'][ii]
            synch_pair = (Nnodes - params['tuplenetwork'][-1] + pair[0], Nnodes - params['tuplenetwork'][-1] + pair[1])
            y, t = obtaindynamicsNET(params, params['tspan'], params['tstep'], 3)

            typeplot = 'small'
            fig = plot3x3_signals(y,t, typeplot, params['tstep'], params['All_signals'][ii])
            saving = newfolder + '/Dynamics' + str(ii) + typeplot + '.png'
            fig.savefig(saving)
            plt.show()
            print('\n Nodes %i and %i should be synchronized:' %synch_pair)
            print('Correlation 6 with 7: ' + str(maxcrosscorrelation(y[6], y[7])))
            print('Correlation 6 with 8: ' + str(maxcrosscorrelation(y[6], y[8])))
            print('Correlation 7 with 8: ' + str(maxcrosscorrelation(y[7], y[8])))
        