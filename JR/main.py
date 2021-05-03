'''Main Script. It runs the Genetic Algorithm and Saves the results. Modifications should be
performed in this module's code.'''

from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matfuns import networkmatrix, maxcrosscorrelation, creating_signals, networkmatrix_exc_inh  # This last function will be used later on
from plotfuns import plotcouplings3x3V2, plot3x3_signals, plot_genfit, plotanynet, plotinputsoutputs
from networkJR import obtaindynamicsNET
from fitfuns import fitness_function_cross_V2 
from filefuns import check_create_results_folder, test_folder
import galgs

# We first determine the paramaters of the JR model
params = {'A': 3.25, 'B': 22.0, 'v0': 6.0}
params['a'], params['b'], params['e0'] = 100.0, 50.0, 2.5
params['pbar'], params['delta'], params['f'] = 155.0, 65.0, 8.5

params['C'] = C = 133.5
params['C1'], params['C2'], params['C3'], params['C4'] = C, 0.8*C, 0.25*C, 0.25*C  # Dimensionless

params['r'] = 0.56  # mV^(-1)

params['delta'] = 72.09
params['f'] = 8.6
params['stimulation_mode'] = 1

# Now we define the network architecture. Always feedforward and we can decide wether we want
# recurrencies or not
params['tuplenetwork'] = (3, 3, 3)
params['recurrent'] = False
params['forcednodes'] = (0, 1, 2)
# Nnodes, matrix = networkmatrix(params['tuplenetwork'], params['recurrent'])
# indivsize = 2*np.count_nonzero(matrix)
# params['matrix'] = matrix

# For further tests
Nnodes, matrix_exc, matrix_inh = networkmatrix_exc_inh(params['tuplenetwork'], params['recurrent'], v = 0) # v indicates which of the weight tests one wants to perform
# (see the function in the matfuns script for further information on the tests)
indivsize = np.count_nonzero(matrix_exc) + np.count_nonzero(matrix_inh)

params['Nnodes'] = Nnodes
params['matrix_exc'] = matrix_exc
params['matrix_inh'] = matrix_inh
params['tstep'] = 0.001
params['tspan'] = (0, 40)

# Now we define the settings for the construction of the input signals.
t = np.linspace(params['tspan'][0], params['tspan'][1], int((params['tspan'][1] - params['tspan'][0])/params['tstep']))
amp = 250
dc = 120
f = 2
params['pairs'] = ((0, 1), (0, 2), (1, 2))
params['unsync'] = (2, 1, 0)
# For each pair corresponds a matrix of signals. In total: a matrix o matrices
All_signals = np.zeros(shape=(3, 3, t.size))
for ii, pair in enumerate(params['pairs']):
    All_signals[ii] = creating_signals(t, amp, dc, f, pair)
params['All_signals'] = All_signals


############################ GENETIC ALGORITHM PARAMETER SETUP ####################################
num_generations = 5
popsize = 4        # Population size
mutindprob = 0.2    # Probability that an individual undergoes mutation
coprob = 0.5        # Crossover probability
maxgene = 0.6*C     # Maximum coupling value of a connection
mingene = 0         # Minimum coupling value of a connection
par_processes = 4  # How many cores will be used in order to parallelize the GA.
L = 15		    # After how many non-improving generations exctinction occurs
# Initialization of the necessary GA functions:
# this has to go before the if name = main and before running the algorithm.
toolbox, creator = galgs.initiate_DEAP(fitness_function_cross_V2, params, generange = (mingene, maxgene), indsize = indivsize, v = 2)
    
if __name__ == '__main__':
    # Folder management
    results_dir = check_create_results_folder()
    newfolder = test_folder(results_dir)
    fig_idx = 1
    with Pool(processes=par_processes) as piscina:
        # Running GA
        maxfits, avgfits, bestsols, extinction_generations = galgs.main_DEAP_extinction(num_generations, popsize, mutindprob, coprob, indivsize, toolbox, creator, 1, L, piscina, v = 2)
        maxfits_avg = np.mean(maxfits, axis = 1)  # Mean of the different fitnesses
        best_indivs_gen = np.argmax(maxfits_avg)  # Generation of the optimal individual
        solution = bestsols[best_indivs_gen]  # Optimal individual

        # Plot the maximum fitnesses and average fitnesses of each generation
        fig_genfit = plot_genfit(num_generations, maxfits, avgfits, best_indivs_gen, extinction_generations, v = 2)
        fig_genfit.savefig(newfolder + "/fitness.jpg")
        
        # Show the coupling matrices corresponding to the best individual of the evolution
        fig_couplings = plotcouplings3x3V2(solution, params['matrix_exc'], params['matrix_inh'], (mingene, maxgene))
        fig_couplings.savefig(newfolder + "/bestweights.jpg")

        # Save some variables for repetition
        np.save(newfolder + '/best_ind', solution)
        np.save(newfolder + '/signals', params['All_signals'])  # In case we wanted to repeat the dynamics computation        

        # Resulting dynamics of the individual and crosscorrelations of last layers dynamics
        solution = np.array(solution)
        params['individual'] = solution
        f = open(newfolder+'/correlation.txt', 'a+')
        for ii in range(3):
            params['signals'] = params['All_signals'][ii]
            pair = params['pairs'][ii]
            synch_pair = (Nnodes - params['tuplenetwork'][-1] + pair[0], Nnodes - params['tuplenetwork'][-1] + pair[1])
            y, t = obtaindynamicsNET(params, params['tspan'], params['tstep'], 3)
            typeplot = 'small'

            saving = newfolder + '/Dynamics' + str(ii) + typeplot + '.png'
            if ii == 0:
                #fig0 = plot3x3_signals(y, t, typeplot, params['tstep'], params['All_signals'][ii])
                fig0 = plotinputsoutputs(y, t, typeplot, params, True, params['signals'])
                fig0.savefig(saving)
            elif ii == 1:
                #fig1 = plot3x3_signals(y, t, typeplot, params['tstep'], params['signals'])
                fig1= plotanynet(y, t, typeplot, params, True, params['signals'])
                fig1.savefig(saving)
            else:
                #fig2 = plot3x3_signals(y, t, typeplot, params['tstep'], params['signals'])
                fig2 = plotanynet(y, t, typeplot, params, True, params['signals'])
                fig2.savefig(saving)

            f.write('Nodes %i and %i should be synchronized:' % synch_pair)  #Saving results in file
            f.write('\nCorrelation 6 with 7: ' + str(maxcrosscorrelation(y[6], y[7])))
            f.write('\nCorrelation 6 with 8: ' + str(maxcrosscorrelation(y[6], y[8])))
            f.write('\nCorrelation 7 with 8: ' + str(maxcrosscorrelation(y[7], y[8]))+ '\n \n')
        f.close()
        f = open(newfolder+'/correlation.txt', 'r')
        print(f.read())  # This way we save the results in a .txt file for later
        
        plt.show()