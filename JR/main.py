"""Main Script. It runs the Genetic Algorithm and Saves the results. Modifications should be
performed in this module's code."""

from multiprocessing import Pool
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from matfuns import networkmatrix, maxcrosscorrelation, creating_signals, networkmatrix_exc_inh, fastcrosscorrelation  # This last function will be used later on
from plotfuns import plotcouplings, plot3x3_signals, plot_genfit, plotanynet, plotinputsoutputs, plot_bestind_normevol, plot_inputs, plot_fftoutputs
from networkJR import obtaindynamicsNET
from fitfuns import fit_func_cross_V3
from filefuns import check_create_results_folder, test_folder
from signals import build_dataset
import galgs
matplotlib.use('Agg')

# JR MODEL PARAMETERS
params = dict(A=3.25, B=22.0, v0=6.0)
params['a'], params['b'], params['e0'] = 100.0, 50.0, 2.5
params['pbar'], params['delta'], params['f'] = 155.0, 65.0, 8.5

params['C'] = C = 133.5
params['C1'], params['C2'], params['C3'], params['C4'] = C, 0.8*C, 0.25*C, 0.25*C  # Dimensionless

params['r'] = 0.56  # mV^(-1)

params['delta'] = 72.09
params['f'] = 8.6

# NETWORK ARCHITECTURE PARAMETERS
params['tuplenetwork'] = (3, 6, 3)
params['recurrent'] = False
params['forcednodes'] = (0, 1, 2)

Nnodes, matrix_exc, matrix_inh = networkmatrix_exc_inh(params['tuplenetwork'], params['recurrent'], v=0)
# (see the function in the matfuns script for further information on the tests)
indivsize = np.count_nonzero(matrix_exc) + np.count_nonzero(matrix_inh)

params['Nnodes'] = Nnodes
params['matrix_exc'] = matrix_exc
params['matrix_inh'] = matrix_inh
params['tstep'] = 0.001
params['tspan'] = (0, 2000)

# INPUT SIGNALS: TRAINING AND TESTING SETS
t = np.linspace(params['tspan'][0], params['tspan'][1], int((params['tspan'][1] - params['tspan'][0])/params['tstep']))

params['pairs'] = ((1, 2), (0, 2), (0, 1))  # Pairs of correlated first layer nodes
idx = params['Nnodes'] - params['tuplenetwork'][-1]

params['output_pairs'] = tuple([(idx + pair[0], idx + pair[1]) for pair in params['pairs']])
# Correlated nodes at the output, basically: ((10, 11), (9, 11), (9,10)) with more freedom

params['unsync'] = (0, 1, 2)  # Unsynchronized nodes, have to align with the correlated pairs.
params['n'] = 1  # Amount of elements in the training set, at least 10
params['train_dataset'] = build_dataset(params['n'], params['tuplenetwork'][0],
                                        params['pairs'], t, offset=10)
params['test_dataset'] = build_dataset(int(2*params['n']),
                                       params['tuplenetwork'][0],
                                       params['pairs'], t, offset=10)

######################### GENETIC ALGORITHM PARAMETER SETUP ###################
num_generations = 200
popsize = 200       # Population size
mutindprob = 0.1    # Probability that an individual undergoes mutation
coprob = 0.8        # Crossover probability
maxgene = 0.2*C     # Maximum coupling value of a connection
mingene = 0         # Minimum coupling value of a connection
par_processes = 32  # How many cores will be used in order to parallelize the GA.
L = 35              # After how many non-improving generations exctinction occurs

# Maybe coprob=0.5 and mutindprob=0.2 are not the best, imma try for the next
# test the values coprob=0.8 and mutindprob=0.1
params['maxvalue'] = maxgene
# Initialization of the necessary GA functions:
# this has to go before the if name = main and before running the algorithm.
toolbox, creator = galgs.initiate_DEAP(fit_func_cross_V3, params,
                                       generange=(mingene, maxgene),
                                       indsize=indivsize, v=2)

if __name__ == '__main__':
    # Folder management
    results_dir = check_create_results_folder()
    newfolder = test_folder(results_dir)
    fig_idx = 1
    cheatlist = maxgene*np.random.rand(indivsize)
    # If we want to see how will the first layer nodes behave we have to uncomment
    # this piece of code:
    with Pool(processes=par_processes) as piscina:
        # Running GA
        maxfits, avgfits, bestsols, ext_gens = galgs.main_DEAP_extinction(num_generations,
                                                                          popsize, mutindprob,
                                                                          coprob, indivsize, toolbox,
                                                                          creator, 1, L, cheatlist, piscina, v=2)
        maxfits_avg = np.mean(maxfits, axis=1)  # Mean of the different fitnesses
        best_indivs_gen = np.argmax(maxfits_avg)  # Generation of the optimal individual
        solution = bestsols[best_indivs_gen]  # Optimal individual
        solution = np.array(solution)
        # Plot the maximum fitnesses and average fitnesses of each generation
        fig_genfit = plot_genfit(num_generations, maxfits, avgfits, best_indivs_gen, ext_gens, v=2)
        fig_genfit.savefig(newfolder + "/fitness.jpg")
        
        # Show the coupling matrices corresponding to the best individual of the evolution
        fig_couplings = plotcouplings(solution, params['matrix_exc'], params['matrix_inh'],
                                      (mingene, np.amax(solution)), params, True)
        fig_couplings.savefig(newfolder + "/bestweights.jpg")

        # Plot the evolution of the norm of the best solution
        fig_normevol = plot_bestind_normevol(bestsols, num_generations, params)
        fig_normevol.savefig(newfolder + "/normevol.jpg")

        np.save(newfolder + '/best_ind', solution)  # Save the best individual
        np.save(newfolder + '/best_sols', bestsols)
        plt.show()

        # Finally print the tests results and plot some of the dynamics
        params['individual'] = solution
        
        params['signals'] = params['test_dataset'][0][0]
        y, t = obtaindynamicsNET(params, params['tspan'], params['tstep'], v=3)
        modidx = int(params['tspan'][-1]-10)*1000
        plot_inputs(y, params['signals'][:, -modidx:], params, t, newfolder)
        plot_fftoutputs(y, params, newfolder)
        galgs.test_solution(params, newfolder, whatplot='net')

