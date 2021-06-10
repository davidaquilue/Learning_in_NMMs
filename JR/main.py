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
#matplotlib.use('Agg')

# JR MODEL PARAMETERS
params = dict(A=3.25, B=22.0, v0=6.0)
params['a'], params['b'], params['e0'] = 100.0, 50.0, 2.5
params['pbar'] = 155.0

params['C'] = C = 133.5
params['C1'], params['C2'], params['C3'], params['C4'] = C, 0.8*C, 0.25*C, 0.25*C  # Dimensionless

params['r'] = 0.56  # mV^(-1)

params['delta'] = 0
params['f'] = 0

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
params['t'] = np.linspace(params['tspan'][0], params['tspan'][1], int((params['tspan'][1] - params['tspan'][0])/params['tstep']))

params['pairs'] = ((1, 2), (0, 2), (0, 1))  # Pairs of correlated first layer nodes
idx = params['Nnodes'] - params['tuplenetwork'][-1]

params['output_pairs'] = tuple([(idx + pair[0], idx + pair[1]) for pair in params['pairs']])
# Correlated nodes at the output, basically: ((10, 11), (9, 11), (9,10)) with more freedom

params['unsync'] = (0, 1, 2)    # Unsynchronized nodes, have to align with the correlated pairs.
params['n'] = 1                 # Amount of elements in the training set, depending on the tspan more or less
params['offset'] = 10           # Baseline of the input vector
params['train_dataset'] = build_dataset(params['n'], params['tuplenetwork'][0],
                                        params['pairs'], params['t'], offset=params['offset'])

# Test test dataset will be set up in the plotting results file

######################### GENETIC ALGORITHM PARAMETER SETUP ###################
params['num_generations'] = num_generations = 300
params['popsize'] = popsize = 260           # Population size
params['mutindprob'] = mutindprob = 0.25    # Probability that an individual undergoes mutation
params['coprob'] = coprob = 0.7             # Crossover probability
params['maxvalue'] = maxgene = 0.1*C        # Maximum coupling value of a connection
params['minvalue'] = mingene = 0            # Minimum coupling value of a connection
par_processes = 30                          # How many cores will be used in order to parallelize the GA.
params['L'] = L = 40                        # After how many non-improving generations exctinction occurs


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

    # Setting up a first individual that we know it works to improve convergence.
    # cheatlist = galgs.get_OP(params)
    cheatlist = maxgene*np.random.rand(indivsize)

    # Run Genetic Algorithm with parallel fitness evaluations of individuals.
    with Pool(processes=par_processes) as piscina:
        # Running GA
        maxfits, avgfits, bestsols, extgens = galgs.main_DEAP_extinction(num_generations,
                                                                         popsize, mutindprob,
                                                                         coprob, indivsize, toolbox,
                                                                         creator, 1, L, cheatlist, piscina, v=2)
        # Save the needed variables to later plot
        np.save(newfolder + '/maxfits.npy', maxfits)
        np.save(newfolder + '/avgfits.npy', avgfits)
        np.save(newfolder + '/extgens.npy', np.array(extgens))
        np.save(newfolder + '/bestsols.npy', bestsols)
        del params['train_dataset']  # We are not interested in saving the training set.
        np.save(newfolder + '/params.npy', params)
