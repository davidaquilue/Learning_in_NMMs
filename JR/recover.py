"""Small script that will be used to recover the dynamics and results after a
fail occured after the GA"""

import numpy as np
import os
import matplotlib.pyplot as plt
from networkJR import obtaindynamicsNET
from galgs import test_solution
from signals import build_dataset, build_p_inputs
from filefuns import check_create_results_folder, get_num
from plotfuns import plot_inputs, plotcouplings, plot_genfit, plot_bestind_normevol, plot_corrs, plot_363

typical_recover = True
all_nodes = True
no_nodes = True
visualize_best = False

# The folder that will be used to run this script will always be the last one in the Results folder.
print('Ignore the following printed line:')

# We get the last folder
results_dir = check_create_results_folder()
newfolder = get_num(results_dir)
numnow = int(newfolder[-2:])-1
if numnow < 10:
    newfolder = results_dir + '/0' + str(numnow)
else:
    newfolder = results_dir + '/' + str(numnow)

# Now we want to determine if the folder contains old (before GA graph improvement) or new results:
new = os.path.isfile(newfolder + '/params.npy')
print('New results: ' + str(new))
if new:
    maxfits = np.load(newfolder + '/maxfits.npy')
    avgfits = np.load(newfolder + '/avgfits.npy')
    ext_gens = np.load(newfolder + '/extgens.npy')
    bestsols = np.load(newfolder + '/bestsols.npy')
    params = np.load(newfolder + '/params.npy', allow_pickle=True).item()
    maxfits_avg = np.mean(maxfits, axis=1)  # Mean of the different fitnesses
    best_indivs_gen = np.argmax(maxfits_avg)  # Generation of the optimal individual
    solution = bestsols[best_indivs_gen]  # Optimal individual
    solution = np.array(solution)

else:
    from main import params
    solution = np.load(newfolder + '/best_ind.npy')
    bestsols = np.load(newfolder + '/best_sols.npy')


# Now we rearrange some of the params for faster simulations
params['tspan'] = (0, 500)
params['t'] = np.linspace(params['tspan'][0], params['tspan'][1], int((params['tspan'][1] - params['tspan'][0])/params['tstep']))
params['individual'] = solution
if typical_recover:
    params['test_dataset'] = build_dataset(int(2*params['n']),
                                           params['tuplenetwork'][0],
                                           params['pairs'], params['t'], offset=params['offset'],
                                           shift=params['shift'])

    if new:
        # Plot the maximum fitnesses and average fitnesses of each generation
        fig_genfit = plot_genfit(params['num_generations'], maxfits, avgfits, best_indivs_gen, ext_gens, v=2)
        fig_genfit.savefig(newfolder + "/fitness.jpg")

    # Show the coupling matrices corresponding to the best individual of the evolution
    fig_couplings = plotcouplings(solution, params['matrix_exc'], params['matrix_inh'],
                                  (params['minvalue'], params['maxvalue']), params, True)
    fig_couplings.savefig(newfolder + "/bestweights.jpg")

    # Plot the evolution of the norm of the best solution
    fig_normevol = plot_bestind_normevol(bestsols, bestsols.shape[0], params)
    fig_normevol.savefig(newfolder + "/normevol.jpg")

    # Finally print the tests results and plot some of the dynamics
    params['signals'] = params['test_dataset'][0][0]
    y, t = obtaindynamicsNET(params, params['tspan'], params['tstep'], v=3)
    modidx = int(params['tspan'][-1] - 10) * 1000
    plot_inputs(y, params['signals'][:, -modidx:], params, t, newfolder)
    test_solution(params, newfolder, whatplot='net')

# I want to test some other things aswell, on the results obtained. When analyzing these things comment
# everything on top

# Primer de tot, tots correlacionats:

# M'he de fer les gràfiques aqui millor, i crear els inputs també perquè si no està una mica complicat.
if all_nodes:
    # No hace falta un dataset entero, solo un conjunto de señales y un
    params['pairs'] = ((0, 1, 2), )
    params['signals'] = build_p_inputs(3, params['t'], params['offset'], (0, 1, 2))
    y, t = obtaindynamicsNET(params, params['tspan'], params['tstep'], v=3)
    plot_corrs(y, 0, params, newfolder)
    fig_all = plot_363(y, t, 'small', params, True, params['signals'])
    fig_all.savefig(newfolder + '/Dynamics_all.png')

if no_nodes:
    params['pairs'] = ((0,),)
    params['signals'] = build_p_inputs(3, params['t'], params['offset'], (0,))
    y, t = obtaindynamicsNET(params, params['tspan'], params['tstep'], v=3)
    plot_corrs(y, 0, params, newfolder)
    fig_none = plot_363(y, t, 'small', params, True, params['signals'])
    fig_none.savefig(newfolder + '/Dynamics_none.png')

if visualize_best:
    # A lo mejor se podria hacer una animación más currada, o un análisis más matemático o así.
    ii = 1
    for gg, bestsol in enumerate(bestsols):
        if maxfits_avg[gg] > 2.75:
            fig_couplings = plotcouplings(bestsol, params['matrix_exc'], params['matrix_inh'],
                                          (params['minvalue'], params['maxvalue']), params, True)
            ii += 1
            if ii >= 20:
                plt.show()
                ii = 1


# See how it goes with everything the same but only with two different regimes (node and alpha) nyeh complicat
