"""Small script that will be used to recover the dynamics and results after a
fail occured after the GA"""

import numpy as np
import matplotlib.pyplot as plt
from networkJR import obtaindynamicsNET
from galgs import test_solution
from signals import build_dataset
from filefuns import check_create_results_folder, get_num
from plotfuns import plot_inputs, plotcouplings, plot_genfit, plot_bestind_normevol

results_dir = check_create_results_folder()
newfolder = get_num(results_dir)
numnow = int(newfolder[-2:])-1
if numnow < 10:
    newfolder = results_dir + '/0' + str(numnow)
else:
    newfolder = results_dir + '/' + str(numnow)
print(newfolder)
maxfits = np.load(newfolder + '/maxfits.npy')
avgfits = np.load(newfolder + '/avgfits.npy')
ext_gens = np.load(newfolder + '/extgens.npy')
bestsols = np.load(newfolder + '/bestsols.npy')
params = np.load(newfolder + '/params.npy', allow_pickle=True).item()
params['tspan'] = (0, 500)
params['t'] = np.linspace(params['tspan'][0], params['tspan'][1], int((params['tspan'][1] - params['tspan'][0])/params['tstep']))
params['test_dataset'] = build_dataset(int(2*params['n']),
                                       params['tuplenetwork'][0],
                                       params['pairs'], params['t'], offset=params['offset'])

maxfits_avg = np.mean(maxfits, axis=1)  # Mean of the different fitnesses
best_indivs_gen = np.argmax(maxfits_avg)  # Generation of the optimal individual
solution = bestsols[best_indivs_gen]  # Optimal individual
solution = np.array(solution)
# Plot the maximum fitnesses and average fitnesses of each generation
fig_genfit = plot_genfit(params['num_generations'], maxfits, avgfits, best_indivs_gen, ext_gens, v=2)
fig_genfit.savefig(newfolder + "/fitness.jpg")

# Show the coupling matrices corresponding to the best individual of the evolution
fig_couplings = plotcouplings(solution, params['matrix_exc'], params['matrix_inh'],
                              (params['minvalue'], np.amax(solution)), params, True)
fig_couplings.savefig(newfolder + "/bestweights.jpg")

# Plot the evolution of the norm of the best solution
fig_normevol = plot_bestind_normevol(bestsols, params['num_generations'], params)
fig_normevol.savefig(newfolder + "/normevol.jpg")

# Finally print the tests results and plot some of the dynamics
params['individual'] = solution

params['signals'] = params['test_dataset'][0][0]
y, t = obtaindynamicsNET(params, params['tspan'], params['tstep'], v=3)
modidx = int(params['tspan'][-1] - 10) * 1000
plot_inputs(y, params['signals'][:, -modidx:], params, t, newfolder)
test_solution(params, newfolder, whatplot='net')

