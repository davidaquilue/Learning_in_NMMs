"""Small script that will be used to recover the dynamics and results after a
fail occured after the GA"""

import numpy as np
import matplotlib.pyplot as plt
from main import params
from galgs import test_solution
from networkJR import obtaindynamicsNET 
from filefuns import check_create_results_folder, get_num
from plotfuns import plot_inputs, plot_fftoutputs, plotcouplings

results_dir = check_create_results_folder()
newfolder = get_num(results_dir)
numnow = int(newfolder[-2:])-1
if numnow < 10:
    newfolder = results_dir + '/0' + str(numnow)
else:
    newfolder = results_dir + '/' + str(numnow)
print(newfolder)

#solution = np.load(newfolder + '/best_ind.npy')
solution = np.ones(72)
print(solution.shape)
fig_couplings = plotcouplings(solution, params['matrix_exc'], params['matrix_inh'],
                              (0, 1), params, True)
fig_couplings.savefig(newfolder + "/bestweights.jpg")
"""
params['individual'] = solution

params['signals'] = params['test_dataset'][0][0]
y, t = obtaindynamicsNET(params, params['tspan'], params['tstep'], v=3)
modidx = int(params['tspan'][-1]-10)*1000
plot_inputs(y, params['signals'][:, -modidx:], params, t, newfolder)
plot_fftoutputs(y, params, newfolder)
test_solution(params, newfolder, whatplot='net')
"""
