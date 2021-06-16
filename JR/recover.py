"""Small script that will be used to recover the dynamics and results after a
fail occured after the GA"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from networkJR import obtaindynamicsNET, individual_to_weights
from galgs import test_solution
from signals import build_dataset, build_p_inputs
from filefuns import check_create_results_folder, get_num
from plotfuns import plot_inputs, plotcouplings, plot_genfit, plot_bestind_normevol, plot_corrs, plot_363
import os, sys


# The get_num function outputs a print that I don't want to show. With this class and the following line we block it.
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Visualization of parameters, very poorly written...
def createplot(sol, maxval):
    matrix_exc = params['matrix_exc']
    matrix_inh = params['matrix_inh']
    bandw = True
    minmaxvals = (params['minvalue'], params['maxvalue'])
    if maxval > minmaxvals[1]:
        minmaxvals = (minmaxvals[0], maxval)
    weights_exc, weights_inh = individual_to_weights(sol, matrix_exc, matrix_inh)
    weights_exc = np.ma.masked_where(weights_exc == 0, weights_exc)
    weights_inh = np.ma.masked_where(weights_inh == 0, weights_inh)
    nnodes = matrix_exc.shape[0]
    ticks = np.linspace(0, nnodes - 1, nnodes)
    if bandw:
        colormap = cm.get_cmap('Greys')
    else:
        colormap = cm.get_cmap('viridis')

    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    im1 = axes[0].imshow(weights_exc, vmin=minmaxvals[0], vmax=minmaxvals[1], cmap=colormap)
    axes[0].set(title='Connectivity matrix $E$', xlabel='Pre-Synaptic Node', ylabel='Post-Synaptic Node')
    axes[0].set(title='Connectivity matrix $E$', xlabel='Pre-Synaptic Node', ylabel='Post-Synaptic Node',
             xticks=ticks, yticks=ticks)
    axes[0].set_xticks(np.arange(-.5, params['Nnodes']-1, 1), minor=True)
    axes[0].set_yticks(np.arange(-.5, params['Nnodes']-1, 1), minor=True)
    axes[0].grid(which='minor', color='k')

    im2 = axes[1].imshow(weights_inh, vmin=minmaxvals[0], vmax=minmaxvals[1], cmap=colormap)
    axes[1].set(title='Connectivity matrix $I$', xlabel='Pre-Synaptic Node', ylabel='Post-Synaptic Node',
                xticks=ticks, yticks=ticks)
    axes[1].set_xticks(np.arange(-.5, params['Nnodes'] - 1, 1), minor=True)
    axes[1].set_yticks(np.arange(-.5, params['Nnodes'] - 1, 1), minor=True)
    axes[1].grid(which='minor', color='k')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    title = fig.suptitle('Generation %i' % 0 + '. Avg Fitness: %d' % maxfits_avg[0], fontsize=20)
    return fig, [im1, im2, title]


def update_weights(g):
    solution = bestsols[g]
    matrix_exc = params['matrix_exc']
    matrix_inh = params['matrix_inh']
    minmaxvals = (params['minvalue'], params['maxvalue'])
    weights_exc, weights_inh = individual_to_weights(solution, matrix_exc, matrix_inh)
    imgs[0].set_data(weights_exc)
    str = 'Generation %i' % g + '. Fitness: %f' % maxfits_avg[g]
    imgs[1].set_data(weights_inh)
    imgs[2].set_text(str)
    return imgs


# When executing the script:
if __name__ == '__main__':
    with HiddenPrints():
        results_dir = check_create_results_folder()
        newfolder = get_num(results_dir)
    # We get the last folder
    numnow = int(newfolder[-2:])-1
    if numnow < 10:
        newfolder = results_dir + '/0' + str(numnow)
    else:
        newfolder = results_dir + '/' + str(numnow)

    print('Recovering test %i' % numnow)
    print("\nEnter 1 if you want to perform these tests. 0 if you don't want to.")
    typical_recover = int(input('Typical recover: '))
    all_nodes = int(input('All nodes correlated: '))
    no_nodes = int(input('No nodes correlated: '))
    visualize_best = int(input('Evolution video of bestsols: '))
    printparams = int(input('Print params: '))

    # Now we want to determine if the folder contains old (before GA graph improvement) or new results:
    new = os.path.isfile(newfolder + '/params.npy')
    print('New results: ' + str(new))
    if new:
        maxfits = np.load(newfolder + '/maxfits.npy')
        avgfits = np.load(newfolder + '/avgfits.npy')
        ext_gens = np.load(newfolder + '/extgens.npy')
        bestsols = np.load(newfolder + '/bestsols.npy')
        params = np.load(newfolder + '/params.npy', allow_pickle=True).item()
        try:
            check = params['bestmin']
        except:
            params['bestmin'] = False

        try:
            check = params['shift']
        except:
            params['shift'] = 0
        maxfits_avg = np.mean(maxfits, axis=1)  # Average of 3 fitvalues of best individual/generation
        # Choosing the optimal individual resulting from the realization of the algorithm
        if params['bestmin']:
            representation = np.amin(maxfits, axis = 1)
            best_indivs_gen = np.argmax(representation)
        else:
            best_indivs_gen = np.argmax(maxfits_avg)  # Generation of the optimal individual
        solution = bestsols[best_indivs_gen]  # Optimal individual
        solution = np.array(solution)
        if printparams:
            print(params)

    else:
        from main import params
        solution = np.load(newfolder + '/best_ind.npy')
        bestsols = np.load(newfolder + '/best_sols.npy')
        if printparams:
            print('Cannot print params in the old version.')

    # Now we rearrange some of the params for faster simulations
    params['tspan'] = (0, 1000)
    params['t'] = np.linspace(params['tspan'][0], params['tspan'][1], int((params['tspan'][1] - params['tspan'][0])/params['tstep']))
    params['individual'] = solution
    if typical_recover:
        if new:
            check = os.path.isfile(newfolder + '/inputs_0.png')
            if check:
                print('Typical recover already performed on this results. If want to repeat please delete inputs_0.png\n')
            else:
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
        else:
            print('Typical recovery cannot be performed.')

    # I want to test some other things aswell, on the results obtained.

    if all_nodes:
        check = os.path.isfile(newfolder + '/corrsall.png')
        if check:
            print('All correlated nodes recover already performed on this results. If want to repeat please delete corrsall.png\n')
        else:
            # No hace falta un dataset entero, solo un conjunto de señales y un
            params['pairs'] = ((0, 1, 2), )
            params['signals'] = build_p_inputs(3, params['t'], params['offset'], (0, 1, 2))
            y, t = obtaindynamicsNET(params, params['tspan'], params['tstep'], v=3)
            plot_corrs(y, 0, params, newfolder)
            fig_all = plot_363(y, t, 'small', params, True, params['signals'])
            fig_all.savefig(newfolder + '/Dynamics_all.png')

    if no_nodes:
        check = os.path.isfile(newfolder + '/corrsnone.png')
        if check:
            print('No correlated nodes recover already performed on this results. If want to repeat please delete corrsnone.png\n')
        else:
            params['pairs'] = ((0,),)
            params['signals'] = build_p_inputs(3, params['t'], params['offset'], (0,))
            y, t = obtaindynamicsNET(params, params['tspan'], params['tstep'], v=3)
            plot_corrs(y, 0, params, newfolder)
            fig_none = plot_363(y, t, 'small', params, True, params['signals'])
            fig_none.savefig(newfolder + '/Dynamics_none.png')

    maxval = np.amax(bestsols)
    if visualize_best:
        check = os.path.isfile(newfolder + '/bestsolsev.mp4')
        if check:
            print('Evolution video recover already performed on this results. If want to repeat please delete bestsolsev.mp4\n')
        else:
            fig, imgs = createplot(bestsols[0], maxval)
            frames = np.arange(100, bestsols.shape[0], 1)
            ani = animation.FuncAnimation(fig, update_weights, frames=frames, interval=75, repeat_delay=5000)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(newfolder + '/bestsolsev.mp4', writer=writer)
            #plt.show()