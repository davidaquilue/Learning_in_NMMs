"""Plotting functions.

Functions included: plot3x3, plot3x3_signals, plotanynet, plotcouplings3x3, plotcouplings3x3V2, plotgenfit,
plot_bestind_normevol"""

import matplotlib
#matplotlib.use('Agg')
from matplotlib import cm
from matfuns import psd, findlayer, fastcrosscorrelation as ccross
import numpy as np
import matplotlib.pyplot as plt
from networkJR import individual_to_weights

# General fontsizes
labelfontsize = 15
labelticksize = 15
titlefontsize = 20
suptitlefontsize = 26


def plot3x3(y, t, span, tstep):
    """
    Plots the dynamics of a 3x3x3 network of cortical columns. First layer red, second blue and third green. N_nodes = 9
    Inputs:
    y:      Vector shaped (N_nodes, nsteps) where y[i,nsteps] represents the PSP of the pyramidal population of column i
    t:      Time vector
                fig0 = plot3x3_signals(y,t, typeplot, params['tstep'], params['All_signals'][ii])
                saving = newfolder + '/Dynamics' + str(ii) + typeplot + '.png'
                fig.savefig(saving)
                fig0 = plot3x3_signals(y,t, typeplot, params['tstep'], params['All_signals'][ii])
                saving = newfolder + '/Dynamics' + str(ii) + typeplot + '.png'
                fig.savefig(saving)
    span:   If 'large', 7 seconds of dynamics are presented, if 'small', 3 seconds. Enough for alpha rhythms
    tstep:  Timestep in the simulations, used to obtain the maxima of PSD of each node.
    """
    fig, axes = plt.subplots(nrows=3, ncols=3)
    fig.subplots_adjust(hspace=0.5)
    fig.set_figheight(2*3)
    fig.set_figwidth(12)
    # Maybe this should be always fixed so we always work in the same span.
    yspan = (np.min(y)-1, np.max(y)+1)

    if span == 'large':
        xspan = (t[-7001], t[-1])
    elif span == 'small':
        xspan = (t[-3000], t[-1])

    p1 = 0
    p2 = 0
    p3 = 0
    for ii in range(9):
        yy = y[ii]
        f, PSD = psd(yy, tstep)
        maxf = np.abs(f[np.argmax(PSD)])
        lab = 'Node: ' + str(ii)
        if ii < 3:
            ax = axes[p1, 0]
            p1 = p1+1
            ax.plot(t, yy, 'r', label=lab)
            ax.set(xlim=xspan, ylim=yspan)
            ax.set_xlabel(r'time (s)', fontsize=labelfontsize)
            ax.set_ylabel(r'$y_1-y_2$', fontsize=labelfontsize)
            ax.set_title(r'max of PSD f = %g Hz' % maxf)
            ax.tick_params(labelsize=labelticksize)

        elif ii >= 3 and ii < 6:
            ax = axes[p2, 1]
            p2 = p2+1
            ax.plot(t, yy, 'b', label=lab)
            ax.set(xlim=xspan, ylim=yspan)
            ax.set_xlabel(r'time (s)', fontsize=labelfontsize)
            ax.set_ylabel(r'$y_1-y_2$', fontsize=labelfontsize)
            ax.set_title(r'max of PSD f = %g Hz' % maxf)
            ax.tick_params(labelsize=labelticksize)
            
        else:
            ax = axes[p3, 2]
            p3 = p3+1
            ax.plot(t, yy, 'g', label=lab)
            ax.set(xlim=xspan, ylim=yspan)
            ax.set_xlabel(r'time (s)', fontsize=labelfontsize)
            ax.set_ylabel(r'$y_1-y_2$', fontsize=labelfontsize)
            ax.set_title(r'max of PSD f = %g Hz' % maxf)
            ax.tick_params(labelsize=labelticksize)

    plt.tight_layout()
    # So that we can save it if we want:
    return fig


def plot3x3_signals(y, t, span, tstep, signals):
    """
    Plots the dynamics of a 3x3x3 network of cortical columns. First layer red, second blue and third green. N_nodes = 9
    Inputs:
    y:      Vector shaped (N_nodes, nsteps) where y[i,nsteps] represents the PSP of the pyramidal population of column i
    t:      Time vector
    span:   If 'large', 7 seconds of dynamics are presented, if 'small', 3 seconds. Enough for alpha rhythms
    tstep:  Timestep in the simulations, used to obtain the maxima of PSD of each node.
    """
    fig, axes = plt.subplots(nrows=3, ncols=4)
    fig.subplots_adjust(hspace=0.5)
    fig.set_figheight(2*3)
    fig.set_figwidth(12)
    # Maybe this should be always fixed so we always work in the same span.
    yspan = (np.min(y)-1, np.max(y)+1)

    if span == 'large':
        xspan = (t[-7001], t[-1])
    elif span == 'small':
        xspan = (t[-3000], t[-1])
    
    p1 = 0
    for ii in range(3):
        ax = axes[ii, 0]
        p1 = p1+1
        ax.plot(t[-10000:], signals[ii, -10000:], 'k')
        ax.set(xlim=xspan, ylim=(np.amin(signals[ii]), 1.3*np.amax(signals[ii])))
        ax.set_xlabel(r'time (s)', fontsize=labelfontsize)
        ax.set_ylabel(r'$Hz$', fontsize=labelfontsize)
        ax.tick_params(labelsize=labelticksize)

    p1 = 0
    p2 = 0
    p3 = 0
    for ii in range(9):
        yy = y[ii]
        f, PSD = psd(yy, tstep)
        maxf = np.abs(f[np.argmax(PSD)])
        lab = 'Node: ' + str(ii)
        if ii < 3:
            ax = axes[p1, 1]
            p1 = p1+1
            ax.plot(t, yy, 'r', label=lab)
            ax.set(xlim=xspan, ylim=yspan)
            ax.set_xlabel(r'time (s)', fontsize=labelfontsize)
            ax.set_ylabel(r'$y_1-y_2$', fontsize=labelfontsize)
            ax.set_title(r'max of PSD f = %g Hz' % maxf)
            ax.tick_params(labelsize=labelticksize)

        elif ii >= 3 and ii < 6:
            ax = axes[p2, 2]
            p2 = p2+1
            ax.plot(t, yy, 'b', label=lab)
            ax.set(xlim=xspan, ylim=yspan)
            ax.set_xlabel(r'time (s)', fontsize=labelfontsize)
            ax.set_ylabel(r'$y_1-y_2$', fontsize=labelfontsize)
            ax.set_title(r'max of PSD f = %g Hz' % maxf)
            ax.tick_params(labelsize=labelticksize)
            
        else:
            ax = axes[p3,3]; p3 = p3+1
            ax.plot(t, yy,'g', label = lab)
            ax.set(xlim = xspan, ylim = yspan)
            ax.set_xlabel(r'time (s)', fontsize = labelfontsize)
            ax.set_ylabel(r'$y_1-y_2$', fontsize = labelfontsize)
            ax.set_title(r'max of PSD f = %g Hz' %maxf)
            ax.tick_params(labelsize = labelticksize)

    plt.tight_layout()
    # So that we can save it if we want:
    return fig


def plot_363(y, t, span, params, bool_sig, signals):
    if span == 'large':
        xspan = (t[-50001], t[-1])
    elif span == 'small':
        xspan = (t[-20000], t[-1])
    yspan = (np.min(y)-1, np.max(y)+1)

    fig = plt.figure(figsize=(26, 15))
    # Set up the axis
    # Señales
    axp0 = fig.add_subplot(5, 4, 5)
    axp1 = fig.add_subplot(5, 4, 9)
    axp2 = fig.add_subplot(5, 4, 13)
    # Primera capa
    ax0 = fig.add_subplot(5, 4, 6)
    ax1 = fig.add_subplot(5, 4, 10)
    ax2 = fig.add_subplot(5, 4, 14)
    # Outputs
    ax9 = fig.add_subplot(5, 4, 8)
    ax10 = fig.add_subplot(5, 4, 12)
    ax11 = fig.add_subplot(5, 4, 16)
    # Capa del medio
    ax3 = fig.add_subplot(6, 4, 3)
    ax4 = fig.add_subplot(6, 4, 7)
    ax5 = fig.add_subplot(6, 4, 11)
    ax6 = fig.add_subplot(6, 4, 15)
    ax7 = fig.add_subplot(6, 4, 19)
    ax8 = fig.add_subplot(6, 4, 23)
    axesp = [axp0, axp1, axp2]
    axesy = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11]

    for ii in range(signals.shape[0]):
        ax = axesp[ii]
        ax.plot(t[-(t.size-10000):], signals[ii, -(t.size-10000):], 'k')
        ax.set(xlim=xspan, ylim=(np.amin(signals[ii]), 1.3 * np.amax(signals[ii])), ylabel='$Hz$')
        ax.tick_params(labelsize=labelticksize)
        ax.grid(True)
        if ii != signals.shape[0]-1:
            ax.xaxis.set_ticklabels([])
        else:
            ax.set(xlabel='time(s)')
        if ii == 0:
            ax.set(title='Input signal $p(t)$')

    for ii in range(params['Nnodes']):
        layerii = findlayer(ii, params['tuplenetwork'])
        ax = axesy[ii]
        ax.plot(t, y[ii], color=cm.tab10(layerii))
        ax.set(xlim=xspan, ylim=yspan, ylabel='$y_1 - y_2$')
        ax.tick_params(labelsize=labelticksize)
        ax.grid(True)
        if ii == 2 or ii == 8 or ii == 11:
            ax.set(xlabel='time(s)')
        else:
            ax.xaxis.set_ticklabels([])
        if ii == 0 or ii == 3 or ii == 9:
            ax.set_title('Layer %i' % layerii)

    return fig


def plotanynet(y, t, span, params, bool_sig, signals):
    """
    Plots the dynamics of a ZxZxZx....xZ network of cortical columns. It is necessary that all layers have the same number of cortical columns.
    y:              Vector shaped (N_nodes, nsteps) where y[i,nsteps] represents the PSP of the pyramidal population of column i
    t:              Time vector
    span:           If 'large', 7 seconds of dynamics are presented, if 'small', 3 seconds. Enough for alpha rhythms
    params:         Dictionary of parameters
    bool_sig:       Boolean indicating if we want to add the input signals to the plot
    signals:        Z x timesteps array containing the different input signals.
    """
    tuplenetwork = params['tuplenetwork']

    # First let's check if all the layers are the same
    equal = True
    for layer, num_nodes in enumerate(tuplenetwork[1:]):
        if num_nodes != tuplenetwork[layer-1]:
            equal = False
    if equal:
        maxnodesperlayer = tuplenetwork[0]
    else:
        maxnodesperlayer = max(tuplenetwork)
    yspan = (np.min(y)-1, np.max(y)+1)
    layers = len(tuplenetwork)
    Nnodes = params['Nnodes']
    fig, axes = plt.subplots(nrows=maxnodesperlayer, ncols=layers+1)#,figsize=(6*(layers+1), 4*maxnodesperlayer))
    
    fig.subplots_adjust(hspace=0.5)
    fig.set_figheight(6*layers)
    fig.set_figwidth(4*maxnodesperlayer)

    if span == 'large':
        xspan = (t[-15001], t[-1])
    elif span == 'small':
        xspan = (t[-7000], t[-1])

    cc = 0
    if bool_sig:
        for ii in range(signals.shape[0]):
            ax = axes[ii, cc]
            ax.plot(t[-30000:], signals[ii, -30000:], 'k')
            ax.set(xlim=xspan, ylim=(np.amin(signals[ii]), 1.3*np.amax(signals[ii])))
            ax.set_xlabel(r'time (s)', fontsize=labelfontsize)
            ax.set_ylabel(r'$Hz$', fontsize=labelfontsize)
            ax.tick_params(labelsize=labelticksize)
        cc += 1

    idxlayers = np.zeros(layers)
    for ii in range(Nnodes):
        f, PSD = psd(y[ii], params['tstep'])
        maxf = np.abs(f[np.argmax(PSD)])
        layerii = findlayer(ii, tuplenetwork)
        ax = axes[int(idxlayers[layerii]), layerii+1]
        idxlayers[layerii] = int(idxlayers[layerii]) + 1
        yy = y[ii]
        ax.plot(t, yy, color=cm.tab10(layerii))
        ax.set(xlim=xspan, ylim=yspan)
        ax.set_xlabel(r'time (s)', fontsize=labelfontsize)
        ax.set_ylabel(r'$y_1-y_2$', fontsize=labelfontsize)
        ax.set_title(r'max of PSD f = %g Hz' %maxf)
        ax.tick_params(labelsize=labelticksize)
    
    plt.tight_layout()
    return fig


def plotinputsoutputs(y, t, span, params, bool_sig, signals):
    """A function that plots input signals and first and last layers' dynamics.
    First and last layer should have the same nodes. This function will allow to obtain results from more bizarre
    network architectures."""
    nodesfirstlast = params['tuplenetwork'][0]
    yspan = (np.min(y)-1,np.max(y)+1)
    if span == 'large':
        xspan = (t[-7001],t[-1])
    elif span == 'small':
        xspan = (t[-3000],t[-1])
    
    fig, axes = plt.subplots(nrows=nodesfirstlast, ncols=3, figsize=(21,10))
    # Plot the signals first
    cc = 0
    for ii in range(signals.shape[0]):
        ax = axes[ii, cc]
        ax.plot(t[-10000:], signals[ii, -10000:], 'k')
        ax.set(xlim=xspan, ylim=(np.amin(signals[ii]), 1.3*np.amax(signals[ii])))
        ax.set_xlabel(r'time (s)', fontsize=labelfontsize)
        ax.set_ylabel(r'$Hz$', fontsize=labelfontsize)
        ax.tick_params(labelsize=labelticksize)
    cc += 1

    # Then plot the first layer dynamics
    for ii in range(nodesfirstlast):
        yy = y[ii]
        ax = axes[ii, cc]
        ax.plot(t[-10000:], yy[-10000:], 'r')
        ax.set(xlim = xspan, ylim = yspan)
        ax.set_xlabel(r'time (s)', fontsize = labelfontsize)
        ax.set_ylabel(r'$y_1 - y_2$', fontsize = labelfontsize)
        ax.tick_params(labelsize = labelticksize)
    cc += 1

    # Finally the last layer's
    for ii in range(nodesfirstlast):
        yy = y[params['Nnodes'] - nodesfirstlast + ii]
        ax = axes[ii, cc]
        ax.plot(t[-10000:], yy[-10000:], 'g')
        ax.set(xlim = xspan, ylim = yspan)
        ax.set_xlabel(r'time (s)', fontsize = labelfontsize)
        ax.set_ylabel(r'$y_1 - y_2$', fontsize = labelfontsize)
        ax.tick_params(labelsize = labelticksize)
    
    plt.tight_layout()
    return fig


def plotcouplings3x3(solution):
    """Returns an imshow of the value of the couplings of a certain set of weights.
    Valid only for the case of one excitatory and inhibitory coefficient per column.
    solution = [alpha0, alpha1, alpha2,..., alpha8, beta8, beta7, ..., beta0]"""
    fig, axes = plt.subplots(1, 2)
    minim = np.amin(solution); maxim = np.amax(solution)
    alphas = np.reshape(solution[0:9], (3, 3)).T
    betas = np.reshape(np.flip(solution[9:]), (3, 3)).T
    ax = axes[0]
    ax.imshow(alphas, vmin = minim, vmax = maxim)
    ax.set_title('Excitatory Coupling Coefficients')
    ax.set_xlabel('Layers')
    ax = axes[1]
    im = ax.imshow(betas, vmin = minim, vmax = maxim)
    ax.set_title('Inhibitory Coupling Coefficients')
    ax.set_xlabel('Layers')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig


def plotcouplings3x3V2(solution, matrix_exc, matrix_inh, maxminvals):
    """Returns an imshow of the excitatory and inhibitory weight matrix. solution is the vector of the individiual
     with best fitness."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    weights_exc, weights_inh = individual_to_weights(solution, matrix_exc, matrix_inh)
    weights_exc = np.ma.masked_where(weights_exc == 0, weights_exc)
    weights_inh = np.ma.masked_where(weights_inh == 0, weights_inh)
    nnodes = matrix_exc.shape[0]
    ticks = np.linspace(0, nnodes-1, nnodes)

    ax = axes[0]
    ax.imshow(weights_exc, vmin=maxminvals[0], vmax=maxminvals[1])
    ax.set(title='Excitatory Coupling Coefficients', xlabel='Pre-Synaptic Node', ylabel='Post-Synaptic Node',
           xticks=ticks, yticks=ticks)
    ax.set_xticks(np.arange(-.5, nnodes - 1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nnodes - 1, 1), minor=True)
    ax.grid(which='minor', color='k')

    ax = axes[1]
    im = ax.imshow(weights_inh, vmin=maxminvals[0], vmax=maxminvals[1])
    ax.set(title='Excitatory Coupling Coefficients', xlabel='Pre-Synaptic Node', ylabel='Post-Synaptic Node',
           xticks=ticks, yticks=ticks)
    fig.subplots_adjust(right=0.8)
    ax.set_xticks(np.arange(-.5, nnodes - 1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nnodes - 1, 1), minor=True)
    ax.grid(which='minor', color='k')

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig


def plot_genfit(num_generations, maxfits, avgfits, best_indivs_gen, extinction_generations = [], v = 1):
    '''Returns the plot of the highest and average fitnesses per generation. Additionally 
    indicates extinctions and the best individual's fitnesses'''
    fig, ax = plt.subplots(1,1, figsize = (18,9))

    gens = np.arange(0, num_generations)
    if v == 1:
        ax.plot(gens, maxfits, label = "Best individual's fitness")
        ax.plot(best_indivs_gen, maxfits[best_indivs_gen], 'r*', label = 'Optimal individual in algorithm')
    elif v == 2:
        ax.plot(gens, maxfits[:,0], label = "Best individual's fitness 0")
        ax.plot(gens, maxfits[:,1], label = "Best individual's fitness 1")
        ax.plot(gens, maxfits[:,2], label = "Best individual's fitness 2")
        ax.plot(gens, avgfits, label = "Average fitness of population")
        ax.plot(best_indivs_gen*np.ones(3), maxfits[best_indivs_gen,:], 'r*', label = 'Optimal individual in algorithm')
    else:
        print("Select proper v. Corresponding to the GA's v.")

    for extinction in extinction_generations: ax.axvline(extinction, c='k')
    ax.plot([],[], 'k', label = 'Extinction in the generation')
    ax.set_title('Evolution of fitness')
    ax.set_xlabel('Generations')
    ax.set_ylabel('Fitness')
    ax.legend(bbox_to_anchor = (1.04,1), borderaxespad=0)
    plt.tight_layout()
    return fig


def plot_bestind_normevol(bestsols, num_generations, params):
    # MODIFY THIS FUNCTION TO SHOW THE DIFFERENCES BETWEEN EXCITATORY AND INHI
    # BITORY NORMS!!! 
    ''' Returns the evolution of the norm of the best individual in each generation.'''
    fig, ax = plt.subplots(1,1)
    len_exc = np.count_nonzero(params['matrix_exc'])
    gens = np.arange(0, num_generations)
    norms_exc = np.array([np.sum(weight[0:len_exc]**2) for weight in bestsols])
    norms_inh = np.array([np.sum(weight[len_exc:]**2) for weight in bestsols])
    
    ax.plot(gens, norms_exc, label='Norm of all excitatory weights')
    ax.plot(gens, norms_inh, label='Norm of all inhibitory weights')

    ax.set_title('Norm of the best individual for each generation')
    ax.set_ylabel('Norm')
    ax.set_xlabel('Generations')
    ax.legend(loc='best')
    return fig


def plot_sigsingleJR(t, y, signal):
    '''Returns the dynamics of the signal and a single JR column in the same
    plot. y and signal should have equivalent sizes.'''
    xspan = (t[-10000], t[-1])
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18,9))
    for ii in range(signal.shape[0]):
        ax = axes[ii, 0]
        ax.plot(t, signal[ii,:], 'k')
        ax.set(xlabel='s', ylabel='Hz', title='Input signal', xlim=xspan,
               ylim=(70, 200))
    for ii in range(y.shape[0]): # y is output as a column vector, fuck 1d
        ax = axes[ii, 1]
        ax.plot(t, y[ii,:], 'r')
        ax.set(xlabel='s', ylabel='mV', title='JR dynamics', xlim=xspan,
               ylim=(-8, 20))
    plt.tight_layout()
    return fig


def plot_inputs(y, signals, params, t, newfolder):
    inputnodes = params['tuplenetwork'][0]
    ylim = (np.amin(y[0:3])-1, np.amax(y[0:3])+1)
    for ii in range(inputnodes):
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        axes[0].plot(t, signals[ii], 'k')
        axes[0].set(xlim=(10, params['tspan'][1]), xlabel='s', ylabel='Hz',
                    title='Input signal and response of node %i' % ii)
        axes[1].plot(t, y[ii], 'r')
        axes[1].set(ylim=ylim, xlim=(10, params['tspan'][1]), xlabel='s',
                    ylabel='mV')
        fig.savefig(newfolder + '/inputs_' + str(ii) + '.png')


def plot_fftoutputs(y, params, newfolder):
    nodeslast = params['tuplenetwork'][-1]
    idx = params['Nnodes'] - nodeslast
    fig, axes = plt.subplots(nodeslast, 1, figsize=(8, 6*nodeslast))
    for ii in range(nodeslast):
        aux = idx + ii
        f, psds = psd(y[aux], params['tstep'])
        axes[ii].semilogy(f, psds)
        axes[ii].set(xlim=(-0.01, 40), title='PSD of node %i output' % aux,
                     xlabel='f ($Hz$)', ylabel='PSD',
                     ylim=(10**(-2), 10**12))
    fig.savefig(newfolder + '/fftoutputs.png')


def plot_corrs(y, idx, params, newfolder):
    nnodes = params['Nnodes']
    pair = params['pairs'][idx]
    corr_array = np.zeros((nnodes, nnodes))
    tickslab = []
    for ii in range(nnodes):
        tickslab.append('y' + str(ii))
        for jj in range(nnodes):
            corr_array[ii, jj] = ccross(y[ii], y[jj], int((params['tspan'][-1]-20)/params['tstep']))

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(corr_array, vmin=0, vmax=1)
    ticks = np.linspace(0, nnodes-1, nnodes)
    ax.set(xticks=ticks, xticklabels=tickslab, yticks=ticks, yticklabels=tickslab,
           title='Cross-correlations between nodes. (%i, %i) set' % pair)
    fig.colorbar(im)
    fig.savefig(newfolder + '/corrs' + str(idx) + '.png')

    return fig


def draw_neural_net(ax, left, right, bottom, top, layer_sizes, conn_matrix, maxvalue, bandw = False):
    """
    Draw a neural network cartoon using matplotilb.

    :usage:
        fig = plt.figure(figsize=(12, 12))
        draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])

    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    """
    n_layers = len(layer_sizes)
    cmap = cm.get_cmap('viridis')
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    # Nodes
    node = 0
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            xcirc = n * h_spacing + left
            ycirc = layer_top - m * v_spacing
            circle = plt.Circle((xcirc, ycirc), v_spacing / 4.,
                                color='w', ec=cm.tab10(n), zorder=4, lw=3)
            ax.add_artist(circle)
            ax.annotate(str(node), xy=(xcirc, ycirc-0.01), zorder=5, xycoords='axes fraction', ha='center')
            node += 1
    # Edges
    init_o = 0
    init_m = 0
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        init_o += layer_sizes[n]
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                if bandw:
                    color = str(1 - conn_matrix[init_o + o, init_m + m]/maxvalue)
                else:
                    color = cmap(conn_matrix[init_o + o, init_m + m]/maxvalue)
                if bandw and float(color) > 0.95:
                    continue
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c=color)
                ax.add_artist(line)

        init_m += layer_size_a

    return ax


def plotcouplings(solution, matrix_exc, matrix_inh, minmaxvals, params, bandw=False):
    """Returns an imshow of the excitatory and inhibitory weight matrix plus the network diagrams. solution is
    the vector of the individiual with best fitness."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    weights_exc, weights_inh = individual_to_weights(solution, matrix_exc, matrix_inh)
    weights_exc = np.ma.masked_where(weights_exc == 0, weights_exc)
    weights_inh = np.ma.masked_where(weights_inh == 0, weights_inh)
    nnodes = matrix_exc.shape[0]
    ticks = np.linspace(0, nnodes-1, nnodes)
    if bandw:
        colormap = cm.get_cmap('Greys')
    else:
        colormap = cm.get_cmap('viridis')
    ax = axes[0, 0]
    ax.imshow(weights_exc, vmin=minmaxvals[0], vmax=minmaxvals[1], cmap=colormap)
    ax.set(title='Excitatory Coupling Coefficients', xlabel='Pre-Synaptic Node', ylabel='Post-Synaptic Node',
           xticks=ticks, yticks=ticks)
    ax.set_xticks(np.arange(-.5, params['Nnodes']-1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, params['Nnodes']-1, 1), minor=True)
    ax.grid(which='minor', color='k')

    ax = axes[0, 1]
    im = ax.imshow(weights_inh, vmin=minmaxvals[0], vmax=minmaxvals[1], cmap=colormap)
    ax.set(title='Inhibitory Coupling Coefficients', xlabel='Pre-Synaptic Node', ylabel='Post-Synaptic Node',
           xticks=ticks, yticks=ticks)
    ax.set_xticks(np.arange(-.5, params['Nnodes'] - 1, 1), minor=True)
    ax.set_yticks(np.arange(-.5, params['Nnodes'] - 1, 1), minor=True)
    ax.grid(which='minor', color='k')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    ax = axes[1, 0]
    draw_neural_net(ax, 0.1, 0.9, 0.1, 0.9, params['tuplenetwork'], weights_exc, minmaxvals[1], bandw)
    ax.axis('off')
    ax.set(title='Network diagram of excitatory weights')

    ax = axes[1, 1]
    draw_neural_net(ax, 0.1, 0.9, 0.1, 0.9, params['tuplenetwork'], weights_inh, minmaxvals[1], bandw)
    ax.axis('off')
    ax.set(title='Network diagram of inhibitory weights')
    return fig
