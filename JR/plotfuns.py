'''Plotting functions.

Functions included: plot3x3, plot3x3_signals, plotanynet, plotcouplings3x3, plotcouplings3x3V2, plotgenfit,
plot_bestind_normevol '''

from matplotlib import cm
from matfuns import psd, findlayer
import numpy as np
import matplotlib.pyplot as plt

# General fontsizes
labelfontsize = 15
labelticksize = 15
titlefontsize = 20
suptitlefontsize = 26

def plot3x3(y,t, span, tstep):
    ''' 
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
    '''
    fig, axes = plt.subplots(nrows = 3, ncols = 3)
    fig.subplots_adjust(hspace=0.5)
    fig.set_figheight(2*3)
    fig.set_figwidth(12)
    # Maybe this should be always fixed so we always work in the same span.
    yspan = (np.min(y)-1,np.max(y)+1)

    if span == 'large':
        xspan = (t[-7001],t[-1])
    elif span == 'small':
        xspan = (t[-3000],t[-1])

    p1 = 0; p2 = 0; p3 = 0
    for ii in range(9):
        yy = y[ii]
        f, PSD = psd(yy,tstep)
        maxf = np.abs(f[np.argmax(PSD)])
        lab = 'Node: ' + str(ii)
        if ii < 3:
            ax = axes[p1,0]; p1 = p1+1
            ax.plot(t, yy, 'r', label = lab)
            ax.set(xlim = xspan, ylim = yspan)
            ax.set_xlabel(r'time (s)', fontsize = labelfontsize)
            ax.set_ylabel(r'$y_1-y_2$', fontsize = labelfontsize)
            ax.set_title(r'max of PSD f = %g Hz' %maxf)
            ax.tick_params(labelsize = labelticksize)

        elif ii >= 3 and ii < 6:
            ax = axes[p2,1]; p2 = p2+1
            ax.plot(t, yy,'b', label = lab)
            ax.set(xlim = xspan, ylim = yspan)
            ax.set_xlabel(r'time (s)', fontsize = labelfontsize)
            ax.set_ylabel(r'$y_1-y_2$', fontsize = labelfontsize)
            ax.set_title(r'max of PSD f = %g Hz' %maxf)
            ax.tick_params(labelsize = labelticksize)
            
        else:
            ax = axes[p3,2]; p3 = p3+1
            ax.plot(t, yy,'g', label = lab)
            ax.set(xlim = xspan, ylim = yspan)
            ax.set_xlabel(r'time (s)', fontsize = labelfontsize)
            ax.set_ylabel(r'$y_1-y_2$', fontsize = labelfontsize)
            ax.set_title(r'max of PSD f = %g Hz' %maxf)
            ax.tick_params(labelsize = labelticksize)

    plt.tight_layout()
    # So that we can save it if we want:
    return fig


def plot3x3_signals(y,t, span, tstep, signals):
    ''' 
    Plots the dynamics of a 3x3x3 network of cortical columns. First layer red, second blue and third green. N_nodes = 9
    Inputs:
    y:      Vector shaped (N_nodes, nsteps) where y[i,nsteps] represents the PSP of the pyramidal population of column i
    t:      Time vector
    span:   If 'large', 7 seconds of dynamics are presented, if 'small', 3 seconds. Enough for alpha rhythms
    tstep:  Timestep in the simulations, used to obtain the maxima of PSD of each node.
    '''
    fig, axes = plt.subplots(nrows = 3, ncols = 4)
    fig.subplots_adjust(hspace=0.5)
    fig.set_figheight(2*3)
    fig.set_figwidth(12)
    # Maybe this should be always fixed so we always work in the same span.
    yspan = (np.min(y)-1,np.max(y)+1)

    if span == 'large':
        xspan = (t[-7001],t[-1])
    elif span == 'small':
        xspan = (t[-3000],t[-1])
    
    p1 = 0
    for ii in range(3):
        ax = axes[ii, 0]; p1 = p1+1
        ax.plot(t[-10000:], signals[ii, -10000:], 'k')
        ax.set(xlim = xspan, ylim = (np.amin(signals[ii]), 1.3*np.amax(signals[ii])))
        ax.set_xlabel(r'time (s)', fontsize = labelfontsize)
        ax.set_ylabel(r'$Hz$', fontsize = labelfontsize)
        ax.tick_params(labelsize = labelticksize)

    p1 = 0; p2 = 0; p3 = 0
    for ii in range(9):
        yy = y[ii]
        f, PSD = psd(yy,tstep)
        maxf = np.abs(f[np.argmax(PSD)])
        lab = 'Node: ' + str(ii)
        if ii < 3:
            ax = axes[p1,1]; p1 = p1+1
            ax.plot(t, yy, 'r', label = lab)
            ax.set(xlim = xspan, ylim = yspan)
            ax.set_xlabel(r'time (s)', fontsize = labelfontsize)
            ax.set_ylabel(r'$y_1-y_2$', fontsize = labelfontsize)
            ax.set_title(r'max of PSD f = %g Hz' %maxf)
            ax.tick_params(labelsize = labelticksize)

        elif ii >= 3 and ii < 6:
            ax = axes[p2,2]; p2 = p2+1
            ax.plot(t, yy,'b', label = lab)
            ax.set(xlim = xspan, ylim = yspan)
            ax.set_xlabel(r'time (s)', fontsize = labelfontsize)
            ax.set_ylabel(r'$y_1-y_2$', fontsize = labelfontsize)
            ax.set_title(r'max of PSD f = %g Hz' %maxf)
            ax.tick_params(labelsize = labelticksize)
            
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


def plotanynet(y, t, span, params, bool_sig, signals):
    ''' 
    Plots the dynamics of a ZxZxZx....xZ network of cortical columns. It is necessary that all layers have the same number of cortical columns.
    y:              Vector shaped (N_nodes, nsteps) where y[i,nsteps] represents the PSP of the pyramidal population of column i
    t:              Time vector
    span:           If 'large', 7 seconds of dynamics are presented, if 'small', 3 seconds. Enough for alpha rhythms
    params:         Dictionary of parameters
    bool_sig:       Boolean indicating if we want to add the input signals to the plot
    signals:        Z x timesteps array containing the different input signals.
    '''
    tuplenetwork = params['tuplenetwork']
    yspan = (np.min(y)-1,np.max(y)+1)
    nodesperlayer = tuplenetwork[0]
    layers = len(tuplenetwork)
    Nnodes = params['Nnodes']
    fig, axes = plt.subplots(nrows = nodesperlayer, ncols = layers+1)
    
    fig.subplots_adjust(hspace=0.5)
    fig.set_figheight(2*layers)
    fig.set_figwidth(4*nodesperlayer)

    if span == 'large':
        xspan = (t[-7001],t[-1])
    elif span == 'small':
        xspan = (t[-3000],t[-1])

    cc = 0
    if bool_sig:
        for ii in range(signals.shape[0]):
            ax = axes[ii, cc]
            ax.plot(t[-10000:], signals[ii, -10000:], 'k')
            ax.set(xlim = xspan, ylim = (np.amin(signals[ii]), 1.3*np.amax(signals[ii])))
            ax.set_xlabel(r'time (s)', fontsize = labelfontsize)
            ax.set_ylabel(r'$Hz$', fontsize = labelfontsize)
            ax.tick_params(labelsize = labelticksize)
        cc += 1

    rr = 0
    for ii in range(Nnodes):
        if rr == nodesperlayer:
            cc += 1; rr = 0
            print(cc, rr)
        ax = axes[rr,cc]
        yy = y[ii]
        layerii = findlayer(ii,tuplenetwork)
        ax.plot(t, yy, color = cm.tab10(layerii))
        ax.set(xlim = xspan, ylim = yspan)
        ax.set_xlabel(r'time (s)', fontsize = labelfontsize)
        ax.set_ylabel(r'$y_1-y_2$', fontsize = labelfontsize)
        ax.tick_params(labelsize = labelticksize)
        rr += 1
    
    plt.tight_layout()
    return fig

def plotinputsoutputs(y, t, span, params, bool_sig, signals):
    ''' A function that plots input signals and first and last layers' dynamics. 
    First and last layer should have the same nodes. This function will allow to obtain results from more bizarre
    network architectures.'''
    nodesfirstlast = params['tuplenetwork'][0]
    yspan = (np.min(y)-1,np.max(y)+1)
    if span == 'large':
        xspan = (t[-7001],t[-1])
    elif span == 'small':
        xspan = (t[-3000],t[-1])
    
    fig, axes = plt.subplots(nrows = nodesfirstlast, ncols = 3)
    # Plot the signals first
    cc = 0
    for ii in range(signals.shape[0]):
        ax = axes[ii, cc]
        ax.plot(t[-10000:], signals[ii, -10000:], 'k')
        ax.set(xlim = xspan, ylim = (np.amin(signals[ii]), 1.3*np.amax(signals[ii])))
        ax.set_xlabel(r'time (s)', fontsize = labelfontsize)
        ax.set_ylabel(r'$Hz$', fontsize = labelfontsize)
        ax.tick_params(labelsize = labelticksize)
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
    ''' 
    Returns an imshow of the value of the couplings of a certain set of weights.
    Valid only for the case of one excitatory and inhibitory coefficient per column.
    solution = [alpha0, alpha1, alpha2,..., alpha8, beta8, beta7, ..., beta0]'''
    fig, axes = plt.subplots(1, 2)
    minim = np.amin(solution); maxim = np.amax(solution)
    alphas = np.reshape(solution[0:9],(3,3)).T
    betas = np.reshape(np.flip(solution[9:]),(3,3)).T
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

from networkJR import individual_to_weights
def plotcouplings3x3V2(solution, matrix_exc, matrix_inh, maxminvals):
    ''' Returns an imshow of the excitatory and inhibitory weight matrix. solution is the vector of the individiual with best fitness.'''
    fig, axes = plt.subplots(1, 2)
    weights_exc, weights_inh = individual_to_weights(solution, matrix_exc, matrix_inh)
    ax = axes[0]
    ax.imshow(weights_exc, vmin = maxminvals[0], vmax = maxminvals[1])
    ax.set_title('Excitatory Coupling Coefficients')
    ax.set_xlabel('Node')
    ax.set_ylabel('Node')
    ax = axes[1]
    im = ax.imshow(weights_inh, vmin = maxminvals[0], vmax = maxminvals[1])
    ax.set_title('Inhibitory Coupling Coefficients')
    ax.set_xlabel('Node')
    ax.set_ylabel('Node')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    return fig  

# Plotting GA results:
def plot_genfit(num_generations, maxfits, avgfits, best_indivs_gen, extinction_generations = [], v = 1):
    '''Returns the plot of the highest and average fitnesses per generation. Additionally 
    indicates extinctions and the best individual's fitnesses'''
    fig, ax = plt.subplots(1,1)

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
    ax.legend(loc='best')

    return fig

def plot_bestind_normevol(bestsols, num_generations):
    ''' Returns the evolution of the norm of the best individual in each generation.'''
    fig, ax = plt.subplots(1,1)
    gens = np.arange(0, num_generations)
    norms = np.array([np.sum(weight**2) for weight in bestsols])
    ax.plot(gens, norms)
    ax.set_title('Norm of the best individual for each generation')
    ax.set_ylabel('Norm')
    ax.set_xlabel('Generations')
    return fig