'''Plotting functions.

Functions included: plot3x3, plot3x3_signals, plotanynet, plotcouplings3x3, plotcouplings3x3V2 '''
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
    plt.show()
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
    plt.show()
    # So that we can save it if we want:
    return fig


def plotanynet(y, t, span, tuplenetwork):
    ''' 
    Plots the dynamics of a ZxZxZx....xZ network of cortical columns. It is necessary that all layers have the same number of cortical columns.
    y:              Vector shaped (N_nodes, nsteps) where y[i,nsteps] represents the PSP of the pyramidal population of column i
    t:              Time vector
    span:           If 'large', 7 seconds of dynamics are presented, if 'small', 3 seconds. Enough for alpha rhythms
    tuplenetwork:   Tuple like (Z,Z,Z,Z,...,Z) representing the architecture of the network
    '''
    yspan = (np.min(y)-1,np.max(y)+1)
    nodesperlayer = tuplenetwork[0]
    layers = len(tuplenetwork)
    fig, axes = plt.subplots(nrows = layers, ncols = nodesperlayer)
    
    fig.subplots_adjust(hspace=0.5)
    fig.set_figheight(2*layers)
    fig.set_figwidth(4*nodesperlayer)
    
    if span == 'large':
        xspan = (t[-7001],t[-1])
    elif span == 'small':
        xspan = (t[-3000],t[-1])
        
    Nnodes = 0
    for nnodes in tuplenetwork: Nnodes += nnodes
    rr = 0; cc = 0
    for ii in range(Nnodes):
        ax = axes[cc,rr]; cc += 1
        if cc == nodesperlayer:
            rr += 1; cc = 0
        yy = y[ii]
        layerii = findlayer(ii,tuplenetwork)
        ax.plot(t, yy, color = cm.tab10(layerii))
        ax.set(xlim = xspan, ylim = yspan)
        ax.set_xlabel(r'time (s)', fontsize = labelfontsize)
        ax.set_ylabel(r'$y_1-y_2$', fontsize = labelfontsize)
        ax.tick_params(labelsize = labelticksize)
    
    plt.tight_layout()
    plt.show()
    # If we want to save the figure:
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

    plt.show()
    return fig

from networkJR import individual_to_weights
def plotcouplings3x3V2(solution, matrix, maxminvals):
    ''' Returns an imshow of the excitatory and inhibitory weight matrix. solution is the vector of the individiual with best fitness.'''
    fig, axes = plt.subplots(1, 2)
    weights_exc, weights_inh = individual_to_weights(solution, matrix)
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

    plt.show()
    return fig  

