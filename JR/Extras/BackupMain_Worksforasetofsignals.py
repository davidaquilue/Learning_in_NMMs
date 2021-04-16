# Running Genetic Algorithm and Saving Figures in Cluster.
import os
path = os.getcwd()
#print(path) # First time to see how the format of the directory is
import numpy as np; import matplotlib.pyplot as plt
from plotfuns import plot3x3, plotcouplings3x3V2
from networkJR import obtaindynamicsNET
from matfuns import networkmatrix, regularity, maxcrosscorrelation
from multiprocessing import Pool, cpu_count
import GAlgs
from fitfuns import fitness_function_cross
from scipy import signal

# We first determine the paramaters of the JR model
params = {'A': 3.25, 'B': 22.0, 'v0': 6.0} 
params['a'], params['b'], params['e0'], params['pbar'], params['delta'], params['f'] = 100.0, 50.0, 2.5, 155.0, 65.0, 8.5
C = 133.5
params['C'], params['C1'], params['C2'], params['C3'], params['C4'] = C, C, 0.8*C, 0.25*C, 0.25*C # All dimensionless

params['r'] = 0.56 #mV^(-1)

params['delta'] = 72.09
params['f'] = 8.6
params['stimulation_mode'] = 1

# Now we define the network architecture. Always feedforward and we can decide wether we want recurrencies or not
params['tuplenetwork'] = (3,3,3)
params['recurrent'] = False
params['forcednodes'] = (0,1,2)
Nnodes, matrix =  networkmatrix(params['tuplenetwork'], params['recurrent'])
indivsize = 2*np.count_nonzero(matrix)
params['Nnodes'] = Nnodes
params['matrix'] = matrix
params['tstep'] = 0.001
params['tspan'] = (0,40)

# Maybe add some noise?
def creating_signals(t, amp, dc, freq, pair_comp):
    '''
    Returns three squared signals in a matrix, with dc component = dc, max amplitude = dc+amp and two rows that are complementary

    Inputs:
    t:          Time vector
    amp:        Amplitude that will be added to the dc value when the square signal is at +1
    dc:         Vertical offset of the squared signal.
    freq:       In hertz, frequency of the squared signal
    pair_comp:  2-uple defining which of the two rows will be complementary

    Outputs:
    signals:    (3, t.size) array containing the three signals in each row.  
    '''

    # Each signal will have a random phase and a random duty period.
    phase1 = np.random.random(); duty1 = np.random.random()
    phase2 = np.random.random(); duty2 = np.random.random()
    # The not complementary is:
    for ii in range(3):
        if ii not in pair_comp: non_comp = ii
    signals = np.zeros(shape = (3, t.size))
    signals[pair_comp[0]] = dc + amp*(1+signal.square(2*np.pi*(freq*t + phase1), duty1))/2
    signals[pair_comp[1]] = dc + amp*np.abs(1-(1+signal.square(2*np.pi*(freq*t + phase1), duty1))/2)
    signals[non_comp] = dc + amp*(1+signal.square(2*np.pi*(freq*t + phase2), duty2))/2


    return signals

t = np.linspace(params['tspan'][0], params['tspan'][1], int((params['tspan'][1] - params['tspan'][0])/params['tstep']))
freq = 2; amp = 250; dc = 120; pair_comp = (0,2)
params['signals'] = creating_signals(t, amp, dc, freq, pair_comp)

params['pairs'] = ((0,1), (0,2), (1,2))
# For each pair a matrix of signals, that is, a matrix o matrices

'''
params['individual'] = np.load('best_ind.npy')
y , t = obtaindynamicsNET(params, params['tspan'], params['tstep'], 3)

fig = plot3x3(y,t, 'small', params['tstep'])
plt.show()



'''
# This has to go before the if name = main and before running the program.

maxgene = 0.4*C
mingene = 0
toolbox, creator = GAlgs.initiate_DEAP(fitness_function_cross, params, generange = (mingene,maxgene), indsize = indivsize)
    
if __name__ == '__main__':
    last_test = int(os.listdir(path + '/Resultats')[-1])
    print('Last Test: ' + str(last_test))
    testn = last_test + 1
    newfolder = path + '/Resultats/' + str(testn)
    os.makedirs(newfolder)

    print('Actual test: ' + str(testn))
    with Pool(processes= 4) as piscina:
        num_generations = 10; popsize = 10; mutindprob = 0.5; coprob = 0.5; indsize = indivsize
        maxfits, avgfits, bestsols = GAlgs.main_DEAP(num_generations, popsize, mutindprob, coprob, indsize, toolbox, creator, 1, piscina)
        # Plot the maximum fitnesses and average fitnesses of each generation
        gens = np.linspace(1,num_generations, num_generations)
        plt.plot(gens, maxfits)
        plt.plot(gens, avgfits)
        plt.title('Evolution of fitness')
        plt.savefig(newfolder + "/fitness.jpg")
        plt.show()
        # Obtain the best solution of the population
        solution = bestsols[np.argmax(maxfits)]


        fig = plotcouplings3x3V2(solution, matrix, (mingene, maxgene))
        fig.savefig(newfolder + "/bestweights.jpg")
        plt.show()
        np.save(newfolder + '/best_ind', solution)

        norms = np.array([np.sum(weight**2) for weight in bestsols])
        plt.plot(gens, norms)
        plt.title('Evolution of the norm of the max fitness weights')
        plt.savefig(newfolder + "/normevol.jpg")
        plt.show()

        solution = np.array(solution)
        params['individual'] = solution
        # And let's observe the dynamics
        y , t = obtaindynamicsNET(params, params['tspan'], params['tstep'], 3)

        typeplot = 'small'
        fig = plot3x3(y,t, typeplot, params['tstep'])
        saving = newfolder + '/Dynamics' + typeplot + '.png'
        fig.savefig(saving)
        plt.show()
        print('Correlation 6 with 7: ' + str(maxcrosscorrelation(y[6], y[7])))
        print('Correlation 6 with 8: ' + str(maxcrosscorrelation(y[6], y[8])))
        print('Correlation 7 with 8: ' + str(maxcrosscorrelation(y[7], y[8])))

# Estaria guay que a lo mejor escupiera las diferentes autocorrelaciones para ver porqué son tan parecidas.
