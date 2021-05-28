'''Genetic Algorithm functions. 

Functions included: initiate_DEAP, main_DEAP'''
from deap import base, creator, tools
from tqdm import tqdm
from matfuns import fastcrosscorrelation as ccross
from networkJR import obtaindynamicsNET
from plotfuns import plot_363, plotinputsoutputs
import matplotlib.pyplot as plt
import numpy as np

# DEAP Algorithm.
def initiate_DEAP(fitness_func, params, generange = (0,1), indsize = 18, mutprob = 0.05, tmntsize = 3, v = 1):
    '''
    Registers the different classes and methods required to run the DEAP genetic algorithm. Basic characteristics of the problem need to be passed:

    fitness_func:   The name of the fitness function 
    generange:      The (min,max) values of the individuals' genes.
    indsize:        The individual size. The amount of genes per individual.
    mutprob:        The probability of mutation of a certain gene once the individual has been chosen to mutate
    tmtnsize:       In the case of tournament selection, the size of the tournament.
    v:              Version of the GA used. v = 1 works for single output fit funcs. v = 2 works for 3-uple output fit funcs.

    One has to make sure that the initiate_DEAP functions is called before the if __name__ == '__main__'. The toolbox has to be registered before that (global scope)
    '''
    if v == 1:
        creator.create('FitnessMax', base.Fitness, weights = (1.0,)) # If wanted to minimize, weight negative
    elif v == 2:
        creator.create('FitnessMax', base.Fitness, weights = (1.0, 1.0, 1.0))
    creator.create('Individual', list, fitness = creator.FitnessMax)
    toolbox = base.Toolbox()
    # How we generate the genes of each individual:
    toolbox.register('genes', np.random.uniform, generange[0], generange[1])
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.genes, indsize)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('evaluate', fitness_func, params) # The evaluation will be performed calling the alias evaluate. It is important to not fix its argument here. 
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, 
                     mu=(generange[1] - generange[0])/2, 
                     sigma=(generange[1] - generange[0])/20, indpb=mutprob)  # Real number only mutation. Let's see how it works
    toolbox.register('select', tools.selTournament, tournsize=tmntsize)

    return toolbox, creator


def main_DEAP(num_generations, popsize, mutindprob, coprob, indsize, toolbox, creator, parallel, pool = None, v = 2):
    '''
    Runs the DEAP Genetic Algorithm. Main characteristics of the algorithm need to be passed now:

    num_generations: Number of generations that the GA will run
    popsize:        Population size for each generation
    mutindprob:     Probability of selecting an individual for mutation
    coprob:         Probability of selecting a pair of individuals for crossover
    indsize:        Size of the individuals (number of genes)
    toolbox:        The DEAP toolbox created in the initiate_DEAP function
    creator:        The DEAP creator class created in the initiate_DEAP function
    parallel:       Boolean indicating if we want to evaluate fitness functions on individuals in a parallel manner
    pool:           The pool of processes in the case that we use parallelization. with Pool(processes = 4) as pool.
    Version of the GA used. v = 1 works for single output fit funcs. v = 2 works for 3-uple output fit funcs.


    Outputs:
    bestsols:   An array containing, in each row, the best individual of each generation
    maxfits:    A list containing the largest fitness in each generation
    avgfits:    A list containing the average fitness in each generation

    If we are using parallel processing, the with Pool as pool has to be called inside if __name__=='__main__'
    '''
    if parallel:
        
        toolbox.register('map', pool.map)
    else:
        toolbox.register('map', map)
    
    if v == 1:
        maxfits = []
    elif v == 2:
        maxfits = np.zeros((num_generations, 3))
    avgfits = []
    bestsols = np.zeros((num_generations, indsize))
    pop = toolbox.population(n = popsize) # pop will be a list of popsize individuals
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit # We store the results in the value of the fitness attribute of the individual class.

    for i in tqdm(range(num_generations), desc = 'Genetic Algorithm Running'):
        # Selection of the best individuals from the previous population
        offspring = toolbox.select(pop,len(pop))
        offspring = list(toolbox.map(toolbox.clone, offspring)) 

        # Mutation and crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < coprob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < mutindprob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Now we recalculate the fitness of the newly created offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # And finally reassign the offspring as the new population:
        pop[:] = offspring

        # Storing different results
        if v == 1:  # If version 1, only one value of fitness is returned
            fits = np.array([ind.fitness.values[0] for ind in pop])
            idx = np.argmax(fits)
            maxfits.append(np.amax(fits))
            avgfits.append(np.mean(fits))

        elif v == 2:  # If version 2, three different fitness values are returned
            fit0 = np.array([ind.fitness.values[0] for ind in pop])
            fit1 = np.array([ind.fitness.values[1] for ind in pop])
            fit2 = np.array([ind.fitness.values[2] for ind in pop])
            fits = (fit0+fit1+fit2)/3
            idx = np.argmax(fits)
            maxfits[i, :] = np.array([fit0[idx], fit1[idx], fit2[idx]])
            avgfits.append(np.mean(fits))

        bestsols[i, :] = np.array(pop[idx])

    return maxfits, avgfits, bestsols


def main_DEAP_extinction(num_generations, popsize, mutindprob, coprob, indsize, toolbox, creator, parallel, L, pool=None, v=2):
    '''
    Runs the DEAP Genetic Algorithm. Main characteristics of the algorithm need to be passed now:

    num_generations: Number of generations that the GA will run
    popsize:        Population size for each generation
    mutindprob:     Probability of selecting an individual for mutation
    coprob:         Probability of selecting a pair of individuals for crossover
    indsize:        Size of the individuals (number of genes)
    toolbox:        The DEAP toolbox created in the initiate_DEAP function
    creator:        The DEAP creator class created in the initiate_DEAP function
    parallel:       Boolean indicating if we want to evaluate fitness functions on individuals in a parallel manner
    L:              Number of generations that have to pass without an increase in the maximal fitness before an extinction
    pool:           The pool of processes in the case that we use parallelization. with Pool(processes = 4) as pool.
    Version of the GA used. v = 1 works for single output fit funcs. v = 2 works for 3-uple output fit funcs.


    Outputs:
    bestsols:   An array containing, in each row, the best individual of each generation
    maxfits:    A list containing the largest fitness in each generation
    avgfits:    A list containing the average fitness in each generation

    If we are using parallel processing, the with Pool as pool has to be called inside if __name__=='__main__'
    '''
    if parallel:
        
        toolbox.register('map', pool.map)
    else:
        toolbox.register('map', map)
    
    if v == 1:
        maxfits = []
    elif v == 2:
        maxfits = np.zeros((num_generations, 3))
    avgfits = []
    overall_fit = [] # This is the array that we will use to determine if there is extinction
    gens_passed_after_ext = 0
    bestsols = np.zeros((num_generations, indsize))
    pop = toolbox.population(n = popsize) # pop will be a list of popsize individuals

    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    extinction_generations = []
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit # We store the results in the value of the fitness attribute of the individual class.

    for i in tqdm(range(num_generations), desc = 'Genetic Algorithm Running'):
        # Selection of the best individuals from the previous population
        offspring = toolbox.select(pop,len(pop))
        offspring = list(toolbox.map(toolbox.clone, offspring)) 

        # Mutation and crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < coprob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < mutindprob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Now we recalculate the fitness of the newly created offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # And finally reassign the offspring as the new population:
        pop[:] = offspring

        # Storing different results
        if v == 1: # If version 1, only one value of fitness is returned
            fits = np.array([ind.fitness.values[0] for ind in pop])
            idx = np.argmax(fits)
            maxfit = fits[idx]
            maxfits.append(maxfit)
            avgfits.append(np.mean(fits))

        elif v == 2: # If version 2, three different fitness values are returned
            fit0 = np.array([ind.fitness.values[0] for ind in pop])
            fit1 = np.array([ind.fitness.values[1] for ind in pop])
            fit2 = np.array([ind.fitness.values[2] for ind in pop])
            fits = (fit0+fit1+fit2)/3
            idx = np.argmax(fits)
            maxfits[i,:] = np.array([fit0[idx], fit1[idx], fit2[idx]])
            maxfit = fits[idx]
            avgfits.append(np.mean(fits))
        
        overall_fit.append(maxfit)
        bestsols[i,:] = np.array(pop[idx])
        gens_passed_after_ext += 1

        # Now we check for extinction:
        if gens_passed_after_ext  > L:
            extinction = True
            for ffit in overall_fit[i-L+1:]:# We check if one of the last L-1 best fitnesses is bigger than the fitness L generations before
                if np.abs(ffit) > np.abs(1.2*overall_fit[i-L]):
                    # If only one of the last fitnesses is 20% than the first one of the L before the actual generations, no extinction
                    extinction = False

            if extinction:
                pop = toolbox.population(n = popsize) # We rewrite the population with newly generated individuals
                pop[0] = creator.Individual(bestsols[i, :].tolist())  # But the first of the individuals will be the best solution of the generation before extinction
                # It's important to use the creator individual so that it will have the same attributes of before
                # Same process as before.
                fitnesses = list(toolbox.map(toolbox.evaluate, pop))
                for ind, fit in zip(pop, fitnesses):
                    ind.fitness.values = fit

                extinction_generations.append(i) # In this array we store the generations in which there has been an extinction.
                gens_passed_after_ext = 0

    return maxfits, avgfits, bestsols, extinction_generations


def test_solution(params, newfolder, whatplot='net', rangeplot ='large'):
    """This function obtains the dynamics from the testing set and plots
    some of the dynamics. It also prints and saves the correlations between
    the output nodes which will indicate if the solution has been able 
    to learn."""
    f = open(newfolder+'/correlation.txt', 'a+')
    outpairs = params['output_pairs']

    if whatplot == 'inout':
        pltfun = plotinputsoutputs
    elif whatplot == 'net':
        pltfun = plot_363
    else:
        print('Select a valid whatplot string.')

    # First of all print the output correlations and whether the learning
    # process was a success or a fail.
    for ii, synch_pair in enumerate(outpairs):
        # Iterate over evey pair of signals
        saving = newfolder + '/Dynamics' + str(ii) + rangeplot + '.png'

        # For each pair of correlated inputs there are n realizations in the test_set

        for nn, signalsforpair in enumerate(params['test_dataset'][ii]):
            f.write('%i set, %i element in the set: \n' % (ii, nn))
            params['signals'] = signalsforpair
            y, t = obtaindynamicsNET(params, params['tspan'], params['tstep'], 3)

            f.write('Nodes %i and %i should be synchronized:' % synch_pair)
            f.write('\nCorrelation %i with %i: ' % (9, 10) + str(ccross(y[9], y[10])))
            f.write('\nCorrelation %i with %i: ' % (9, 11) + str(ccross(y[9], y[11])))
            f.write('\nCorrelation %i with %i: ' % (10, 11) + str(ccross(y[10], y[11])) + '\n \n') 

        # Then obtain a plot for each of the correlation pairs.
        if ii == 0:
            fig0 = pltfun(y, t, rangeplot, params, True, params['signals'])
            fig0.savefig(saving)
        elif ii == 1:
            fig1 = pltfun(y, t, rangeplot, params, True, params['signals'])
            fig1.savefig(saving)
        else:
            fig2 = pltfun(y, t, rangeplot, params, True, params['signals'])
            fig2.savefig(saving)
    f.close()
    f = open(newfolder+'/correlation.txt', 'r')
    print(f.read())  # This way we save the results in a .txt file for later
    plt.show()
