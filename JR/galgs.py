"""Genetic Algorithm functions.

Functions included: mutUniform, initiate_DEAP, main_DEAP"""
from deap import base, creator, tools
from tqdm import tqdm
from matfuns import fastcrosscorrelation as ccross
from networkJR import obtaindynamicsNET
from plotfuns import plot_363, plotinputsoutputs, plot_corrs
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from itertools import repeat

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


# Mutation function, modified because the ones of the library don't fit exactly my needs:
def mutUniform(individual, low, high, indpb):
    """This function applies a uniform mutation with a range [low, high) on
    the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.
    :param individual: Individual to be mutated.
    :param low: Lower bound of the random value mutation
    :param high: Higher bound of the random value mutation
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        mu = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(high, Sequence):
        sigma = repeat(high, size)
    elif len(high) < size:
        raise IndexError("high must be at least the size of individual: %d < %d" % (len(high), size))

    for i, l, h in zip(range(size), low, high):
        if np.random.random() < indpb:
            individual[i] += np.random.uniform(l, h)

    return individual,


# DEAP Algorithm.
def initiate_DEAP(fitness_func, params, generange=(0, 1), indsize=18, mutprob=0.05, tmntsize=3, v=1):
    """
    Registers the different classes and methods required to run the DEAP genetic algorithm. Basic characteristics of the problem need to be passed:

    fitness_func:   The name of the fitness function 
    generange:      The (min,max) values of the individuals' genes.
    indsize:        The individual size. The amount of genes per individual.
    mutprob:        The probability of mutation of a certain gene once the individual has been chosen to mutate
    tmtnsize:       In the case of tournament selection, the size of the tournament.
    v:              Version of the GA used. v = 1 works for single output fit funcs. v = 2 works for 3-uple output fit funcs.

    One has to make sure that the initiate_DEAP functions is called before the if __name__ == '__main__'. The toolbox has to be registered before that (global scope)
    """
    if v == 1:
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))  # If wanted to minimize, weight negative
    elif v == 2:
        creator.create('FitnessMax', base.Fitness, weights=(1.0, 1.0, 1.0))
    else:
        print('Select a valid v (v = 1, 2)')
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # How we generate the genes of each individual:
    toolbox.register('genes', np.random.uniform, generange[0], generange[1])
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.genes, indsize)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('evaluate', fitness_func, params)  # The evaluation will be performed calling the alias evaluate. It is important to not fix its argument here.
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', mutUniform,
                     low=generange[0],
                     high=generange[1], indpb=mutprob)
    if v == 1:
        toolbox.register('select', tools.selTournament, tournsize=tmntsize)
    # Since tournament does not make multivariate selection, NSGA shall be used for more than one fitness values.
    elif v == 2:
        toolbox.register('select', tools.selNSGA2)
    return toolbox, creator


def main_DEAP(num_generations, popsize, mutindprob, coprob, indsize, toolbox, creator, parallel, pool = None, v = 2):
    """
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
    """
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
        
        # I noticed that i am not implementing elitism which might be very
        # important in our case.
        elite = toolbox.select(pop, 6)
        offspring = toolbox.select(pop, len(pop)-6)
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
        offspring.extend(elite)
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


def main_DEAP_extinction(num_generations, popsize, mutindprob, coprob,
                         indsize, toolbox, creator, parallel, L, cheatlist,
                         pool=None, v=2):
    """
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
    """
    if parallel:
        toolbox.register('map', pool.map)
    else:
        toolbox.register('map', map)
    
    if v == 1:
        maxfits = []
    elif v == 2:
        maxfits = np.zeros((num_generations, 3))
    avgfits = []
    overall_fit = []  # This is the array that we will use to determine if there is extinction
    gens_passed_after_ext = 0
    bestsols = np.zeros((num_generations, indsize))
    pop = toolbox.population(n=popsize)  # pop will be a list of popsize individuals
    
    # The cheatlist is the first individual of all, passed through the main script.
    # We can choose an initial individual that we know that will work to start the population,
    # make it converge faster.
    pop[0] = creator.Individual(cheatlist)

    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    extinction_generations = []
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit # We store the results in the value of the fitness attribute of the individual class.

    for i in tqdm(range(num_generations), desc = 'Genetic Algorithm Running'):
        # Selection of the best individuals from the previous population
        # I noticed that i am not implementing elitism which might be very
        # important in our case.
        elite_size = int(np.ceil(0.10*len(pop)/2)*2)  # Make sure it's even 
        elite = toolbox.select(pop, elite_size)
        offspring = toolbox.select(pop, len(pop)-elite_size)
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
        offspring.extend(elite)
        pop[:] = offspring

        # Storing different results
        if v == 1:  # If version 1, only one value of fitness is returned
            fits = np.array([ind.fitness.values[0] for ind in pop])
            idx = np.argmax(fits)
            maxfit = fits[idx]
            maxfits.append(maxfit)
            avgfits.append(np.mean(fits))

        elif v == 2:  # If version 2, three different fitness values are returned
            fit0 = np.array([ind.fitness.values[0] for ind in pop])
            fit1 = np.array([ind.fitness.values[1] for ind in pop])
            fit2 = np.array([ind.fitness.values[2] for ind in pop])
            fits = (fit0+fit1+fit2)/3
            idx = np.argmax(fits)
            maxfits[i, :] = np.array([fit0[idx], fit1[idx], fit2[idx]])
            maxfit = fits[idx]
            avgfits.append(np.mean(fits))

        overall_fit.append(maxfit)
        bestsols[i, :] = pop[idx]
        gens_passed_after_ext += 1

        # Now we check for extinction:
        if gens_passed_after_ext > L:
            extinction = True
            for ffit in overall_fit[i-L+1:]:
                # We check if one of the last L-1 best fitnesses is bigger than that of the actualgen-L
                if np.abs(ffit) > overall_fit[i-L] + 0.1*np.abs(overall_fit[i-L]):
                    # If only one of the last L fitnesses is 20% better than the actualgen-L, no extinction
                    extinction = False

            if extinction:
                pop = toolbox.population(n=popsize)  # We rewrite the population with newly generated individuals
                pop[0] = creator.Individual(bestsols[i, :].tolist())
                # But the first of the individuals will be the best solution of the generation before extinction
                # It's important to use the creator individual so that it will have the same attributes of before

                fitnesses = list(toolbox.map(toolbox.evaluate, pop))  # Reevaluate the fitnesses of the new population
                for ind, fit in zip(pop, fitnesses):
                    ind.fitness.values = fit

                extinction_generations.append(i)  # We store the generations in which there has been an extinction.
                gens_passed_after_ext = 0

    return maxfits, avgfits, bestsols, extinction_generations


def test_solution(params, newfolder, whatplot='net', rangeplot='large'):
    """!!!IT WORKS FOR THE 363 NETWORK ONLY RIGHT NOW!!!
    This function obtains the dynamics from the testing set and plots
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
    idx4corr = int((params['tspan'][1] - params['tspan'][0] - 20)/params['tstep'])
    # Print the correlations between all the nodes. MAIN RESULTS
    pairsi = ((1, 2), (0, 2), (0, 1))
    pairsf = ((10, 11), (9, 11), (9, 10))
    listheaders = ['Signals', 'Corrs', 'First Layer', 'Corrs', 'Last layer', 'Corrs']  # Table things
    for ii, synch_pair in enumerate(outpairs):
        # Iterate over evey pair of possible correlations
        saving = newfolder + '/Dynamics' + str(ii) + rangeplot + '.png'

        # For each pair of correlated inputs there are n realizations in the test_set

        for nn, signalsforpair in enumerate(params['test_dataset'][ii]):
            f.write('Sample %i from set %i:\n' % (nn, ii))
            f.write('Nodes %i and %i should be correlated: \n' % pairsf[ii])
            # We obtain the dynamics for the sample
            params['signals'] = signalsforpair
            y, t = obtaindynamicsNET(params, params['tspan'], params['tstep'], 3)

            # We calculate all the correlations and save them in a table.
            # Correlations between input signals
            ccps = [ccross(signalsforpair[pair[0]], signalsforpair[pair[1]], idx4corr) for pair in pairsi]
            # Correlations between nodes in the first layer
            ccnis = [ccross(y[pair[0]], y[pair[1]], idx4corr) for pair in pairsi]
            # Correlations between nodes in the last layer
            ccnfs = [ccross(y[pair[0]], y[pair[1]], idx4corr) for pair in pairsf]

            # Build the table
            table = []
            for kk in range(3):
                table.append(['p%i and p%i' % pairsi[kk], ccps[kk], 'nodes %i and %i' % pairsi[kk],
                              ccnis[kk], 'nodes %i and %i' % pairsf[kk], ccnfs[kk]])
            content = tabulate(table, headers=listheaders, floatfmt=".4f")
            f.write(content)
            f.write('\n \n')

        # Then we obtain a sample plot for each of the correlation pairs.
        if ii == 0:
            fig0 = pltfun(y, t, rangeplot, params, True, params['signals'])
            corrs0 = plot_corrs(y, ii, params, newfolder)
            fig0.savefig(saving)
        elif ii == 1:
            fig1 = pltfun(y, t, rangeplot, params, True, params['signals'])
            corrs1 = plot_corrs(y, ii, params, newfolder)
            fig1.savefig(saving)
        else:
            fig2 = pltfun(y, t, rangeplot, params, True, params['signals'])
            corrs2 = plot_corrs(y, ii, params, newfolder)
            fig2.savefig(saving)
    f.close()
    f = open(newfolder+'/correlation.txt', 'r')
    print(f.read())  # This way we save the results in a .txt file for later
    plt.show()


def get_OP(params):
    maxval = params['maxvalue']
    matrix_exc = params['matrix_exc']
    matrix_inh = params['matrix_inh']
    idexes = np.nonzero(matrix_exc)
    exc_w = np.copy(matrix_exc)
    exc_w[idexes] = 10**(-4)
    inh_w = np.copy(matrix_inh)
    inh_w[idexes] = 10**(-4)

    # Editem les diferents connexions. Aquí es on fem el copy paste per obtenir els diferents resultats en directe.
    exc_w[3, [0, 1]] = [1*maxval, 1*maxval]
    exc_w[4, [0, 1]] = [1*maxval, 1*maxval]
    exc_w[5, [0, 2]] = [1*maxval, 1*maxval]
    exc_w[6, [0, 2]] = [1*maxval, 1*maxval]
    exc_w[7, [1, 2]] = [1*maxval, 1*maxval]
    exc_w[8, [1, 2]] = [1*maxval, 1*maxval]

    exc_w[9, [3, 4]] = [1*maxval, 1*maxval]
    exc_w[10, [3, 4]] = [1*maxval, 1*maxval]
    exc_w[9, [5, 6]] = [1*maxval, 1*maxval]
    exc_w[11, [5, 6]] = [1*maxval, 1*maxval]
    exc_w[10, [7, 8]] = [1*maxval, 1*maxval]
    exc_w[11, [7, 8]] = [1*maxval, 1*maxval]

    inh_w[3, [0, 1]] = [0.5*maxval, 0.5*maxval]
    inh_w[4, [0, 1]] = [0.5*maxval, 0.5*maxval]
    inh_w[5, [0, 2]] = [0.5*maxval, 0.5*maxval]
    inh_w[6, [0, 2]] = [0.5*maxval, 0.5*maxval]
    inh_w[7, [1, 2]] = [0.5*maxval, 0.5*maxval]
    inh_w[8, [1, 2]] = [0.5*maxval, 0.5*maxval]

    inh_w[9, [3, 4]] = [0.5*maxval, 0.5*maxval]
    inh_w[10, [3, 4]] = [0.5*maxval, 0.5*maxval]
    inh_w[9, [5, 6]] = [0.5*maxval, 0.5*maxval]
    inh_w[11, [5, 6]] = [0.5*maxval, 0.5*maxval]
    inh_w[10, [7, 8]] = [0.5*maxval, 0.5*maxval]
    inh_w[11, [7, 8]] = [0.5*maxval, 0.5*maxval]
    # Aquí acaba la part de copy paste

    OP = np.append(exc_w[idexes].flatten(), inh_w[idexes].flatten())
    OPlist = OP.tolist()

    return OPlist
