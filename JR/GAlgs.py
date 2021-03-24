# Genetic Algorithms. Two modules: PyGAD and DEAP
from deap import base, creator, tools
from tqdm import tqdm; import numpy as np

# DEAP Algorithm. When using parallelization, one will have to open a pool using 'with' or if __name__ = main... pass the pool and close the pool when the algorithm is finished.
def initiate_DEAP(fitness_func, parallel, generange = (0,1), indsize = 18, mutprob = 0.05, tmntsize = 3, pool = None):
    '''
    Registers the different classes and methods required to run the DEAP genetic algorithm. Basic characteristics of the problem need to be passed:

    fitness_func:   The name of the fitness function 
    parallel:       Boolean indicating if we want to evaluate fitness functions on individuals in a parallel manner 
    generange:      The (min,max) values of the individuals' genes.
    indsize:        The individual size. The amount of genes per individual.
    mutprob:        The probability of mutation of a certain gene once the individual has been chosen to mutate
    tmtnsize:       In the case of tournament selection, the size of the tournament.
    pool:           The pool of processes in the case we use parallelization
    '''
    creator.create('FitnessMax', base.Fitness, weights = (1.0,)) # If wanted to minimize, weight negative
    creator.create('Individual', list, fitness = creator.FitnessMax)

    toolbox = base.Toolbox()
    # How we generate the genes of each individual:
    toolbox.register('genes', np.random.uniform, generange[0], generange[1])
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.genes, indsize)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('evaluate', fitness_func) # The evaluation will be performed calling the alias evaluate. It is important to not fix its argument here. 
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, mu = (generange[1]- generange[0])/2, sigma = (generange[1]- generange[0])/20,indpb = mutprob) # Real number only mutation. Let's see how it works
    toolbox.register('select', tools.selTournament, tournsize = 3)
    if parallel:
        # Initialization of the process pool and setting up the map function
        toolbox.register('map', pool.map)
    else:
        toolbox.register('map', map)


def main( num_generations, popsize, mutindprob, coprob, indsize):
    '''
    Runs the DEAP Genetic Algorithm. Main characteristics of the algorithm need to be passed now:

    num_generations: Number of generations that the GA will run
    popsize:        Population size for each generation
    mutindprob:     Probability of selecting an individual for mutation
    coprob:         Probability of selecting a pair of individuals for crossover

    Outputs:
    bestsols:   An array containing, in each row, the best individual of each generation
    maxfits:    A list containing the largest fitness in each generation
    avgfits:    A list containing the average fitness in each generation
    '''
    bestsols = np.zeros((num_generations, indsize))
    maxfits = []
    avgfits = []
    pop = toolbox.population(n = popsize) # pop will be a list of popsize individuals

    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit # We store the results in the value of the fitness attribute of the individual class.


    for i in tqdm(range(num_generations)):
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
        fits = np.array([ind.fitness.values[0] for ind in pop])
        idx = np.argmax(fits)
        maxfits.append(np.amax(fits))
        avgfits.append(np.mean(fits))
        bestsols[i,:] = np.array(pop[idx])

    return maxfits, avgfits, bestsols

# PyGAD algorithm. Might clean up afterwards if I have more time (which I won't) but since I don't plan on using pyGAD anymore, i won't waste my time in here.