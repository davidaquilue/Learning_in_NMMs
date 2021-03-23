# Libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from functions import S, psd, normalize, regularity, networkmatrix, HeunNET, plot3x3, plotanynet, crosscorrelation, plotcouplings3x3
from matplotlib import cm
import time
from deap import base, creator, tools
from tqdm import tqdm
import random
usefastmath = True # We can choose whether we want to use numba or not. It will speed up our calculations

# General plotting settings
labelfontsize = 15
labelticksize = 15
titlefontsize = 20
suptitlefontsize = 26

########################################### NMMs PARAMETERS ###########################################################
# Parameters in each line have different units

params = {'A': 3.25, 'B': 22.0, 'v0': 6.0} # All in mV
params['a'], params['b'], params['e0'], params['pbar'], params['delta'], params['f'] = 100.0, 50.0, 2.5, 155.0, 65.0, 8.5 # All in Hz
C = 133.5
params['C'], params['C1'], params['C2'], params['C3'], params['C4'] = C, C, 0.8*C, 0.25*C, 0.25*C # All dimensionless
params['r'] = 0.56 #mV^(-1)

########################################### NETWORK SIMULATION PARAMETERS ##############################################
params['Net'] = True      # Code written admits False to work with single column simulations
tuplenetwork = (3,3,3)    # Topology of the network. (Nodes_layer1, Nodes_layer2, ...., Nodes_layerK)
recurrent = False         # If there is a feedback pyramidal loop in each node.

# A function that returns the total number of nodes (Nnodes) in the network and a connection matrix. 
# ith row of matrix: Connections coming into node i
# ith column of matrix: Connections going out of node i
# Element (i,j) in the matrix: 1 if node i receives input from node j. 0 if it does not. 
Nnodes, matrix = networkmatrix(tuplenetwork, recurrent)
tuplemat = tuple(map(tuple, matrix)) # In order to work with numba we transform the matrix to a tupled array.
params['matrix'] = tuplemat
params['Nnodes'] = Nnodes
params['tuplenetwork'] = tuplenetwork

# Time simulation constants.
tspan = (0,30)
tstep = 0.001

##  change delta and f in order to have periodic, quasi periodic or chaotic behaviour in input nodes.
## It is interesting to have in mind the colormaps from the project to select the values.
params['delta'] = 72.09
params['f'] = 8.6

######################################### GENETIC ALGORITHM PARAMETERS #############################################
num_generations = 5               # Total number of generations that the GA will be run
popsize = 30                      # Population size for each generation
mutindprob = 0.2                  # Probability of chosing an individiual for mutation
coprob = 0.5                      # Probability of choosing a pair of inds for crossover
fitness_func = fitness_function3  # fitness_func: The name of the fitness function
generange = (0,100*C)             # The (min,max) values of the individuals' genes.
indsize = 18                      # The individual size. The amount of genes.
mutprob = 0.05                    # The probability of mutation of a certain gene, when the individual has been chosen to mutate
tmntsize = 3                      # In the case of tournament selection, the size of the tournament.
parallel = True                   # GA fitness evaluation parallelization.

###################################### FUNCTIONS USED TO OBTAIN THE DYNAMICS #############################################

def unpackingGA(params):
    # We define an unpacking function that everytime we input the dictionary (params) it returns a tuple with all the values.
    # This will be useful to modify the values of the parameters but
    
    A, B, v0, a, b, e0, pbar = params['A'],params['B'],params['v0'],params['a'], params['b'], params['e0'], params['pbar']
    delta, f, C, C1, C2 = params['delta'], params['f'], params['C'], params['C1'], params['C2']
    C3, C4, r = params['C3'], params['C4'], params['r']
    Net = params['Net']
    if Net:
        #alpha = params['alpha']
        #beta = params['beta']
        matrix = params['matrix']
        Nnodes = params['Nnodes']
        tuplenetwork = params['tuplenetwork']
        forcednode = params['forcednode']
        individual = params['individual']
        return (A, B, v0, a, b, e0 , pbar, delta, f, C, C1, C2, C3, C4, r, individual, matrix, Nnodes, tuplenetwork, forcednode)
    else:  
        return (A, B, v0, a, b, e0 , pbar, delta, f, C, C1, C2, C3, C4, r )

@njit(fastmath = usefastmath)
def derivativesNETGA(inp, t, paramtup):
    # We repeat the function where the full ODE system is represented. 
    # Auxiliary variables zi have been used in order to reduce the system to a set of first order ODEs.
    # inp has to be a (N, 6) matrix.
    # N is the number of nodes. Only 1 row because the input corresponds to the values of the variables at each time step.

    #Unpacking of values.
    A, B, v0, a, b, e0, pbar, delta, f, C, C1, C2, C3, C4, r, individual, matrix, Nnodes, tuplenetwork, forcednode = paramtup

    # Now the input will be a matrix where each row i corresponds to the variables (z0,y0,z1,y1,z2,y2) of each node i.
    # Initialization of the output. It will be the same dimension as the input.
    dz = np.zeros_like(inp)
    # Now we obtain the derivatives of every variable for every node.
    for nn in range(Nnodes):
      x = inp[nn] # This will extract the row corresponding to each node.
      z0 = x[0]
      y0 = x[1]
      z1 = x[2]
      y1 = x[3]
      z2 = x[4]
      y2 = x[5]
      
      # Coupled intensities, we obtain them from a function.
      pa, pb = couplingvalGA(inp, matrix[nn], individual, C3, e0, r, v0, nn, tuplenetwork, Nnodes)
      pbar = np.random.uniform(120,320)

      # If the node is the forced node, there is periodic driving, if not, the stimulation comes from constant and other nodes.
      delta1 = 0
      if nn == forcednode:
        delta1 = delta
      # Derivatives of each variable.
      dz0 = A*a*S(y1-y2,e0,r,v0) - 2*a*z0 - a**2*y0
      dy0 = z0
      dz1 = A*a*(pbar + C2*S(C1*y0,e0,r,v0) + delta1*np.sin(2*np.pi*f*t)) - a**2*y1 - 2*a*z1 + pa
      dy1 = z1
      dz2 = B*b*(C4*S(C3*y0,e0,r,v0)) - 2*b*z2 - b**2*y2 + pb
      dy2 = z2
      dz[nn] = np.array([dz0, dy0, dz1, dy1, dz2, dy2])

    return dz

@njit(fastmath = usefastmath)
def couplingvalGA(inp, connections, individual, C3, e0, r, v0, nn, tuplenetwork, Nnodes):
  # This function obtains the effects of the coupling for each node. 
  Suma = 0
  Sumb = 0
  # We obtain the contribution of each node.
  for node,value in enumerate(connections):
    if value == 1:  #by this we have acces to the nodes to which the current node is linked to
      alpha = individual[node]
      beta = individual[-(node+1)]
      Suma = Suma + alpha*S(inp[node,3]-inp[node,5], e0, r, v0)
      Sumb = Sumb + beta*S(C3*inp[node, 1], e0, r, v0)
  # Since the nodes in a layer have all the same connections:
  if nn < tuplenetwork[0]: # First layer.
    Ni = 1
    Nj = 3

  ### ESTA ES UNA DE LAS COSAS QUE HABRÁ QUE GENERALIZAR PARA ARQUITECTURAS MÁS COMPLEJAS.
  # Como solo afecta en un factor de division, de momento no me voy a matar en esto.
  
  else:
    Ni = 3
    Nj = 3
  pa = Suma/np.sqrt(Ni*Nj)
  pb = Sumb/np.sqrt(Ni*Nj)
  return pa,pb

def obtaindynamicsNETGA(params, tspan, tstep):
    # This function will allow us to obtain the dynamics of the Network for a determined set of parameters.
    # Used to simplify the main body of the code.
    
    funparams = unpackingGA(params)
    Nnodes = params['Nnodes']

    x0=10*np.random.normal(size=(Nnodes,6)) # starting from random normally distributed IC
    # S'HAURIA D'ENVIAR TAMBE UN VECTOR AMB RANDOM PHASES PER TAL QUE ELS INPUTS TINGUIN
    # UNA FASE DIFERENT!

    x1,t1 = HeunNET(x0, tspan, tstep, derivativesNETGA, funparams) 
    
    # The outputted matrix is of the type (Nnodes, timevalues, Nvariables)
    # First argument is the node. Second is the time. Third is the variable (y0,z0,...)
    
    # We want to get rid of the transitory in the ode.
    x, t = x1[:,-35000:,:], t1[-35000:] # We get only the last 10 seconds. Enough to observe characteristics

    y1 = x[:,:,3]
    y2 = x[:,:,5]
    yout=y1-y2
    # We output de difference between y1-y2 for each node for each timestep integrated.
    return yout,t

###################################### FITNESS FUNCTION AND ITS COMPLEMENTS #############################################
@njit(fastmath = True)
def toint(bin):
  # We pass the numpy array and we convert it to integer
  num = 0
  for ii in range(bin.size):
    num += bin[-(ii+1)]*2**(ii)
  return num
  
def fitness_function3(individual):#, individualidx):
  # DEAP we will start working with lists. We could also make individiuals inherit from np arrays but later on.
  individual = np.array(individual)
  params['individual'] = individual

  # Fem tres passades, per cada passada canviem quin node alterem amb delta, els altres dos constants. 
  # Veiem si l'output regular es el desitjat
  nodes0layer = tuplenetwork[0]
  nodeslastlayer = tuplenetwork[-1]
  fit = 0
  regthreshold = 0.95

  # A cada iteració d'aquest for forcem un node de la primera capa diferent
  itermax = 5
  # Repetim el proces 10 vegades, per assegurarnos que les regularitats son fiables.
  for iter in range(itermax):
    regs = np.zeros((3,3))
    arraybits = np.zeros((3,3))

    for ii in range(nodes0layer):
      params['forcednode'] = ii
      y , _ = obtaindynamicsNETGA(params, tspan, tstep)

      # Per cada node de la primera capa forçat mirem com son les regularitats de la ultima capa
      for ll,yy in enumerate(y[-nodeslastlayer:]):
    
        regs[ii,ll] = regularity(yy)
        if 0.7 < regs[ii,ll] < regthreshold:
          fit -= 10 # Se puede mirar tambien si 5 o asi, a lo mejor es muy restrictiva
    
    # We obtain an array of 1s and 0s where each row will be a different code
    arraybits[np.where(regs >= regthreshold)] = 1
    arraybits[np.where(regs < regthreshold)] = 0
    # Different codes
    a = toint(arraybits[0]); b = toint(arraybits[1]); c = toint(arraybits[2])
    # And a penalizing term for repeated codes
    pen = 0
    if a==b or a==c or b==c:
      pen = 50
    fit += (a-b)**2 + (a-c)**2 + (b-c)**2 - pen
  # We add a small term. We want the regularities to be as far away from 0.9 as possible.
  if fit <= 0:
    return 0,
  else:
    return fit,


###################################### DEAP INITIALIZATION OF FUNCTIONS ##########################################

# The GA class has a method called cal_pop_fitness that calculates the fitness of all the elements in the
# population. This function can be optimized for multiple threads using the multiprocessing pool
creator.create('FitnessMax', base.Fitness, weights = (1.0,))
creator.create('Individual', list, fitness = creator.FitnessMax)

toolbox = base.Toolbox()
# We generate the genes of each individual:
toolbox.register('genes', np.random.uniform, generange[0], generange[1])
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.genes, indsize)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('evaluate', fitness_func) # The evaluation will be performed calling the alias evaluate. It is important to not fix its argument here. 
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutGaussian, mu = (generange[1]- generange[0])/2, sigma = (generange[1]- generange[0])/20,indpb = mutprob) # Real number only mutation. Let's see how it works
toolbox.register('select', tools.selTournament, tournsize = 3)

def mainGA(toolbox, num_generations, popsize, mutindprob, coprob, indsize):
  # The function that will run the main algorithm of the GA. It will have three outputs:

  # bestsols: An array containing, in each row, the best individual of each generation
  # maxfits: A list containing the largest fitness in each generation
  # avgfits: A list containing the average fitness in each generation.

  bestsols = np.zeros((num_generations, indsize)) # Here we will store the best solution of the generation.
  maxfits = []
  avgfits = []
  pop = toolbox.population(n = popsize) # pop will be a list of popsize individuals
  fitnesses = list(toolbox.map(toolbox.evaluate, pop))
  for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit # We store the results in the value of the fitness attribute of the individual class.
  
  
  for i in tqdm(range(num_generations)):
    # Selection of the best individuals from the previous population
    offspring = toolbox.select(pop,len(pop))
    offspring = list(map(toolbox.clone, offspring)) 

    # Mutation and crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
      if random.random() < coprob:
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values
    
    for mutant in offspring:
      if random.random() < mutindprob:
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


#################################################### EXECUTION OF GA #######################################################
from multiprocessing import Pool

if __name__ == '__main__':

  if parallel:
    # Initialization of the process pool and setting up the map function
    pool = Pool(processes = 40)
    toolbox.register('map', pool.map)

  # First run for numba so that it can precompile the code and work swiftly during the GA
  print('Starting first Run')
  params['individiual'] = 1*C*np.random.rand(2*Nnodes)
  params['forcednode'] = np.random.randint(0,3)
  y,t = obtaindynamicsNETGA(params, tspan, tstep)
  print('First run completed')

  ################################### NOW GA RUNNING ############################################################
  print('Starting GA...')
  print()
  maxfits, avgfits, bestsols = mainGA(toolbox, num_generations, popsize, mutindprob, coprob, indsize)

  # Plotting fitnesses
  generations = np.linspace(1, num_generations,num_generations)
  plt.plot(generations, maxfits)
  plt.plot(generations, avgfits)
  plt.show()

  # Plotting how the coefficients of the solution look
  npmaxfits = np.array(maxfits)
  idx = np.argmax(npmaxfits)
  solution = bestsols[idx]
  plotcouplings3x3(solution)

