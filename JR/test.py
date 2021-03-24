# test. Parece que el singleJR funciona todo correcto.
from singleJR import obtaindynamics
import numpy as np
import matplotlib.pyplot as plt
from plotfuns import plot3x3, plotcouplings3x3
from matfuns import networkmatrix

params = {'A': 3.25, 'B': 22.0, 'v0': 6.0} 
params['a'], params['b'], params['e0'], params['pbar'], params['delta'], params['f'] = 100.0, 50.0, 2.5, 155.0, 65.0, 8.5
C = 133.5
params['C'], params['C1'], params['C2'], params['C3'], params['C4'] = C, C, 0.8*C, 0.25*C, 0.25*C # All dimensionless

params['r'] = 0.56 #mV^(-1)

params['delta'] = 72.09
params['f'] = 8.6
params['stimulation_mode'] = 1

from networkJR import obtaindynamicsNET
# Now we define the network architecture. Always feedforward and we can decide wether we want recurrencies or not
params['tuplenetwork'] = (3,3,3)
params['recurrent'] = True
params['GAactive'] = 0
params['forcednode'] = 1
Nnodes, matrix =  networkmatrix(params['tuplenetwork'], params['recurrent'])
params['Nnodes'] = Nnodes
params['matrix'] = matrix
individual = 100*C*np.random.random(2*np.count_nonzero(matrix))
params['individual'] = individual

y,t = obtaindynamicsNET(params, (0,50), 0.001, v=2)

fig = plot3x3(y,t,'small',0.001)
## Okay de momento parece que está la mayoria de cosas correctas