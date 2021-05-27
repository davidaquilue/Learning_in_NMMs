''' Some test per no anar repetint tot bro'''
import numpy as np
import matplotlib.pyplot as plt
from signals import build_p_inputs
from matfuns import fastcrosscorrelation as ccross, findlayer, psd, networkmatrix_exc_inh
from networkJR import obtaindynamicsNET
from plotfuns import plotcouplings, plot_corrs, plot_363

# Set up of JR params
params = dict(A=3.25, B=22.0, v0=6.0)
params['a'], params['b'], params['e0'] = 100.0, 50.0, 2.5
params['pbar'], params['delta'], params['f'] = 155.0, 65.0, 8.5

params['C'] = C = 133.5
params['C1'], params['C2'], params['C3'], params['C4'] = C, 0.8*C, 0.25*C, 0.25*C  # Dimensionless

params['r'] = 0.56  # mV^(-1)

params['delta'] = 0  # No periodic driving
params['f'] = 0

# MANUAL TESTS TO SEE IF THE NETWORK IS ABLE TO "LEARN" WHAT WE WANT IT TO LEARN
# NETWORK ARCHITECTURE PARAMETERS
params['tuplenetwork'] = (3, 6, 3)
params['recurrent'] = False
params['forcednodes'] = (0, 1, 2)

Nnodes, matrix_exc, matrix_inh = networkmatrix_exc_inh(params['tuplenetwork'], params['recurrent'], v=0)

params['Nnodes'] = Nnodes
params['matrix_exc'] = matrix_exc
params['matrix_inh'] = matrix_inh
maxval = 0.3*C  # Quan tots estan a 0.3C el valor mig puja força pero sembla que es podra treballar en aquest regim

# Matrix exc és una matriu Nnodes X Nnodes, igual que matrix inh. Columna ii determina a quins nodes el node ii afecta
# fila ii determina quins nodes afecten al node ii
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

params['individual'] = np.append(exc_w[idexes].flatten(), inh_w[idexes].flatten())
params['tstep'] = 0.001
params['tspan'] = (0, 500)

# INPUT SIGNALS: TRAINING AND TESTING SETS
offset = 10
ampnoise = 2
amps = [110, 120]
paircorr = (1, 2)

t = np.linspace(params['tspan'][0], params['tspan'][1], int((params['tspan'][1] - params['tspan'][0])/params['tstep']))
params['signals'] = build_p_inputs(params['tuplenetwork'][0], t, offset, paircorr, ampnoise)

y, t = obtaindynamicsNET(params, params['tspan'], params['tstep'], v=3)
fig1 = plot_363(y, t, 'large', params, params['signals'])
fig3 = plot_corrs(y, params)
print('Correlated pair: %s' % (paircorr, ))
print('Crosscorr 0,1: %f ' % ccross(y[9], y[10], 250000))
print('Crosscorr 0,2: %f ' % ccross(y[9], y[11], 250000))
print('Crosscorr 1,2: %f ' % ccross(y[11], y[10], 250000))
fig2 = plotcouplings(params['individual'], matrix_exc, matrix_inh, params=params, minmaxvals=(0, maxval), bandw=True)
plt.show()

# Other test shit
'''
inputnodes = 3
t = np.arange(0, 80, 0.001)
offset = 80
n = 5
corrpairs = ((0, 1), (0, 2), (1, 2))

tuplenetwork = (3,3,3)
Nnodes = 0
# Okei comprovem que sí que funciona i que per tant es pot utilitzar dintre del
# codi principal de integració de les dinàmiques.
data = build_dataset(10, 3, corrpairs, t, offset=120, shift=False)
p_inputs = data[0]
print(p_inputs.shape)
p_inputs = p_inputs[0]
fig, axes = plt.subplots(3, 1)

for ii in range(inputnodes):
    ax = axes[ii]
    ax.plot(t, p_inputs[ii])
    ax.set(ylim = (80, 250))
plt.show()

dataset = build_dataset(n, inputnodes, corrpairs, t, offset)
print(len(dataset))
print(dataset[0].shape)
fig, axes = plt.subplots(3, 3)

# It is clear that each element of the dataset is the same
print(np.linalg.norm(dataset[1]-dataset[2]))

for jj in range(len(corrpairs)):
    data_set = dataset[jj]
    for ii in range(inputnodes):
        ax = axes[ii, jj]
        ax.plot(t, data_set[12, ii])

    print('cc1: ' + str(fastcrosscorrelation(data_set[12,0], data_set[12,1])))
    print('cc2: ' + str(fastcrosscorrelation(data_set[12,0], data_set[12,2])))
    print('cc3: ' + str(fastcrosscorrelation(data_set[12,2], data_set[12,1])))
plt.show()
'''
