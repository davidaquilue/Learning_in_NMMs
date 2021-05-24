''' Some test per no anar repetint tot bro'''
import numpy as np
import matplotlib.pyplot as plt
from signals import build_p_inputs, build_dataset, build_p_inputs_shifted, build_p_vector_soft, p_times, p_amplitudes, build_p_vector
from matfuns import fastcrosscorrelation, findlayer, psd, networkmatrix_exc_inh
from networkJR import obtaindynamicsNET, individual_to_weights
from plotfuns import plotanynet, plotcouplings3x3V2

params = {'A': 3.25, 'B': 22.0, 'v0': 6.0}
params['a'], params['b'], params['e0'] = 100.0, 50.0, 2.5
params['pbar'], params['delta'], params['f'] = 155.0, 65.0, 8.5

params['C'] = C = 133.5
params['C1'], params['C2'], params['C3'], params['C4'] = C, 0.8*C, 0.25*C, 0.25*C  # Dimensionless

params['r'] = 0.56  # mV^(-1)

params['delta'] = 72.09
params['f'] = 8.6

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

exc_w[3, [0, 1]] = [1*maxval, 0.5*maxval]
exc_w[4, [0, 1]] = [0.5*maxval, 1*maxval]
exc_w[5, [0, 2]] = [1*maxval, 0.5*maxval]
exc_w[6, [0, 2]] = [0.5*maxval, 1*maxval]
exc_w[7, [1, 2]] = [1*maxval, 0.5*maxval]
exc_w[8, [1, 2]] = [0.5*maxval, 1*maxval]

inh_w[8, [1,2]] = [0.1*maxval, 0.1*maxval]


# Així doncs es podrà anar modificant les diferents connexions de manera manual

# Individual son les dues matrius juntes posades en un vector fila. Tindré amagat la linea que fa la transformacio.
# TENIR EN COMPTE QUE NOMÉS ES POSEN A L'INDIVIDUAL ELS ELEMENTS NO NULS, SI VOLEM QUE ALGUNA CONNEXIO SIGUI MOLT
# BAIXA, PER AQUESTES PROVES, CALDRÀ POSARLA EN 1E-4 O ALGUNA COSA AIXI, I NO NOMES 0.
params['individual'] = np.append(exc_w[idexes].flatten(), inh_w[idexes].flatten())

params['tstep'] = 0.001
params['tspan'] = (0, 70)

# INPUT SIGNALS: TRAINING AND TESTING SETS
offset = 10
ampnoise = 2
amps = [110, 120]

t = np.linspace(params['tspan'][0], params['tspan'][1], int((params['tspan'][1] - params['tspan'][0])/params['tstep']))
params['signals'] = build_p_inputs(params['tuplenetwork'][0], t, offset, (0, 1), ampnoise)

y, t = obtaindynamicsNET(params, params['tspan'], params['tstep'], v=3)
fig = plotanynet(y, t, 'large', params, True, params['signals'])

fig = plotcouplings3x3V2(params['individual'], matrix_exc, matrix_inh, maxminvals=(0, maxval))
plt.show()

# Other shit
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
