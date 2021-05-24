''' Some test per no anar repetint tot bro'''
import numpy as np
import matplotlib.pyplot as plt
from signals import build_p_inputs, build_dataset, build_p_inputs_shifted, build_p_vector_soft, p_times, p_amplitudes, build_p_vector
from matfuns import fastcrosscorrelation, findlayer, psd
from singleJR import obtaindynamics, derivatives_signal, unpacking_signal


t = np.arange(0, 40, 0.001)
t_order = p_times((t[0], t[-1]))
amps_order = p_amplitudes(t_order)
p_vec = build_p_vector_soft(t, t_order, amps_order, 250)

#Test per a veure la beta activity
params = {'A': 3.25, 'B': 22.0, 'v0': 6.0}
params['a'], params['b'], params['e0'] = 100.0, 50.0, 2.5
params['pbar'], params['delta'], params['f'] = 155.0, 65.0, 8.5

params['C'] = C = 135
params['C1'], params['C2'], params['C3'], params['C4'] = C, 0.8*C, 0.25*C, 0.25*C  # Dimensionless

params['r'] = 0.56  # mV^(-1)

# No m'interessen les freqüències!
params['delta'] = 0
params['f'] = 0
params['signal'] = p_vec

y, t = obtaindynamics(params, (0, 40), 0.001, derivatives_signal, unpacking_signal)
plt.figure(figsize=(21,10))
plt.subplot(211)
plt.plot(t, p_vec[10000:], 'k')
plt.xlim((t_order[-3]-2, t_order[-3]+2))
plt.subplot(212)
plt.plot(t, y, 'r')
plt.xlim((t_order[-3]-2, t_order[-3]+2))
plt.show()







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
