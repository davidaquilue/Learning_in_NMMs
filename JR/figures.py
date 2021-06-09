""" Final figures for the TFG report"""
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import sys
from signals import build_p_inputs, build_dataset, build_p_inputs_shifted, build_p_vector_soft, p_times, p_amplitudes, build_p_vector
from matfuns import fastcrosscorrelation, findlayer, psd
from singleJR import obtaindynamics, derivatives_signal, unpacking_signal, derivatives, unpacking

plt.style.use('./tfg.mplstyle')

"""
from main import params

# Plot de la sigmoide i dels impulse responses de les poblacions:
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
t = np.arange(0, 0.2, 0.001)
he = params['a']*params['A']*t*np.exp(-params['a']*t)
hi = params['b']*params['B']*t*np.exp(-params['b']*t)
axes[0].plot(t, he, label='$h_e(t)$')
axes[0].plot(t, hi, label='$h_i(t)$')
axes[0].set(title='$h_e(t)$ and $h_i(t)$ responses', xlabel='t (s)', ylabel='mV')
axes[0].legend(loc='best')
m = np.arange(-10, 25, 0.001)
S = 2*params['e0']/(1+np.exp(params['r']*(params['v0'] - m)))
axes[1].plot(m, S)
axes[1].set(title='Sigmoid encoding firing rate', xlabel='mV', ylabel='Hz')
plt.show()
"""
# Other plots

# Diferents dinàmiques
params = dict(A=3.25, B=22.0, v0=6.0)
params['a'], params['b'], params['e0'] = 100.0, 50.0, 2.5
params['C'] = C = 133.5
params['C1'], params['C2'], params['C3'], params['C4'] = C, 0.8*C, 0.25*C, 0.25*C  # Dimensionless
params['r'] = 0.56
params['delta'] = 0
params['f'] = 0

pbars = [60, 70, 115, 155]  # low node, high node, spikes, alpha
x00 = np.array([1, 1, 1, 1, 1, 1])    # low node
x01 = np.array([2.6291297, 0.16272113, -1.47501781, -8.43321622,  0.86793527, -2.67579234])
x02 = np.array([10, 0, 10, 0, 0, 0])  # spikes
for nn, pbar in enumerate(pbars):
    if nn == 0:
        params['x0'] = x00
        col = 'b'
    elif nn == 1:
        params['x0'] = x01
        col = 'c'
    elif nn == 2:
        params['x0'] = x02
        col = 'r'
    elif nn == 3:
        params['x0'] = x02
        col = 'g'
    params['pbar'] = pbar
    y, t = obtaindynamics(params, (0, 15), 0.001, derivatives, unpacking)
    plt.plot(t, y, c=col)
    plt.xlim(10, 15)
    plt.xlabel('Time ($s$)')
    plt.ylabel('$y_1 - y_2$ ($mV$)')
plt.savefig('/home/david/Desktop/test.png')
plt.show()
