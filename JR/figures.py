""" Final figures for the TFG report"""
import numpy as np
import matplotlib.pyplot as plt
from signals import build_p_inputs, build_dataset, build_p_inputs_shifted, build_p_vector_soft, p_times, p_amplitudes, build_p_vector
from matfuns import fastcrosscorrelation, findlayer, psd
from singleJR import obtaindynamics, derivatives_signal, unpacking_signal
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

# Other plots