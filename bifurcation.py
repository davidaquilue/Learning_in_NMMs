'''Bifurcation diagram to check if JR model has been correctly implemented.'''

from singleJR import obtaindynamics, derivatives, unpacking
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time


params = {'A': 3.25, 'B': 22.0, 'v0': 6.0}
params['a'], params['b'], params['e0'] = 100.0, 50.0, 2.5
params['C'] = C = 135
params['C1'], params['C2'], params['C3'], params['C4'] = C, 0.8*C, 0.25*C, 0.25*C  # Dimensionless
params['r'] = 0.56  # mV^(-1)
params['delta'] = 0
params['f'] = 0

p_vecs = np.arange(0, 4000, 1)

def par_f(pbar):
    params['pbar'] = pbar
    y, t = obtaindynamics(params, (0, 40), 0.001, derivatives, unpacking)
    maxim = np.amax(y)
    minim = np.amin(y)
    return (maxim, minim)
if __name__ == '__main__':

    with Pool(30) as p:
        for pbar in p_vecs:
            params['pbar'] = pbar
            listmaxmin = p.map(par_f, pbar*np.ones(100))
            mins = set()
            maxs = set()
            for pair in listmaxmin:
                maxs.add(pair[0])
                mins.add(pair[1])

            
            plt.plot(pbar*np.ones(len(list(maxs))), list(maxs), '.k')
            plt.plot(pbar*np.ones(len(list(mins))), list(mins), '.k')

        plt.show()

