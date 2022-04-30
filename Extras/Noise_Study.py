# THIS MODULE CONTAINS 2 DIFFERENT STUDIES ABOUT NOISE IN A SINGLE COLUMN JR MODEL

# 1. Study of how dynamics and psds vary for different white noise values.
import numpy as np; from numba import njit
import matplotlib.pyplot as plt
usefastmath = True
from singleJR import Heun_nonoise, obtaindynamics, unpacking as unpacking2
from matfuns import psd, S

params = {'A': 3.25, 'B': 22.0, 'v0': 6.0} 
params['a'], params['b'], params['e0'], params['pbar'], params['delta'], params['f'] = 100.0, 50.0, 2.5, 155.0, 65.0, 8.5
C = 133.5
params['C'], params['C1'], params['C2'], params['C3'], params['C4'] = C, C, 0.8*C, 0.25*C, 0.25*C # All dimensionless

params['r'] = 0.56 #mV^(-1)


tspan = (0,20); params['tspan'] = tspan
tstep = 0.001; params['tstep'] = tstep

def unpacking(params):
    '''Returns the values of the parameters in the dictionary params in a tuple so that we can work with numba'''
    A, B, v0, a, b, e0, pbar = params['A'],params['B'],params['v0'],params['a'], params['b'], params['e0'], params['pbar']
    delta, f, C, C1, C2 = params['delta'], params['f'], params['C'], params['C1'], params['C2']
    C3, C4, r = params['C3'], params['C4'], params['r']
    noise_m = params['noise_m']
    noise_std = params['noise_std']

    return (A, B, v0, a, b, e0, delta, f, C, C1, C2, C3, C4, r, noise_m, noise_std)


@njit(fastmath = usefastmath)
def derivatives(x, t, paramtup):
    '''
    Function that returns the derivatives of the 6-dimensional ODE system.
    Inputs:
    x:          6-element vector: x = (z0, y0, z1, y1, z2, y2). Numpy array
    t:          Value of time
    paramtup:   Tuple with all the parameters inside.
    '''
    #Unpacking of values.
    A, B, v0, a, b, e0, delta, f, C, C1, C2, C3, C4, r, noise_m, noise_std = paramtup
    pbar = np.random.normal(noise_m, noise_std)
    # Assigning variables value for each input.
    z0 = x[0]
    y0 = x[1]
    z1 = x[2]
    y1 = x[3]
    z2 = x[4]
    y2 = x[5]
    # Derivatives of each variable.
    dz0 = A*a*S(y1-y2,e0,r,v0) - 2*a*z0 - a**2*y0
    dy0 = z0
    dz1 = A*a*(pbar + C2*S(C1*y0,e0,r,v0)) - a**2*y1 - 2*a*z1
    dy1 = z1
    dz2 = B*b*(C4*S(C3*y0,e0,r,v0)) - 2*b*z2 - b**2*y2
    dy2 = z2
    return np.array([dz0, dy0, dz1, dy1, dz2, dy2])

# 2. Study of how the integration of the dynamics are affected when the values of the white noise are suddenly changed.
@njit(fastmath = usefastmath)
def derivatives2(x, t, paramtup):
    '''
    Function that returns the derivatives of the 6-dimensional ODE system.
    Inputs:
    x:          6-element vector: x = (z0, y0, z1, y1, z2, y2). Numpy array
    t:          Value of time
    paramtup:   Tuple with all the parameters inside.
    '''
    #Unpacking of values.
    A, B, v0, a, b, e0, pbar, delta, f, C, C1, C2, C3, C4, r = paramtup
    if t > 12.5 and t < 15:
        pbar = np.random.normal(2000,5) # HACER PRUEBAS PARA VER PARA QUE NIVELES NO HAY MALA INTEGRACION
    else:
        pbar = np.random.uniform(120,360)
    # Assigning variables value for each input.
    z0 = x[0]
    y0 = x[1]
    z1 = x[2]
    y1 = x[3]
    z2 = x[4]
    y2 = x[5]
    # Derivatives of each variable.
    dz0 = A*a*S(y1-y2,e0,r,v0) - 2*a*z0 - a**2*y0
    dy0 = z0
    dz1 = A*a*(pbar + C2*S(C1*y0,e0,r,v0)) - a**2*y1 - 2*a*z1 #+ delta*np.sin(2*np.pi*f*t)
    dy1 = z1
    dz2 = B*b*(C4*S(C3*y0,e0,r,v0)) - 2*b*z2 - b**2*y2
    dy2 = z2
    return np.array([dz0, dy0, dz1, dy1, dz2, dy2])


if __name__ == '__main__':
    print('Select study you want to perform (either 1 or 2)')
    study_n = int(input())
    if study_n == 1:
        n_noises = 5
        low_std = 5
        high_std = 100
        noises = np.linspace(0,1500,n_noises)
        fig_ys, axes_ys = plt.subplots(4, n_noises)
        fig_S, axes_S = plt.subplots(2, n_noises)

        for pp,mean in enumerate(noises):
            params['noise_m'] = mean
            params['noise_std'] = low_std
            y, t = obtaindynamics(params, tspan, tstep, derivatives, unpacking)
            ax = axes_ys[0,pp]
            ax.plot(t[-2000:],y[-2000:])
            ax.set(ylim = (-5,50), title = 'Mean noise = %g, low std' %mean)

            ax = axes_S[0, pp]
            ax.plot(t[-2000:],S(y[-2000:], params['e0'], params['r'], params['v0']))
            ax.set(ylim = (0,8), title = 'Mean noise = %g, low std' %mean)

            f, psds = psd(y, tstep)
            ax = axes_ys[1, pp]
            ax.plot(f, psds)
            ax.set(xlim = (0,20))

            params['noise_std'] = high_std
            y, t = obtaindynamics(params, tspan, tstep, derivatives, unpacking)
            ax = axes_ys[2,pp]
            ax.plot(t[-2000:],y[-2000:])
            ax.set(ylim = (-5,50), title = 'Mean noise = %g, high std' %mean)

            ax = axes_S[1, pp]
            ax.plot(t[-2000:],S(y[-2000:], params['e0'], params['r'], params['v0']))
            ax.set(ylim = (0,8), title = 'Mean noise = %g, high std' %mean)

            f, psds = psd(y, tstep)
            ax = axes_ys[3, pp]
            ax.plot(f, psds)
            ax.set(xlim = (0,20))
        plt.tight_layout()

        fig_ys.show()
        input('Next pic?')
        fig_S.show()
        input('finish')
    
    elif study_n == 2:
        y,t = obtaindynamics(params, params['tspan'], params['tstep'], derivatives2, unpacking2)
        firing = S(y, params['e0'], params['r'], params['v0'])
        plt.plot(t,firing)
        print(np.amax(firing)) # We observe a saturation at around 5Hz.
        plt.show()
    
    else:
        print('Please select a correct number of study (1 or 2)')
