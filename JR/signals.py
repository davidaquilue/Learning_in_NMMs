''' Some of the different "internal logic" of the input signals '''

import random
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from plotfuns import plot_sigsingleJR
from singleJR import unpacking_signal, derivatives_signal, obtaindynamics
from filefuns import get_num


def p_times(tspan, lens=[5, 10, 15]):
    '''Returns a list of the times at which there is a change in value.'''
    time = tspan[0] + random.choice(lens)
    time_order = [time]
    while time_order[-1] < tspan[1]:
        time += random.choice(lens)
        time_order.append(time)
    if time_order[-1] > tspan[0]:
        time_order[-1] = tspan[1]
    return time_order

def p_amplitudes(time_order, amps=[-190, -205, -220], sils=True,
                 rand=False, randrange=(20, 120), randst=10):
    '''Returns a list stating which amplitudes correspond to the time_order
    list. One can select if always full random or if nota-silenci-nota...'''
    if rand:
        amps = np.random.uniform(randrange[0], randrange[1], randst)
    if sils:
        amps_order = []
        while len(amps_order) < len(time_order):
            amps_order.append(random.choice(amps))
            amps_order.append(0)
        if len(amps_order) > len(time_order):
            amps_order = amps_order[0:-1]
    else:
        amps_order = [random.choice(amps) for time in time_order]
    return amps_order

def build_p_vector(t, time_order, amps_order, offset, ampnoise=10):
    '''Builds the p(t) signal vector/array p(t) = offset + amp(t) + random(noise)'''
    p_vector = np.zeros_like(t)
    idx = 0
    for ii, tt in enumerate(t):
        p_vector[ii] = offset + amps_order[idx] + np.random.normal(0, ampnoise)
        if tt > time_order[idx]:
            idx += 1
    return p_vector

def build_p_vector_soft(t, time_order, amps_order, offset, ampnoise=10):
    '''Same as build p_vector but i want to soften the discontinuity'''
    p_vector = np.zeros_like(t)
    idx = 0
    ramp_steps = 1000
    for ii, tt in enumerate(t):
        p_vector[ii] = offset + amps_order[idx] + np.random.normal(0, ampnoise)
    
        if tt > time_order[idx]:
            idx += 1
            p_vector[ii-ramp_steps+1:ii+1] = offset + amps_order[idx-1] + np.linspace(0, ramp_steps-1, ramp_steps)*(amps_order[idx]-amps_order[idx-1])/ramp_steps + np.random.normal(0, ampnoise, ramp_steps)
    
    return p_vector


def build_p_inputs(inputnodes, t, offset, corrnodes):
    '''Builds a (inputnodes, t.size) array containing the different input
    vectors p that go into a node.
    All corrnodes will be generated with the same time_order'''
    corr_times = p_times((t[0], t[-1]))
    p_inputs = np.zeros((inputnodes, t.size))
    for node in range(inputnodes):
        if node in corrnodes:
            p_inputs[node, :] = build_p_vector_soft(t, corr_times,
                                                    p_amplitudes(corr_times), offset)
        else:
            times = p_times((t[0], t[-1]))
            p_inputs[node, :] = build_p_vector_soft(t, times, p_amplitudes(times),
                                                    offset)

    return p_inputs


def build_p_inputs_shifted(inputnodes, t, offset, corrnodes, tshift, amp=-205):
    '''Adds a time shift to the input vectors. tshift in seconds '''
    idx_shift = int(tshift/(t[1]-t[0]))
    p_inputs = np.zeros((inputnodes, t.size))
    for node in range(inputnodes):
        time_order = p_times((t[0], t[-1]))
        amps_order = p_amplitudes(time_order, [0, amp], rand=False)
        p_inputs[node, :] = build_p_vector_soft(t, time_order, amps_order, offset)
    
    # Y despues cambiamos el que toque haciendole el shift
    p_inputs[corrnodes[1]] = np.roll(p_inputs[corrnodes[0]], idx_shift)
    return p_inputs

def build_dataset(n, inputnodes, corrpairs, t, offset=0, shift=False, tshift=5):
    '''Returns a list containing the different sets of inputs for each 
    pair of correlated input signals. That is, a len(corrpairs) list where
    each element is an (n, inputnodes, t.size) array, containing n different
    combinations of input signals.'''
    aux_data = np.zeros((n, inputnodes, t.size))
    dataset = []
    for jj in range(len(corrpairs)):
        for nn in range(n):
            if shift:
                p_inputs = build_p_inputs_shifted(inputnodes, t, offset,
                                                  corrpairs[jj], tshift)
            else:
                p_inputs = build_p_inputs(inputnodes, t, offset, corrpairs[jj])
            aux_data[nn] = p_inputs
        dataset.append(np.copy(aux_data))

    return dataset

# These will not be taken into account anymore for the development of my work

def sig_Harmonics(t, f, inputnodes):
    '''Returns three signals, two of them being harmonics of the first one.
    They are squared signals.'''
    signals = np.zeros((inputnodes, t.size))
    fs = [f, 2*f, 3*f]
    for ii in range(inputnodes):
        signals[ii] = signal.square(2*np.pi*fs[ii]*t)

    return signals

def sig_random(t, inputnodes, T_TO_RAND=500, modulate=False, f=0):
    '''Returns inputnodes signals where after a certain T_TO_RAND time, a 
    coin is flipped and depending on the outcome there is a switch in the value
    of the signal. Three valued signals -1,0,1.
    Only +-1 possible switchings.'''
    signals = np.zeros((inputnodes, t.size))
    aux = 0
    val = 0
    for ii in range(inputnodes):
        for tt in range(t.size):
            signals[ii, tt] = val
            aux += 1
            if aux == T_TO_RAND:  # Now toss a coin
                aux = 0
                coin = np.random.binomial(1, 0.5)
                if coin == 1:  # Change val for one of the results
                    if val == 0:
                        coin2 = np.random.binomial(1, 0.5)
                        if coin2 == 1:
                            val = 1
                        else:
                            val = -1
                    elif val == 1:
                        val = 0
                    else:
                        val = 0
    if modulate and f != 0:
        f = int(input('Select a frequency: '))
        fs = f*np.linspace(1, inputnodes, inputnodes)  # Harmonics
        for ii in range(inputnodes):
            signals[ii] = 0.1*np.cos(2*np.pi*fs[ii]*t) + signals[ii]
    return signals

def check_time(note_len, tot_time, maxt):
    '''Checks if the note_len fits in the remaining time of the sequence'''
    if note_len + tot_time >= maxt:
        note_len = maxt-tot_time

    return note_len


def sig_musical(t, inputnodes, ampcos=1, cosoff=0, mode='Cosine', maxt=10,
                lens=[0.5, 1, 2], freqs=[2, 4, 8, 10, 12]):
    '''This function returns inputnodes different signals that randomly choose
    different durations and frequencies of notes and repeats them after every
    maxt seconds, if the t vector is in seconds.

    One can choose between the "Cosine" and "Intensity" modes.
    lens are the possible lengths that a silence or note can take. freqs are
    the possible frequencies that a note can take.'''

    signals = np.zeros((inputnodes, t.size))

    for ii in range(inputnodes):
        tot_time = 0
        times_order = []
        freqs_order = []
        while tot_time < maxt-1:
            note_len = check_time(random.choice(lens), tot_time, maxt-1)
            freq = random.choice(freqs) 
            tot_time += note_len
            times_order.append(tot_time)
            freqs_order.append(freq)

            sil_len = check_time(random.choice(lens), tot_time, maxt-1)
            tot_time += sil_len
            times_order.append(tot_time)
            freqs_order.append(0)

        times_order.append(maxt)  # Ensures silence in last second
        freqs_order.append(0)

        # This part of the code creates lists of the times where a silence or
        # a note comes into play and at which frequency. It is then used to
        # obtain a vector describing this dynamics.
        idx = 0
        corrector = 0
        for jj, tt in enumerate(t):  # Iterate over all the time steps
            tt_mod = tt - corrector  # Allows for sequence repetition

            if mode == 'Intensity':
                signals[ii, jj] = freqs_order[idx]

            elif mode == 'Cosine':
                if freqs_order[idx] == 0:
                    signals[ii, jj] = 0
                else:
                    signals[ii, jj] = cosoff + ampcos*np.cos(2*np.pi*freqs_order[idx]*tt)

            if tt_mod > times_order[idx]:  # Change of frequency
                idx += 1
            if idx == len(times_order):  # Repetition
                corrector += maxt
                idx = 0

    return signals

def add_noise(signals, noiseamp):
    for ii in range(signals.shape[0]):
        signals[ii] = signals[ii] + noiseamp*np.random.normal(size=signals[ii].size)
    return signals
