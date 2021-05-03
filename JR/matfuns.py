''' Collection of mathematical and auxiliary functions. 

Functions included: S, autocorr, findpeaks, psd, normalize, regularity, crosscorrelation, maxcrosscorrelation,
fastcrosscorrelation, networkmatrix, findlayer, networkmatrix_exc_inh, creating_signals'''

from numba import njit
from scipy import signal
import numpy as np
usefastmath = True

@njit(fastmath = usefastmath)
def S(m,e0,r,v0):
    '''Transformation corresponding to the transformation in the somas of the net average PSP into an average action potential density.'''
    return 2*e0/(1+np.exp(r*(v0 - m)))

@njit(fastmath = usefastmath, parallel = False)
def autocorr(x, nintegration = 20000, maxlag = 3, tstep = 0.001): 
  ''' 
  Calculate the autocorrelation of a signal. nintegration is the number of points used to obtain the integration.
  One has to take into account that nintegration+maxlag/tstep have to be less than the size of x.
  '''
  tauvec = np.arange(0,maxlag,tstep) # We obtain the autocorr up to a lag of 3s (should be enough for alpha rhythms).
  acorr = np.zeros_like(tauvec)
  for ii, _ in enumerate(tauvec):
      acorr[ii] = np.sum(x[0:nintegration]*x[ii:nintegration+ii])
  return tauvec, acorr/np.partition(np.abs(acorr),-1)[-1]

@njit(fastmath = usefastmath)
def findpeaks(x):
  # Finds relative maxima. We are going to use it to find regularity. Super simple code but we need something fast.
  peaks = np.array([0.0])
  for ii,xx in enumerate(x[2:-1]):
    if xx-x[ii-1] < 0 and x[ii-1]-x[ii-2]>0:
      peaks = np.append(peaks, x[ii-1])
  return peaks

def psd(y,tstep):
  '''Returns power density as a function of frequency of signal y.'''
  power = np.abs(np.fft.fft(y-np.average(y)))**2
  freqs = np.fft.fftfreq(y.size,tstep)
  idx = np.argsort(freqs)
  f = freqs[idx]
  PSD = power[idx]
  return f, PSD

@njit(fastmath = usefastmath)
def normalize(y):
  mean = np.mean(y)
  std = np.std(y)
  return (y-mean)/std

def regularity(y):
  _, corr = autocorr(y)   # Autocorrelation of signal y
  peaks = findpeaks(corr) # We find all relative maxima. We will be interested in the second highest.
  # Sometimes the function does not find more than one peak and we get an error
  if peaks.size < 2:
    return 0
  else:
    return np.partition(peaks, -2)[-2]

@njit(fastmath = usefastmath)
def crosscorrelation(y1,y2,nintegration = 10000, maxlag = 5, tstep = 0.001):
  '''Obtains the crosscorrelation of signals y1, y2 as a function of the lag.'''
  # Both vectors need to have the same length. We take 10 seconds for the integration.
  tauvec = np.arange(0,maxlag,tstep) # We obtain the crosscorrelation up to a lag of 5s (should be enough for alpha rhythms).
  crosscorr = np.zeros_like(tauvec)

  for ii, _ in enumerate(tauvec):
      norm_factor = np.sqrt(np.sum(y1[0:nintegration]**2)*np.sum(y2[ii:nintegration+ii]**2)) # Sqrt of the energies of the signals
      crosscorr[ii] = np.sum(y1[0:nintegration]*y2[ii:nintegration+ii])/norm_factor
  return tauvec, crosscorr  

@njit(fastmath = usefastmath)
def fastcrosscorrelation(y1, y2, nintegration = 10000):
  '''Obtains the 0-lag crosscorrelation between two functions. Interesting for synchronization'''
  yy1 = normalize(y1)
  yy2 = normalize(y2)  # In order for the crosscorrelation to be between -1 and 1
  norm_factor = np.sqrt(np.sum(yy1[0:nintegration]**2)*np.sum(yy2[0:nintegration]**2))
  crossc = np.sum(yy1[0:nintegration]*yy2[0:nintegration])/norm_factor
  return np.abs(crossc) # We return the abs for now, this does not define the correct mathematical function at the moment

@njit(fastmath = usefastmath)
def maxcrosscorrelation(y1, y2, nintegration = 10000, maxlag = 10, tstep = 0.001):
  '''Returns the maximum of the correlation after exploring them up to a lag of maxlag'''
  _, crossc = crosscorrelation(y1, y2, nintegration, maxlag, tstep)
  maxcrossc = np.amax(crossc)
  # This next part of code is used to account for numerial imprecision in the multiplications and normalizations. 
  if maxcrossc > 1.2:
    print('Correlation value not valid')
    return 0
  else:
    if maxcrossc > 1:
      return 1
    else:
      return maxcrossc
    
@njit(fastmath = usefastmath)
def networkmatrix(tuplenetwork, recurrent):
  '''
  Builds a matrix of connections. Each element j in row i determines if node j excites/inhibits node i.
  tuplenetwork: Tuple like (nodeslayer1, nodeslayer2, ..., nodeslayerN)
  recurrent:    Boolean stating if we want feedback loop in the cortical columns (True), or not (False)

  '''
  netvec = np.array(tuplenetwork)
  
  Nnodes = np.sum(netvec) # Total number of nodes
  Nlayers = netvec.size
  matrix = np.zeros((Nnodes, Nnodes))
  ii = netvec[0] # Because first layer has no connection, we have to start lower in the matrix.
  jj = 0
  for layer in range(Nlayers-1):
      nout = netvec[layer]
      nin = netvec[layer+1]

      matrix[ii:ii+nin, jj:jj+nout] = np.ones((nin,nout))
      
      ii = ii + nin
      jj = jj + nout
  if recurrent:
      matrix = matrix + np.identity(Nnodes)
  return Nnodes, matrix

@njit(fastmath = True)
def networkmatrix_exc_inh(tuplenetwork, recurrent, v):
  '''Returns the connectivity matrices of excitation and inhibition, depending on the version. Kinda hard to visualize.'''
  netvec = np.array(tuplenetwork)
  
  Nnodes = np.sum(netvec) # Total number of nodes
  Nlayers = netvec.size
  matrix = np.zeros((Nnodes, Nnodes))
  ii = netvec[0] # Because first layer has no connection, we have to start lower in the matrix.
  jj = 0
  for layer in range(Nlayers-1):
      nout = netvec[layer]
      nin = netvec[layer+1]

      matrix[ii:ii+nin, jj:jj+nout] = np.ones((nin,nout))
      
      ii = ii + nin
      jj = jj + nout
  if recurrent:
      matrix = matrix + np.identity(Nnodes)
  # v = 0. Original matrices
  matrix_exc = np.copy(matrix)
  matrix_inh = np.copy(matrix)

  # v = 1 corresponds to the excitation test. Making pyramidal columns able to excite further than the following layer. This is done by building a triangular excitatory matrix
  # However we don't want intralayer excitation yet, that's why it is not lower triangular.
  if v == 1:
    jin = 0
    for layer in range(Nlayers-1):
      jend = jin + netvec[layer]
      matrix_exc[jend:, jin:jend] = 1
      jin = jend
  # v = 2 corresponds to the memory test. Pyramidal columns are able to provide excitatory feedback to the columns in the previous layers.
  # I wouldn't say that there is feedback to the input columns... Feedback to the directly before, not more
  elif v == 2 or v == 3:
    iin = netvec[0]
    for layer, nodes_layer in enumerate(netvec):
      if layer == 0:
        continue
      else:
        iend = iin + nodes_layer
        jin = iend
        jend = jin + netvec[layer+1]
        matrix_exc[iin:iend, jin:jend] = 1
  # v = 3, same as before but now inhibitory feedback connections aswell.
        if v == 3:
          matrix_inh[iin:iend, jin:jend] = 1
        iin = iend

  return Nnodes, matrix_exc, matrix_inh

def findlayer(node, tuplenetwork):
  '''Finds the layer to which the node corresponds.'''
  layer = 0
  kk = 0
  for nodes in tuplenetwork:
    if (kk <= node) and (node < kk+nodes):
      return layer
    else:
      kk += nodes
      layer += 1

# Maybe add some small noise? 
def creating_signals(t, amp, dc, freq, pair_comp):
    '''
    Returns three squared signals in a matrix, with dc component = dc, max amplitude = dc+amp and two rows that are complementary

    Inputs:
    t:          Time vector
    amp:        Amplitude that will be added to the dc value when the square signal is at +1
    dc:         Vertical offset of the squared signal.
    freq:       In hertz, frequency of the squared signal
    pair_comp:  2-uple defining which of the two rows will be complementary

    Outputs:
    signals:    (3, t.size) array containing the three signals in each row.  
    '''

    # Each signal will have a random phase and a random duty period.
    phase1 = np.random.random(); duty1 = np.random.random()
    phase2 = np.random.random(); duty2 = np.random.random()
    # The not complementary is:
    for ii in range(3):
        if ii not in pair_comp: non_comp = ii
    signals = np.zeros(shape = (3, t.size))
    signals[pair_comp[0]] = dc + amp*(1+signal.square(2*np.pi*(freq*t + phase1), duty1))/2
    signals[pair_comp[1]] = dc + amp*np.abs(1-(1+signal.square(2*np.pi*(freq*t + phase1), duty1))/2)
    signals[non_comp] = dc + amp*(1+signal.square(2*np.pi*(freq*t + phase2), duty2))/2


    return signals