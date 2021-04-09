# Mathematical functions
# Functions included: S, autocorr, findpeaks, psd, normalize, regularity, crosscorrelation, maxcrosscorrelation,
# networkmatrix, findlayer
import numpy as np
from numba import njit
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
  for ii,tau in enumerate(tauvec):
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
  # Obtains the power density
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
  # Both vectors need to have the same length. We take 10 seconds for the integration.
  tauvec = np.arange(0,maxlag,tstep) # We obtain the crosscorrelation up to a lag of 5s (should be enough for alpha rhythms).
  crosscorr = np.zeros_like(tauvec)

  for ii, tau in enumerate(tauvec):
      norm_factor = np.sqrt(np.sum(y1[0:nintegration]**2)*np.sum(y2[ii:nintegration+ii]**2)) # Sqrt of the energies of the signals
      crosscorr[ii] = np.sum(y1[0:nintegration]*y2[ii:nintegration+ii])/norm_factor
  return tauvec, crosscorr

@njit(fastmath = usefastmath)
def maxcrosscorrelation(y1, y2, nintegration = 10000, maxlag = 10, tstep = 0.001):
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

def findlayer(node, tuplenetwork):
  # Simple function that takes as inputs a node i and the architecture of the network
  # With this information it will output the layer at which node i belongs.
  layers = len(tuplenetwork)
  layer = 0
  kk = 0
  for nodes in tuplenetwork:
    if (kk <= node) and (node < kk+nodes):
      return layer
    else:
      kk += nodes
      layer += 1