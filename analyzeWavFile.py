# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 15:09:09 2016

@author: jacsk
"""
import os
import numpy as np

import matplotlib.pylab as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm

from python_speech_features import mfcc
from python_speech_features import logfbank

import CreateFeature
import CreateUtils

plt.close("all")

fftPower = 10

rawDataFolder = CreateUtils.getRawDataFolder()

# fileName = 'algebra00004.wav'
# fileName = 'cube00001.wav'
# fileName = 'cartoafit00001.wav'
# fileName = 'carneighborhood00010.wav'
# fileName = 'kitchentable00001.wav'
# fileName = 'afittest00000.wav'
# fileName = 'carsound00000.wav'
# fileName = 'biketest00001.wav'
fileName = 'bikeneighborhood00029.wav'

rawDataFile = os.path.join(rawDataFolder, fileName)

(actualDataY, samplingRate) = CreateFeature.readDataFile(rawDataFile)
actualDataX = np.arange(actualDataY.shape[-1]) * 1.0 / samplingRate
timeStep = 1.0 / samplingRate

plotAmplitude = False
plotCorrelation = False
plotFFT = False
plotSpecgram = True
plotCovariance = False
mfccView = False

if plotAmplitude:
    print("Plot out actual amplitudes")
    plt.figure(1)
    plt.plot(actualDataX, actualDataY)
    plt.title('Signal Over time')
    plt.xlabel('time (s)')
    plt.ylabel('signal amplitude')

if plotCorrelation:
    print("Plot Correlation")
    plt.figure(2)
    secondsToOverlap = 1
    samplesToUse = int(secondsToOverlap * samplingRate)
    tempDataSet = actualDataY[:samplesToUse]
    lags = np.arange(-(samplesToUse - 1), (samplesToUse - 1) + 1) * timeStep
    # correlation = scipy.signal.correlate2d(tempDataSet,tempDataSet,'full','wrap')
    correlation = np.correlate(tempDataSet, tempDataSet, 'full')
    plt.plot(lags, correlation)
    plt.title('Correlation')
    plt.xlabel('time shift(s)')
    plt.ylabel('magnitude of correlation')

if plotFFT:
    print("Performing DFFT")
    sp = np.fft.fft(actualDataY)
    timeStep = (actualDataX[1] - actualDataX[0])
    freq = np.fft.fftfreq(actualDataX.shape[-1], timeStep)
    fftData = np.absolute(sp)
    # print("Filtering")
    # filter to positive frequencies
    freq = freq[:len(freq) / 2]
    fftData = fftData[:len(fftData) / 2]
    plt.figure(3)
    # plt.plot(freq,sp.real,freq,sp.imag)
    plt.plot(freq, fftData)
    plt.title('FFT of entire signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

Pxx = None
freqs = None
bins = None
if plotSpecgram:
    print("Computing specgram with {0} point FFT".format(fftPower))
    plt.figure(4)
    Pxx, freqs, bins, im = plt.specgram(actualDataY, NFFT=2 ** fftPower, Fs=1 / timeStep, noverlap=2 ** (fftPower - 1),
                                        cmap=cm.get_cmap("gist_heat"))

    # plt.pcolormesh(bins,freqs,10*np.log10(Pxx),cmap=cm.get_cmap("gist_heat"))
    # plt.imshow(10*np.log10(Pxx),interpolation = 'nearest',cmap = cm.get_cmap("gist_heat"),aspect = 'auto')
    plt.title("Specgram of {0}".format(fileName))
    plt.xlabel("time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()

if plotCovariance:
    print("Computing the covariance")
    # covarianceArray = np.cov(Pxx[:,:Pxx.shape[1]/10.0],ddof = 0)

    totalSlices = 10
    if Pxx is None:
        (Pxx, freqs, bins) = mlab.specgram(actualDataY, NFFT=2 ** fftPower, Fs=1 / timeStep,
                                           noverlap=2 ** (fftPower - 1))
    # print ("Freqs: {0}".format(freqs.shape[0]))
    # print ("Expected Freqs: {0}".format(2**(fftPower - 1) + 1))
    for sliceIndex in range(totalSlices - 1):
        plt.figure(5 + sliceIndex)
        print ('Doing Slice {0}'.format(sliceIndex))
        covarianceArray = np.corrcoef(
            Pxx[:, Pxx.shape[1] * sliceIndex / float(totalSlices):Pxx.shape[1] * (sliceIndex + 1) / float(totalSlices)])
        plt.imshow(covarianceArray, interpolation='nearest', extent=[0, freqs[-1], freqs[-1], 0])
        plt.title('Covariance Matrix for frequency bins')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Frequency (Hz)')
    plt.show()

if mfccView:
    winlen = 0.025
    winstep = 0.01
    numcep = 100
    nfilt = numcep * 2
    nfft = 2 ** fftPower
    lowfreq = 0
    highfreq = samplingRate / 2

    mfcc_feat = mfcc(actualDataY, samplingRate, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft,
                     lowfreq=lowfreq, highfreq=highfreq)
    fbank_feat = logfbank(actualDataY, samplingRate)

    print(mfcc_feat.shape)
    plt.imshow(mfcc_feat.transpose(), interpolation='nearest', aspect='auto', cmap=cm.get_cmap("gist_heat"))
    plt.ylabel('Mel Cepstral Coefficient')
    plt.xlabel('Feature Sample')
    plt.show()
