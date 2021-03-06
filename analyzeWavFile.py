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
import matplotlib.colors as colors

from python_speech_features import mfcc
from python_speech_features import logfbank

import CreateFeature
import CreateUtils
from mpl_toolkits.mplot3d import Axes3D

from sklearn.externals import joblib

plt.close("all")

rawDataFolder = CreateUtils.getRawDataFolder()

# fileName = 'algebra00004.wav'
# fileName = 'cube00001.wav'
# fileName = 'cartoafit00001.wav'
# fileName = 'carneighborhood00010.wav'
# fileName = 'kitchentable00001.wav'
# fileName = 'afittest00000.wav'
# fileName = 'carsound00000.wav'
# fileName = 'biketest00001.wav'
# fileName = 'bikeneighborhood00028.wav'
fileName = 'afitmap00073.wav'

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
plotSemiVariogram = False

if plotAmplitude:
    print("Plot out actual amplitudes")
    plt.figure(1)
    plt.plot(actualDataX, actualDataY)
    plt.title('Signal Over time')
    plt.xlabel('time (s)')
    plt.ylabel('signal amplitude')
    plt.show()

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
    plt.semilogy()
    plt.show()

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
fftPower = 30
transformSpecgram = False
if plotSpecgram:
    plt.figure(4)
    windowSizeInSeconds = 0.1
    NFFT = int(windowSizeInSeconds * samplingRate)
    noverlap = NFFT / 2
    print("Computing specgram with 2 power of {0}".format(np.log2(NFFT)))
    print("Computing specgram with {0} points".format(NFFT))
    windowType = mlab.window_hanning  # mlab.window_none, mlab.window_hanning
    # Pxx, freqs, bins, im = plt.specgram(actualDataY,
    #                                      window=windowType,
    #                                      NFFT=NFFT,
    #                                      Fs=samplingRate,
    #                                      noverlap=noverlap,
    #                                      cmap=cm.get_cmap("hsv"),
    #                                      mode='psd',
    #                                      scale='dB',
    #                                      **{'interpolation': 'nearest', })

    Pxx, freqs, bins = mlab.specgram(actualDataY,
                                     window=windowType,
                                     NFFT=NFFT,
                                     Fs=samplingRate,
                                     noverlap=noverlap,
                                     mode='psd', )

    if transformSpecgram:
        savedFilterFile = CreateUtils.getPathRelativeToRoot(
            os.path.join(CreateUtils.getProcessedDataDatasetsFolder('bikeneighborhoodPackFileNormParticle'), 'SavedFilters', "FFTWindowDefault(sklearn-0.18).pkl"))
        savedFilterFile = CreateUtils.getAbsolutePath(savedFilterFile)
        print("Using Saved filter {0}".format(savedFilterFile))
        savedFilterDict = joblib.load(savedFilterFile)
        pca = savedFilterDict['pca']
        Pxx = np.transpose(pca.transform(np.transpose(Pxx)))

    PxxDb = np.log10(Pxx)
    PxxDb *= 10
    PxxDb[PxxDb < -50] = -50
    thisCmap = cm.get_cmap("jet")
    thisCmap.set_bad('w', 1)
    ax = plt.imshow(PxxDb,
                    interpolation='nearest',
                    cmap=thisCmap,
                    aspect='auto',
                    origin='lower',
                    extent=(bins[0], bins[-1], freqs[0], freqs[-1]))
    plt.title("Specgram of {0}".format(fileName))
    plt.xlabel("time (s)")
    plt.ylabel("Frequency (Hz)")
    cb1 = plt.colorbar(ax)
    cb1.set_label("Power (dB)")
    # fig = plt.figure(2)
    # ax = fig.gca(projection='3d')
    # X, Y = np.meshgrid(bins, freqs)
    # surf = ax.plot_surface(X, Y, Pxx, cmap=cm.get_cmap("coolwarm"))

    plt.show()

if plotCovariance:
    print("Computing the covariance")
    # covarianceArray = np.cov(Pxx[:,:Pxx.shape[1]/10.0],ddof = 0)

    totalSlices = 10
    windowSizeInSeconds = 0.1
    NFFT = int(windowSizeInSeconds * samplingRate)
    noverlap = 0
    scale = 'linear'
    if Pxx is None:
        (Pxx, freqs, bins) = mlab.specgram(actualDataY,
                                           window=mlab.window_none,
                                           NFFT=NFFT,
                                           Fs=samplingRate,
                                           noverlap=noverlap)

    # print ("Freqs: {0}".format(freqs.shape[0]))
    # print ("Expected Freqs: {0}".format(2**(fftPower - 1) + 1))
    for sliceIndex in range(totalSlices - 1):
        plt.figure(5 + sliceIndex)
        print ('Doing Slice {0}'.format(sliceIndex))
        covarianceArray = np.corrcoef(Pxx[:, Pxx.shape[1] * sliceIndex / float(totalSlices):Pxx.shape[1] * (sliceIndex + 1) / float(totalSlices)])
        if scale == 'dB':
            covarianceArray = 10 * np.log10(covarianceArray)
        plt.imshow(covarianceArray, interpolation='nearest', extent=[0, freqs[-1], freqs[-1], 0], cmap=cm.get_cmap("gist_heat"))
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

if plotSemiVariogram:
    windowSizeInSeconds = 1.
    timeStepsPerWindow = int(windowSizeInSeconds * samplingRate)
    hs = np.arange(timeStepsPerWindow)
    hsValues = np.zeros_like(hs)
    for h in hs:
        if h == 0:
            firstVals = actualDataY
            secondVals = actualDataY
        else:
            firstVals = actualDataY[:-h]
            secondVals = actualDataY[h:]
        hsValues[h] = np.mean(np.square(firstVals - secondVals))
    plt.plot(hs * timeStep, hsValues)
    plt.ylabel('Semivariance')
    plt.xlabel('Lag (s)')
    plt.show()


def SVh(P, h, bw):
    """
    Experimental semivariogram for a single lag
    """
    from scipy.spatial.distance import pdist, squareform
    pd = squareform(pdist(P[:, :2]))
    N = pd.shape[0]
    Z = list()
    for i in range(N):
        for j in range(i + 1, N):
            if (pd[i, j] >= h - bw) and (pd[i, j] <= h + bw):
                Z.append((P[i, 2] - P[j, 2]) ** 2.0)
    return np.sum(Z) / (2.0 * len(Z))


def SV(P, hs, bw):
    """
    Experimental variogram for a collection of lags
    """
    sv = list()
    for h in hs:
        sv.append(SVh(P, h, bw))
    sv = [[hs[i], sv[i]] for i in range(len(hs)) if sv[i] > 0]
    return np.array(sv).T
