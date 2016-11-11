# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:29:50 2015

@author: jacsk
"""
import os
import shutil
import re
import gzip
import cPickle
import numpy as np
import numpy.random
import matplotlib.pylab as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm

from sklearn import decomposition

import wave, struct
import hashlib
import scipy.stats
import scipy.io.wavfile
import scipy.signal
import scipy.interpolate
import gpxpy
import gpxpy.gpx
import gc

import yaml
import pandas as pd
from tqdm import tqdm

import CoordinateTransforms

from python_speech_features import mfcc
from python_speech_features import logfbank

import CreateDatasetConfig
import CreateFeatureConfig

import tictoc
import zipfile
import datetime
import dateutil
import dateutil.parser
import pytz
import tzlocal

timer = tictoc.tictoc()

gpxpy.gpx.DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'

doScaryShuffle = True


def find_nearestValue(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearestIndex(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearestIndexValue(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def everyOther(v, offset=0):
    return [v[i] for i in range(offset, len(v), 2)]


def everyOtherX(v, x, offset=0):
    return [v[i] for i in range(offset, len(v), x)]


def shuffleDimensions(x, shuffleOrder):
    for currentAxis in reversed(shuffleOrder):
        x = np.rollaxis(x, currentAxis)
    return x


def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def shuffle_in_unison_inplaceWithP(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], p


def shuffle_in_unison_scary(a, b):
    assert len(a) == len(b)
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def convertPathToThisOS(path):
    isNT = re.match(r"\S:[\\]+", path)
    isPOSIX = re.match(r"/[^/]+", path)
    thisPathOS = ''
    if isNT:
        thisPathOS = 'nt'
    elif isPOSIX:
        thisPathOS = 'posix'

    retPath = path
    if os.name == 'nt' and thisPathOS == 'posix':
        r"E:\\Users\\Joey\Documents\\Virtual Box Shared Folder\\"
        posixMatch = re.match(
            r"(?P<basePath>/media/sena/Greed Island/Users/Joey/Documents/Virtual Box Shared Folder/?)(?P<endPath>.*)",
            path)
        if posixMatch:
            ntBasePathArray = ["E:\\", "Users", "Joey", "Documents", "Virtual Box Shared Folder"]
            endPath = posixMatch.group("endPath")
            endPathArray = endPath.split("/")
            retPath = os.path.join(*ntBasePathArray + endPathArray)
    elif os.name == 'posix' and thisPathOS == 'nt':
        ntMatch = re.match(
            r"(?P<basePath>E:[\\]+Users[\\]+Joey[\\]+Documents[\\]+Virtual Box Shared Folder[\\]*)(?P<endPath>.*)",
            path)
        if ntMatch:
            linuxBasePathArray = ["/media", "sena", "Greed Island", "Users", "Joey", "Documents",
                                  "Virtual Box Shared Folder"]  # VLF signals raw data folder
            endPath = ntMatch.group("endPath")
            endPathArray = endPath.split("\\")
            retPath = os.path.join(*linuxBasePathArray + endPathArray)

    return retPath


def readDataFile(filePatherArg):
    filename, file_extension = os.path.splitext(filePatherArg)
    if file_extension == ".wav":
        useModule = False
        if useModule:
            (actualDataY, samplingRate) = readWAVFileWAVEModule(filePatherArg)
        else:
            (actualDataY, samplingRate) = readWAVFileSciPy(filePatherArg)
    elif file_extension == ".hf":
        (actualDataY, samplingRate) = readHFFile(filePatherArg)
    else:
        raise ValueError("Can only read .hf and .wav files invalid type of {0}".format(file_extension))
    return actualDataY, samplingRate


def readWAVFileWAVEModule(filePatherArg):
    print("Opening WAV File (WAVE Module): {0}".format(filePatherArg))
    waveFile = wave.open(filePatherArg, 'r')
    (numChannels, sampwidth, samplingRate, numFrames, comptype, compname) = waveFile.getparams()
    # print("number of channels: {0}".format(numChannels))
    # print("sample width {0}".format(sampwidth))
    # print("sampling rate {0}".format(samplingRate))
    # print("total frames {0}".format(numFrames))
    frames = waveFile.readframes(numFrames * numChannels)
    waveFile.close()
    out = struct.unpack_from("%dh" % numFrames * numChannels, frames)
    if numChannels == 2:
        left = np.array(list(everyOther(out, 0)))
        # right = np.array (list  (everyOther (out, 1)))
    elif numChannels == 3:
        left = np.array(list(everyOtherX(out, 2, 2)))  # this should be the Z axis
    else:
        left = np.array(out)
        # right = left
    actualDataY = left
    # actualDataX = np.arange(actualDataY.shape[-1])*1.0/samplingRate

    return actualDataY, samplingRate


def readWAVFileSciPy(filePatherArg):
    print("Opening WAV File (SciPy): {0}".format(filePatherArg))
    [samplingRate, waveArray] = scipy.io.wavfile.read(filePatherArg)
    if waveArray.shape[1] == 2:
        actualDataY = waveArray[:, 0]
    elif waveArray.shape[1] >= 3:
        actualDataY = waveArray[:, 2]
    else:
        actualDataY = waveArray[:, 0]
    return actualDataY, samplingRate


def readHFFile(filePatherArg):
    print("Opening HF file {0}".format(filePatherArg))
    with pd.HDFStore(filePatherArg, 'r') as datasetStore:
        waveArray = datasetStore['hsav'].as_matrix()
        samplingRate = datasetStore['samplingRate'].as_matrix()[0, 0]
        # timestamps = datasetStore['timestamps']
        # burstStartStops = datasetStore['gapStartStopIndices']
    if waveArray.shape[1] == 2:
        actualDataY = waveArray[:, 0]
    elif waveArray.shape[1] >= 3:
        actualDataY = waveArray[:, 2]
    else:
        actualDataY = waveArray[:, 0]
    return actualDataY, samplingRate


def getFFTFromDataFile(filePatherArg, showFiguresArg, fftPowerArg):
    (actualDataY, samplingRate) = readDataFile(filePatherArg)

    timeStep = 1.0 / samplingRate
    print("Computing specgram with {0} point FFT".format(fftPowerArg))
    if showFiguresArg:
        plt.figure(1)
        Pxx, freqs, bins, im = plt.specgram(actualDataY, NFFT=2 ** fftPowerArg, Fs=1 / timeStep,
                                            noverlap=2 ** (fftPowerArg - 1), cmap=cm.gist_heat)

        # plt.pcolormesh(bins,freqs,10*np.log10(Pxx),cmap=cm.gist_heat)
        # plt.imshow(10*np.log10(Pxx),interpolation = 'nearest',cmap = cm.gist_heat,aspect = 'auto')
    else:
        (Pxx, freqs, bins) = mlab.specgram(actualDataY, NFFT=2 ** fftPowerArg, Fs=1 / timeStep,
                                           noverlap=2 ** (fftPowerArg - 1))
    gc.collect()
    return Pxx, freqs, bins


def collectMelFeaturesFromWAVFile(filePatherArg, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512, lowfreq=0,
                                  highfreq=None):
    (actualDataY, samplingRate) = readDataFile(filePatherArg)
    if highfreq is None:
        highfreq = samplingRate / 2
    mfcc_feat = mfcc(actualDataY, samplingRate, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft,
                     lowfreq=lowfreq, highfreq=highfreq)
    return mfcc_feat


def collectCovarianceFFTFeaturesFromWAVFile(filePatherArg, fftPowerArg=10, showFiguresArg=False, windowTimeLengthArg=1,
                                            burstMethod=None):
    (Pxx, freqs, bins) = getFFTFromDataFile(filePatherArg, showFiguresArg, fftPowerArg)

    samplingRate = 0
    if burstMethod:
        with pd.HDFStore(filePatherArg, 'r') as datasetStore:
            samplingRate = datasetStore['samplingRate'].as_matrix()[0, 0]
            burstStartStops = datasetStore['gapStartStopIndices'].as_matrix()
        totalNumberOfWindowTime = burstStartStops.shape[0]
    else:
        totalNumberOfWindowTime = int(bins[-1] / windowTimeLengthArg)

    fPsubArg = np.array([])

    if showFiguresArg:
        totalNumberOfWindowTime = 1
    for numWindowTime in tqdm(range(totalNumberOfWindowTime), 'Window Calculations'):
        if burstMethod:
            timeStep = 1.0 / samplingRate
            windowTimeIndex = [find_nearestIndex(bins, burstStartStops[numWindowTime, 0] * timeStep),
                               find_nearestIndex(bins, burstStartStops[numWindowTime + 1, 1] * timeStep)]
        else:
            windowTimeIndex = [find_nearestIndex(bins, windowTimeLengthArg * numWindowTime),
                               find_nearestIndex(bins, windowTimeLengthArg * (numWindowTime + 1))]
        PxxWindow = Pxx[:, windowTimeIndex[0]:windowTimeIndex[1]]
        covarianceArray = np.corrcoef(PxxWindow)
        fC = covarianceArray.flatten()

        if fPsubArg.shape == (0L,):
            fPsubArg = fC
        else:
            fPsubArg = np.vstack((fPsubArg, fC))
    gc.collect()
    return fPsubArg


def collectPatchFFTFeaturesFromWAVFile(filePatherArg, fftPowerArg=10, windowTimeLengthArg=1,
                                       windowFreqBoundsArg=(0, 6000), NTArg=15, NFArg=13, useOverallWindowStatArg=1,
                                       statOrderArg=(0, 1, 2, 3, 4), normalizeStatsArg=True, showFiguresArg=False,
                                       burstMethod=None, burstStartStops=None):
    (Pxx, freqs, bins) = getFFTFromDataFile(filePatherArg, showFiguresArg, fftPowerArg)

    # gets stats for a block of Pxx a 2 dimensional array
    def getStatistics(statArrayArg, statOrderArg2):
        mean = np.mean(statArrayArg)
        variance = np.var(statArrayArg)
        standardDeviation = variance ** (1 / 2.0)
        skewness = scipy.stats.skew(statArrayArg.flatten())
        kurtosis = scipy.stats.kurtosis(statArrayArg.flatten())
        # skewness = ((skewnessSum/totalSamples) -3*average*variance-average**3)/standardDeviation**3 \
        # if standardDeviation > 0 else 0
        # kurtosis = ((kurtosisSum/totalSamples)-4*average *(skewnessSum/totalSamples)+ \
        # 6*average**2*(varianceSum/totalSamples)-3*average**4)/variance**2 if variance > 0 else 0
        fRiArg = np.array([mean, standardDeviation, variance, skewness, kurtosis])
        fRiArg = fRiArg[statOrderArg2]
        return fRiArg

    samplingRate = 0
    if burstMethod:
        with pd.HDFStore(filePatherArg, 'r') as datasetStore:
            samplingRate = datasetStore['samplingRate'].as_matrix()[0, 0]
            burstStartStops = datasetStore['gapStartStopIndices'].as_matrix()
        totalNumberOfWindowTime = burstStartStops.shape[0] - 1
    else:
        totalNumberOfWindowTime = int(bins[-1] / windowTimeLengthArg)
    windowFreqIndex = [find_nearestIndex(freqs, windowFreqBoundsArg[0]),
                       find_nearestIndex(freqs, windowFreqBoundsArg[1])]
    freqIndexesPerpatch = max((windowFreqIndex[1] - windowFreqIndex[0]) / NFArg, 1)
    print("Get stats for moving window")
    if showFiguresArg:
        totalNumberOfWindowTime = 1

    fXsubArgRows = totalNumberOfWindowTime
    fXsubArgColumns = len(statOrderArg) * NFArg * NTArg + (1 if useOverallWindowStatArg else 0)
    fXsubArg = np.zeros((fXsubArgRows, fXsubArgColumns))
    rowsProcessedfXsubArg = 0

    for numWindowTime in tqdm(range(totalNumberOfWindowTime), 'Window Calculations'):
        if burstMethod:
            timeStep = 1.0 / samplingRate
            windowTimeIndex = [find_nearestIndex(bins, burstStartStops[numWindowTime, 0] * timeStep),
                               find_nearestIndex(bins, burstStartStops[numWindowTime + 1, 1] * timeStep)]
        else:
            windowTimeIndex = [find_nearestIndex(bins, windowTimeLengthArg * numWindowTime),
                               find_nearestIndex(bins, windowTimeLengthArg * (numWindowTime + 1))]
        windowTimesPerpatch = (windowTimeIndex[1] - windowTimeIndex[0]) / NTArg
        windowBaseIndex = [windowFreqIndex[0], windowTimeIndex[0]]
        windowEndIndex = [windowFreqIndex[1], windowTimeIndex[1]]

        fC = np.zeros((fXsubArgColumns,))
        rowsProcessedfC = 0

        if useOverallWindowStatArg:
            PxxWindow = Pxx[windowBaseIndex[0]:windowEndIndex[0], windowBaseIndex[1]:windowEndIndex[1]]
            fRi = getStatistics(PxxWindow, statOrderArg2=statOrderArg)
            # fC = np.append(fC,fRi)
            fC[rowsProcessedfC:(rowsProcessedfC + fRi.shape[0])] = fRi
            rowsProcessedfC += fRi.shape[0]

            if showFiguresArg:
                plt.figure(2)
                plt.subplot(1, totalNumberOfWindowTime, numWindowTime + 1)
                plt.imshow(10 * np.log10(PxxWindow), interpolation='nearest', cmap=cm.gist_heat, aspect='auto')

        counter = 0
        for nF in range(NFArg):
            for nT in range(NTArg):
                baseIndex = [windowBaseIndex[0] + nF * freqIndexesPerpatch,
                             windowBaseIndex[1] + nT * windowTimesPerpatch]
                endIndex = [windowBaseIndex[0] + (nF + 1) * freqIndexesPerpatch,
                            windowBaseIndex[1] + (nT + 1) * windowTimesPerpatch]
                # print "baseIndex {0} endIndex {1}".format(baseIndex,endIndex)
                PxxSubWindow = Pxx[baseIndex[0]:endIndex[0], baseIndex[1]:endIndex[1]]
                fRi = getStatistics(PxxSubWindow, statOrderArg2=statOrderArg)
                # fC = np.append(fC,fRi)
                fC[rowsProcessedfC:(rowsProcessedfC + fRi.shape[0])] = fRi
                rowsProcessedfC += fRi.shape[0]

                if showFiguresArg:
                    plt.figure(3)
                    counter += 1
                    plt.subplot(NFArg, NTArg, counter)
                    plt.imshow(10 * np.log10(PxxSubWindow), interpolation='nearest', cmap=cm.gist_heat, aspect='auto')

                    """plt.figure(4)
                    plt.subplot(NF,NT,counter)
                    statVector = np.array(fRi)
                    plt.imshow(statVector.reshape((2,2)),interpolation = 'nearest',
                        cmap = cm.gist_heat,aspect = 'auto',vmin=0.0, vmax = 14.0)
                    """

        if normalizeStatsArg:
            statArray = []
            totalStatsArg = len(statOrderArg)
            for totalStatIndex in range(totalStatsArg):
                stat = fC[totalStatIndex::totalStatsArg]
                stat = stat / stat.max()
                statArray.append(stat)
            fC = np.vstack(tuple(statArray)).reshape((-1,), order='F')
            if showFiguresArg:
                plt.figure(5)
                plt.imshow(statArray[0][:].reshape((NFArg, NTArg)), interpolation='nearest', cmap=cm.gist_heat,
                           aspect='auto')
            gc.collect()
        # #stack the fC onto the previous ones for this WAV file
        # if fXsubArg.shape == (0L,):
        #     fXsubArg = fC
        # else:
        #     fXsubArg = np.vstack((fXsubArg,fC))
        fXsubArg[rowsProcessedfXsubArg, :] = fC
        rowsProcessedfXsubArg += 1
    gc.collect()  # hopefully this gets rid of the variables I just turned to None
    return fXsubArg


def collectFFTFeaturesFromWAVFile(filePatherArg, fftPowerArg=10, showFiguresArg=False, burstMethod=None,
                                  windowFreqBoundsArg=(0, 6000)):
    (Pxx, freqs, bins) = getFFTFromDataFile(filePatherArg, showFiguresArg, fftPowerArg)
    if burstMethod:
        raise NotImplementedError
    windowFreqIndex = [find_nearestIndex(freqs, windowFreqBoundsArg[0]),
                       find_nearestIndex(freqs, windowFreqBoundsArg[1])]
    fXsubArg = numpy.transpose(Pxx[windowFreqIndex[0]:windowFreqIndex[1], :])
    gc.collect()  # hopefully this gets rid of the variables I just turned to None
    return fXsubArg


def collectRawAmplitudeFromWAVFile(filePatherArg, poolingSize=None, poolingType=None, maxTime=None):
    (actualDataY, samplingRate) = readDataFile(filePatherArg)
    # if burstMethod:
    #     with pd.HDFStore(filePatherArg, 'r') as datasetStore:
    #         samplingRate = datasetStore['samplingRate'].as_matrix()[0, 0]
    #         burstStartStops = datasetStore['gapStartStopIndices'].as_matrix()
    # pick off the time I want
    if maxTime is not None:
        actualDataY = actualDataY[:int(maxTime * samplingRate)]
    # downsample if applicable
    if poolingType == 'max':
        actualDataY = numpy.max(
            actualDataY[:actualDataY.shape[0] - (actualDataY.shape[0] % poolingSize)].reshape((-1, poolingSize)), 1)
    elif poolingType == 'mean':
        actualDataY = numpy.mean(
            actualDataY[:actualDataY.shape[0] - (actualDataY.shape[0] % poolingSize)].reshape((-1, poolingSize)), 1)
    elif poolingType == 'downsample':
        incrementScale = 13
        for x in range(poolingSize / incrementScale):
            actualDataY = scipy.signal.decimate(actualDataY, incrementScale)
        if poolingSize % incrementScale > 0:
            actualDataY = scipy.signal.decimate(actualDataY, poolingSize % incrementScale)
    elif poolingType == 'none' or poolingType is None:
        pass
    else:
        raise ValueError("Invalid pooling type")
    fXsubArg = actualDataY
    return fXsubArg


def convertToLinuxEpochFromISODateFormat(isoString):
    return np.int64((dateutil.parser.parse(isoString) - datetime.datetime(1970, 1, 1, 0, 0, 0, 0,
                                                                          pytz.UTC)).total_seconds() * 1000.0)


def datetimeMillisecondTimestampFromLinuxEpoch(datetimeObject):
    return np.int64((datetimeObject - datetime.datetime(1970, 1, 1, 0, 0, 0, 0, pytz.UTC)).total_seconds() * 1000.0)


def getGPSDataFromGPXFile(fileNameGPX):
    if not os.path.exists(fileNameGPX):
        raise ValueError("No gpx file found at path {0}".format(fileNameGPX))
    with open(fileNameGPX, 'r') as gpsFile:
        gpx = gpxpy.parse(gpsFile)
    trackPoints = gpx.tracks[0].segments[0].points

    ndtype = [('timestamp', '|S27'), ('altitude', float), ('longitude', float), ('course', float),
              ('horizontalAccuracy', int), ('time', np.float64), ('latitude', float), ('speed', float),
              ('verticalAccuracy', int)]

    gpsDataArray = np.zeros(len(trackPoints), dtype=ndtype)
    startTime = trackPoints[0].time
    for trackPoint, counter in zip(trackPoints, range(len(trackPoints))):
        timeSinceStart = trackPoint.time - startTime
        elapsedTime = timeSinceStart.total_seconds() if timeSinceStart.total_seconds() > 0 else 0
        gpsDataArray['time'][counter] = elapsedTime
        gpsDataArray['timestamp'][counter] = trackPoint.time.isoformat() + 'Z'
        gpsDataArray['latitude'][counter] = trackPoint.latitude
        gpsDataArray['longitude'][counter] = trackPoint.longitude
        gpsDataArray['altitude'][counter] = trackPoint.elevation
        gpsDataArray['verticalAccuracy'][counter] = trackPoint.vertical_dilution
        gpsDataArray['horizontalAccuracy'][counter] = trackPoint.horizontal_dilution
        gpsDataArray['speed'][counter] = 0
        gpsDataArray['course'][counter] = 0
    return gpsDataArray


def getMetadataFromWahooZipFile(zipfilepath):
    zippy = zipfile.ZipFile(zipfilepath)
    cadenceArray = np.array([])
    speedArray = np.array([])
    locationArray = np.array([])
    heartArray = np.array([])
    footpodArray = np.array([])
    maArray = np.array([])
    for zippedfile in zippy.infolist():
        cadenceMatch = re.match('.*BikeCad.csv', zippedfile.filename)
        speedMatch = re.match('.*BikeSpd.csv', zippedfile.filename)
        locationMatch = re.match('.*Loc.csv', zippedfile.filename)
        heartMatch = re.match('.*.Heart.csv', zippedfile.filename)
        footpodMatch = re.match('.*Footpod.csv', zippedfile.filename)
        maMatch = re.match('.*MA.csv', zippedfile.filename)

        if cadenceMatch:
            with zippy.open(zippedfile.filename) as csvFile:
                ndtype = [('_ID', int), ('CrankRevolutions', int), ('Cadence', float), ('WorkoutActive', bool),
                          ('Timestamp', np.int64)]
                cadenceArray = np.genfromtxt(csvFile, delimiter=',', names=True, dtype=ndtype)
        if speedMatch:
            with zippy.open(zippedfile.filename) as csvFile:
                ndtype = [('_ID', int), ('WheelRevolutions', float), ('WheelCircumference', float),
                          ('SpeedInstant', float), ('WorkoutActive', bool),
                          ('Timestamp', np.int64)]
                speedArray = np.genfromtxt(csvFile, delimiter=',', names=True, dtype=ndtype)
        if locationMatch:
            with zippy.open(zippedfile.filename) as csvFile:
                ndtype = [('Longitude', float), ('DistanceOffset', float), ('Accuracy', float), ('Altitude', float),
                          ('WorkoutActive', bool), ('Timestamp', np.int64),
                          ('Latitude', float), ('TotalDistance', float), ('GradeDeg', float), ('_ID', int),
                          ('Speed', float)]
                locationArray = np.genfromtxt(csvFile, delimiter=',', names=True, dtype=ndtype)
        if heartMatch:
            with zippy.open(zippedfile.filename) as csvFile:
                ndtype = [('_ID', int), ('TotalHeartbeats', int), ('Heartrate', int), ('WorkoutActive', bool),
                          ('Timestamp', np.int64)]
                heartArray = np.genfromtxt(csvFile, delimiter=',', names=True, dtype=ndtype)
        if footpodMatch:
            with zippy.open(zippedfile.filename) as csvFile:
                ndtype = [('TotalStrides', float), ('TotalDistance', float), ('_ID', int), ('Cadence', int),
                          ('Speed', float), ('WorkoutActive', bool), ('Timestamp', np.int64)]
                footpodArray = np.genfromtxt(csvFile, delimiter=',', names=True, dtype=ndtype)
        if maMatch:
            with zippy.open(zippedfile.filename) as csvFile:
                ndtype = [('GroundContactTime', float), ('MotionCount', int), ('MotionPowerZ', float), ('Cadence', int),
                          ('MotionPowerX', float), ('WorkoutActive', bool), ('Timestamp', np.int64),
                          ('Smoothness', float), ('MotionPowerY', float), ('_ID', int),
                          ('VerticalOscillation', float), ]
                maArray = np.genfromtxt(csvFile, delimiter=',', names=True, dtype=ndtype)
    return cadenceArray, speedArray, locationArray, heartArray, footpodArray, maArray


def getMetadataFromiPhoneZipFile(zipfilepath):
    zippy = zipfile.ZipFile(zipfilepath)
    accelerometerArray = np.array([])
    gyroscopeArray = np.array([])
    locationArray = np.array([])
    headingArray = np.array([])
    for zippedfile in zippy.infolist():
        accelerometerMatch = re.match('.*Accelerometer.csv', zippedfile.filename)
        gyroscopeMatch = re.match('.*Gyroscope.csv', zippedfile.filename)
        locationMatch = re.match('.*Location.csv', zippedfile.filename)
        headingMatch = re.match('.*.Heading.csv', zippedfile.filename)

        if accelerometerMatch:
            with zippy.open(zippedfile.filename) as csvFile:
                ndtype = [('y', float), ('timestamp', '|S25'), ('time', np.float64), ('z', float), ('x', float)]
                accelerometerArray = np.genfromtxt(csvFile, delimiter=',', dtype=ndtype, names=True)
        if gyroscopeMatch:
            with zippy.open(zippedfile.filename) as csvFile:
                ndtype = [('timestamp', '|S25'), ('time', np.float64), ('xrate', np.float64), ('yrate', np.float64),
                          ('zrate', np.float64)]
                gyroscopeArray = np.genfromtxt(csvFile, delimiter=',', dtype=ndtype, names=True)
        if locationMatch:
            with zippy.open(zippedfile.filename) as csvFile:
                ndtype = [('timestamp', '|S25'), ('altitude', float), ('longitude', float), ('course', float),
                          ('horizontalAccuracy', int), ('time', np.float64), ('latitude', float), ('speed', float),
                          ('verticalAccuracy', int)]
                locationArray = np.genfromtxt(csvFile, delimiter=',', dtype=ndtype, names=True)
        if headingMatch:
            with zippy.open(zippedfile.filename) as csvFile:
                ndtype = [('trueHeading', float), ('timestamp', '|S25'), ('time', np.float64), ('headingAccuracy', int),
                          ('magneticHeading', float)]
                headingArray = np.genfromtxt(csvFile, delimiter=',', dtype=ndtype, names=True)
    return accelerometerArray, gyroscopeArray, locationArray, headingArray


def determineMetadata(fileNameArg, numOfSamples=None, windowTimeLengthArg=1.0, burstMethod=None):
    baseFilePathWithNumber, fileExtension = os.path.splitext(fileNameArg)
    baseFileName = os.path.basename(baseFilePathWithNumber)[:-5]
    baseFileNameWithNumber = os.path.basename(baseFilePathWithNumber)

    startTimeiPhone = None
    zipFilePath = baseFilePathWithNumber + '.zip'
    if not os.path.exists(zipFilePath):
        raise ValueError("Zip file was not found at {0}".format(zipFilePath))

    # gather metadata from zip files
    (cadenceArray, speedArray, locationWahooArray, heartArray, footpodArray, maArray) = getMetadataFromWahooZipFile(
        zipFilePath)
    (accelerometerArray, gyroscopeArray, locationiPhoneArray, headingArray) = getMetadataFromiPhoneZipFile(zipFilePath)
    # Rename columns
    # cadence
    if cadenceArray.dtype.names:
        newNames = list(cadenceArray.dtype.names)
        newNames[newNames.index('Cadence')] = 'CadenceBike'
        cadenceArray.dtype.names = tuple(newNames)
    # wahoo location
    if locationWahooArray.dtype.names:
        newNames = list(locationWahooArray.dtype.names)
        newNames[newNames.index('Latitude')] = 'LatitudeWahoo'
        newNames[newNames.index('Longitude')] = 'LongitudeWahoo'
        newNames[newNames.index('Altitude')] = 'AltitudeWahoo'
        newNames[newNames.index('Speed')] = 'SpeedWahoo'
        newNames[newNames.index('TotalDistance')] = 'TotalDistanceWahoo'
        locationWahooArray.dtype.names = tuple(newNames)
    # wahoo footpod
    if footpodArray.dtype.names:
        newNames = list(footpodArray.dtype.names)
        newNames[newNames.index('Speed')] = 'SpeedFootpod'
        newNames[newNames.index('Cadence')] = 'CadenceFootpad'
        newNames[newNames.index('TotalDistance')] = 'TotalDistanceFootpad'
        footpodArray.dtype.names = tuple(newNames)
    # wahoo ma
    if maArray.dtype.names:
        newNames = list(maArray.dtype.names)
        newNames[newNames.index('Cadence')] = 'CacenceMA'
        maArray.dtype.names = tuple(newNames)
    # iphone location
    if not locationiPhoneArray.dtype.names:
        gpxFilePath = baseFilePathWithNumber + '.gpx'
        if os.path.exists(gpxFilePath):
            locationiPhoneArray = getGPSDataFromGPXFile(gpxFilePath)
        else:
            raise ValueError("No location file. No way to determine iPhone start time")
    if locationiPhoneArray.dtype.names:
        newNames = list(locationiPhoneArray.dtype.names)
        newNames[newNames.index('latitude')] = 'LatitudeiPhone'
        newNames[newNames.index('longitude')] = 'LongitudeiPhone'
        newNames[newNames.index('altitude')] = 'AltitudeiPhone'
        newNames[newNames.index('speed')] = 'SpeediPhone'
        locationiPhoneArray.dtype.names = tuple(newNames)
        startTimeiPhone = dateutil.parser.parse(locationiPhoneArray['timestamp'][0])
        startTimeiPhone = startTimeiPhone.replace(tzinfo=pytz.utc)

    millisecondsSinceLinuxEpoch = datetimeMillisecondTimestampFromLinuxEpoch(startTimeiPhone)
    cadenceElapsedtime = (cadenceArray['Timestamp'] - millisecondsSinceLinuxEpoch) / 1000.0 \
        if cadenceArray.dtype.names else cadenceArray
    speedElapsedtime = (speedArray['Timestamp'] - millisecondsSinceLinuxEpoch) / 1000.0 \
        if speedArray.dtype.names else speedArray
    locationWahooElapsedtime = (locationWahooArray['Timestamp'] - millisecondsSinceLinuxEpoch) / 1000.0 \
        if locationWahooArray.dtype.names else locationWahooArray
    heartElapsedtime = (heartArray['Timestamp'] - millisecondsSinceLinuxEpoch) / 1000.0 \
        if heartArray.dtype.names else heartArray
    footpodElapsedtime = (footpodArray['Timestamp'] - millisecondsSinceLinuxEpoch) / 1000.0 \
        if footpodArray.dtype.names else footpodArray
    maElapsedtime = (maArray['Timestamp'] - millisecondsSinceLinuxEpoch) / 1000.0 \
        if maArray.dtype.names else maArray
    accelerometerElapsedtime = accelerometerArray['time'] \
        if accelerometerArray.dtype.names else accelerometerArray
    gyroscopeElapsedtime = gyroscopeArray['time'] \
        if gyroscopeArray.dtype.names else gyroscopeArray
    locationiPhoneElapsedtime = locationiPhoneArray['time'] \
        if locationiPhoneArray.dtype.names else locationiPhoneArray
    headingElapsedTime = headingArray['time'] \
        if headingArray.dtype.names else headingArray

    if numOfSamples is None:
        numOfSamples = int(locationiPhoneElapsedtime[-1] / windowTimeLengthArg)

    totalMetadataColumns = len(CreateDatasetConfig.allMetadataNamesList)
    zeroFillArray = np.zeros((numOfSamples, totalMetadataColumns)).transpose()
    finalMetadataArray = np.core.records.fromarrays(zeroFillArray, dtype=CreateDatasetConfig.allMetadataDtyesList)

    # Now start to populate the final metadata array
    if burstMethod:
        with pd.HDFStore(fileNameArg, 'r') as datasetStore:
            samplingRate = datasetStore['samplingRate'].as_matrix()[0, 0]
            burstStartStops = datasetStore['gapStartStopIndices'].as_matrix()
    else:
        samplingRate = 0
        burstStartStops = numpy.array([])

    dateOffset = pd.DateOffset(seconds=windowTimeLengthArg)
    datelist = pd.date_range(startTimeiPhone, freq=dateOffset, periods=numOfSamples)

    if burstMethod:
        interpTimes = np.zeros((numOfSamples,))
        for timeStepIndex in tqdm(range(numOfSamples), 'Get GPS Continuous By Time Step'):
            timeSteper = 1.0 / samplingRate
            timeStep = burstStartStops[timeStepIndex, 1] * timeSteper
            interpTimes[timeStepIndex] = timeStep
        finalMetadataArray['ElapsedSeconds'] = interpTimes

        for timeStepIndex in tqdm(range(numOfSamples), 'Get GPS Continuous By Time Step'):
            timeSteper = 1.0 / samplingRate
            timeStep = burstStartStops[timeStepIndex, 1] * timeSteper
            # now set up the arrays
            currentTime = startTimeiPhone + datetime.timedelta(seconds=timeStep)
            finalMetadataArray['BaseFileName'][timeStepIndex] = baseFileName
            finalMetadataArray['Hour'][timeStepIndex] = currentTime.hour
            finalMetadataArray['DayOfYear'][timeStepIndex] = currentTime.timetuple().tm_yday
            finalMetadataArray['DayPercent'][timeStepIndex] = (currentTime.hour + (currentTime.minute / 60.0)) / 24.0
    else:
        interpTimes = np.arange(numOfSamples) * windowTimeLengthArg
        finalMetadataArray['BaseFileName'] = baseFileName
        finalMetadataArray['BaseFileNameWithNumber'] = baseFileNameWithNumber
        finalMetadataArray['ElapsedSeconds'] = interpTimes
        finalMetadataArray['Hour'] = datelist.hour
        finalMetadataArray['DayOfYear'] = datelist.dayofyear
        finalMetadataArray['DayPercent'] = (datelist.hour + (datelist.minute / 60.0)) / 24.0
        finalMetadataArray['SeasonSineWave'] = np.cos(2 * np.pi / 365.0 * (173 - datelist.dayofyear))

    arraysAll = [cadenceArray, speedArray, locationWahooArray, heartArray, footpodArray, maArray,
                 accelerometerArray, gyroscopeArray, locationiPhoneArray, headingArray]
    namesAll = [CreateDatasetConfig.cadenceNames, CreateDatasetConfig.speedNames,
                CreateDatasetConfig.locationWahooNames, CreateDatasetConfig.heartNames,
                CreateDatasetConfig.footpadNames, CreateDatasetConfig.maNames,
                CreateDatasetConfig.accelerometerNames, CreateDatasetConfig.gyroscopeNames,
                CreateDatasetConfig.locationiPhoneNames, CreateDatasetConfig.headingNames]
    elapsedTimeAll = [cadenceElapsedtime, speedElapsedtime, locationWahooElapsedtime, heartElapsedtime,
                      footpodElapsedtime, maElapsedtime, accelerometerElapsedtime, gyroscopeElapsedtime,
                      locationiPhoneElapsedtime, headingElapsedTime]
    for metadataArray, metadataNames, elapsedtime in zip(arraysAll,
                                                         namesAll,
                                                         elapsedTimeAll):
        if metadataArray.dtype.names:
            for metadataName in metadataNames:
                finalMetadataArray[metadataName] = np.interp(interpTimes, elapsedtime, metadataArray[metadataName])
    return finalMetadataArray


def getAllBaseFileNames(rawDataFolderArg, nomatch=None):
    rawFiles = os.listdir(rawDataFolderArg)
    fileSet = set()
    for rawFile in rawFiles:
        matcher = re.match(r'(?P<baseName>[\d\w]+)[\d]{5,10}\.(wav|hf)', rawFile)
        if matcher:
            baseName = matcher.group('baseName')
            if nomatch is None or baseName not in nomatch:
                fileSet.add(baseName)
    if 'myMusicFile' in fileSet:
        fileSet.remove('myMusicFile')

    return list(fileSet)


def filterFilesByFileNumber(files, baseFileName, removeFileNumbers=(), onlyFileNumbers=()):
    if baseFileName in removeFileNumbers or baseFileName in onlyFileNumbers:
        removeMask = np.ones(files.shape, dtype=bool)
        if baseFileName in removeFileNumbers and len(removeFileNumbers[baseFileName]) > 0:
            removeMask[np.array(removeFileNumbers[baseFileName])] = False
        onlyMask = np.ones(files.shape, dtype=bool)
        if baseFileName in onlyFileNumbers and len(onlyFileNumbers[baseFileName]) > 0:
            onlyMask[np.array(onlyFileNumbers[baseFileName])] = False
            onlyMask = np.logical_not(onlyMask)
        finalMask = np.logical_and(removeMask, onlyMask)
        files = files[finalMask]
    return files


def getFileStatisticsOfFile(fileNameArg):
    baseFilePathWithNumber, fileExtension = os.path.splitext(fileNameArg)
    baseFileNameWithNumber = os.path.basename(baseFilePathWithNumber)
    fileNumber = int(baseFileNameWithNumber[-5:])
    gpxFilePath = baseFilePathWithNumber + '.gpx'
    if os.path.exists(gpxFilePath):
        locationiPhoneArray = getGPSDataFromGPXFile(gpxFilePath)
    else:
        raise ValueError("omg no gpx file")

    startTimeiPhone = dateutil.parser.parse(locationiPhoneArray['timestamp'][0])
    startTimeiPhone = startTimeiPhone.replace(tzinfo=pytz.utc)
    runTimeSeconds = locationiPhoneArray['time'][-1]
    dayOfYear = startTimeiPhone.timetuple().tm_yday
    hourStart = startTimeiPhone.hour

    return np.array((startTimeiPhone, runTimeSeconds, dayOfYear, hourStart, baseFileNameWithNumber, fileNumber),
                    dtype=[("startTimeiPhone", object), ("runTimeSeconds", np.float64), ("dayOfYear", np.int64),
                           ("hourStart", np.int64), ("baseFileNameWithNumber", "|S30"), ("fileNumber", np.int64)])


############################
# Main function to build
############################
def getStatisticsOnSet(datasetParameters, featureParameters, allBaseFileNames=None, removeFileNumbers=(),
                       onlyFileNumbers=(), showFiguresArg=False):
    # feature parameters
    rawDataFolder = convertPathToThisOS(featureParameters['rawDataFolder'])
    featureSetName = featureParameters['featureSetName']
    imageShape = featureParameters['imageShape']
    windowTimeLength = featureParameters['feature parameters']['windowTimeLength']
    if allBaseFileNames is None:
        allBaseFileNames = getAllBaseFileNames(rawDataFolder)

    # general dataset variables
    datasetName = datasetParameters['datasetName']
    fileNamesNumbersToSets = datasetParameters['fileNamesNumbersToSets']
    defaultSetName = datasetParameters['defaultSetName']
    allIncludedFiles = []
    outputStrings = []
    allStats = None
    for baseFileName in allBaseFileNames:
        # get all files with this base name
        files = np.array([f2 for f2 in sorted(os.listdir(rawDataFolder)) if
                          re.match(re.escape(baseFileName) + r'\d*\.(wav|hf)', f2)])
        files = filterFilesByFileNumber(files, baseFileName, removeFileNumbers=removeFileNumbers,
                                        onlyFileNumbers=onlyFileNumbers)

        if len(files) == 0:
            print("*********** Base file name {0} had no files **************".format(baseFileName))
        else:
            for fileName in tqdm(files, "BaseFileName: {0}".format(baseFileName)):
                allIncludedFiles.append(str(fileName))
                filePather = os.path.join(rawDataFolder, fileName)
                fileStats = getFileStatisticsOfFile(filePather)
                if allStats is not None:
                    allStats = np.vstack((allStats, fileStats))
                else:
                    allStats = fileStats
                statString = "File:{baseFileName:30} Day:{dayOfYear:3} Hour:{hourStart:4} run time(min):{runTimeMinutes:5.2f}".format(
                    dayOfYear=fileStats["dayOfYear"],
                    hourStart=fileStats["hourStart"],
                    baseFileName=fileStats["baseFileNameWithNumber"],
                    runTimeMinutes=fileStats["runTimeSeconds"] / 60.0)
                outputStrings.append(statString)

    print("Feature Set:{featureSetName} \n Image Shape:{imageShape} \n Window Time:{windowTimeLength} \n".format(
        featureSetName=featureSetName, imageShape=imageShape, windowTimeLength=windowTimeLength))
    print("Data Set:{datasetName} \n total files:{totalFiles}".format(
        datasetName=datasetName,
        totalFiles=len(allIncludedFiles)))
    print '\n'.join(outputStrings)

    if showFiguresArg:
        plt.bar(allStats["fileNumber"], allStats["dayOfYear"])
        plt.ylim(np.min(allStats["dayOfYear"]) - 1, np.max(allStats["dayOfYear"]) + 1)

        ax = plt.gca()
        rects = ax.patches

        labels = []
        for filename in allIncludedFiles:
            setNames = getSetNameForFile(filename, defaultSetName, fileNamesNumbersToSets)
            labels.append(','.join(setNames))

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 1, label, ha='center', va='bottom')
        plt.show()


def buildFeatures(featureParameters, forceRefreshFeatures=False, showFigures=False, maxFiles=10000,
                  allBaseFileNames=None, removeFileNumbers=(), onlyFileNumbers=()):
    processedFilesCount = 0
    rawDataFolder = convertPathToThisOS(featureParameters['rawDataFolder'])
    featureDataFolder = convertPathToThisOS(featureParameters['featureDataFolder'])
    if allBaseFileNames is None:
        allBaseFileNames = getAllBaseFileNames(rawDataFolder)
    if len(allBaseFileNames) == 0:
        print("*********** There are no files to process b/c allBaseFileNames is empty **************")

    # general variables used for collecting features
    featureSetName = featureParameters['featureSetName']
    featureMethod = featureParameters['feature parameters']['featureMethod']
    windowTimeLength = featureParameters['feature parameters']['windowTimeLength']
    burstMethod = featureParameters['feature parameters']['burstMethod'] if 'burstMethod' in featureParameters[
        'feature parameters'] else None
    convertToInt = featureParameters['feature parameters']['convertToInt'] \
        if 'convertToInt' in featureParameters['feature parameters'] else None

    # fft specific
    fftPower = featureParameters['feature parameters']['fftPower'] \
        if 'fftPower' in featureParameters['feature parameters'] else None
    windowFreqBounds = featureParameters['feature parameters']['windowFreqBounds'] \
        if 'windowFreqBounds' in featureParameters['feature parameters'] else None

    # patch specific parameters
    NT = featureParameters['feature parameters']['NT'] if 'NT' in featureParameters['feature parameters'] else None
    NF = featureParameters['feature parameters']['NF'] if 'NF' in featureParameters['feature parameters'] else None
    useOverallWindowStat = featureParameters['feature parameters']['useOverallWindowStat'] \
        if 'useOverallWindowStat' in featureParameters['feature parameters'] else None
    statOrder = featureParameters['feature parameters']['statOrder'] \
        if 'statOrder' in featureParameters['feature parameters'] else None
    normalizeStats = featureParameters['feature parameters']['normalizeStats'] \
        if 'normalizeStats' in featureParameters['feature parameters'] else None

    # MFCC specific parameters
    winstep = featureParameters['feature parameters']['winstep'] if 'winstep' in featureParameters[
        'feature parameters'] else None
    numcep = featureParameters['feature parameters']['numcep'] if 'numcep' in featureParameters[
        'feature parameters'] else None
    nfilt = featureParameters['feature parameters']['nfilt'] if 'nfilt' in featureParameters[
        'feature parameters'] else None

    # RAW amplitude specific
    poolingSize = featureParameters['feature parameters']['poolingSize'] if 'poolingSize' in featureParameters[
        'feature parameters'] else None
    poolingType = featureParameters['feature parameters']['poolingType'] if 'poolingType' in featureParameters[
        'feature parameters'] else None
    maxTime = featureParameters['feature parameters']['maxTime'] if 'maxTime' in featureParameters[
        'feature parameters'] else None

    timer.tic("Feature collection for feature set {0}".format(featureSetName))
    if not os.path.exists(featureDataFolder):
        os.makedirs(featureDataFolder)
    for baseFileName in allBaseFileNames:
        # get all files with this base name
        files = np.array([f2 for f2 in sorted(os.listdir(rawDataFolder)) if
                          re.match(re.escape(baseFileName) + r'\d*\.(wav|hf)', f2)])
        files = filterFilesByFileNumber(files, baseFileName, removeFileNumbers=removeFileNumbers,
                                        onlyFileNumbers=onlyFileNumbers)

        if len(files) == 0:
            print("*********** Base file name {0} had no files **************".format(baseFileName))
        else:
            for fileName in files:
                featureStorePath = os.path.join(featureDataFolder, fileName + ".h5")
                doesIndexExist = os.path.exists(featureStorePath)
                if not doesIndexExist or forceRefreshFeatures:
                    processedFilesCount += 1
                    if processedFilesCount > maxFiles:
                        break
                    filePather = os.path.join(rawDataFolder, fileName)
                    # metadataArray = determineMetadata(filePather, 100, windowTimeLength, burstMethod=None)
                    # get features from the WAV file to build an fP matrix
                    if featureMethod == 'Patch':
                        fXsub = collectPatchFFTFeaturesFromWAVFile(filePather,
                                                                   fftPowerArg=fftPower,
                                                                   windowTimeLengthArg=windowTimeLength,
                                                                   windowFreqBoundsArg=windowFreqBounds,
                                                                   NTArg=NT,
                                                                   NFArg=NF,
                                                                   useOverallWindowStatArg=useOverallWindowStat,
                                                                   statOrderArg=statOrder,
                                                                   normalizeStatsArg=normalizeStats,
                                                                   burstMethod=burstMethod,
                                                                   showFiguresArg=showFigures)
                    elif featureMethod == 'Covariance':
                        fXsub = collectCovarianceFFTFeaturesFromWAVFile(filePather,
                                                                        fftPowerArg=fftPower,
                                                                        showFiguresArg=showFigures,
                                                                        windowTimeLengthArg=windowTimeLength,
                                                                        burstMethod=burstMethod)
                    elif featureMethod == 'MFCC':
                        fXsub = collectMelFeaturesFromWAVFile(filePather,
                                                              winlen=windowTimeLength,
                                                              winstep=winstep,
                                                              numcep=numcep,
                                                              nfilt=nfilt,
                                                              nfft=2 ** fftPower,
                                                              lowfreq=windowFreqBounds[0],
                                                              highfreq=windowFreqBounds[1], )
                    elif featureMethod == 'FFT':
                        fXsub = collectFFTFeaturesFromWAVFile(filePather,
                                                              fftPowerArg=fftPower,
                                                              showFiguresArg=showFigures,
                                                              burstMethod=burstMethod,
                                                              windowFreqBoundsArg=windowFreqBounds)
                    elif featureMethod == "RawAmplitude":
                        fXsub = collectRawAmplitudeFromWAVFile(filePather,
                                                               poolingSize=poolingSize,
                                                               poolingType=poolingType,
                                                               maxTime=maxTime)
                    else:
                        raise ValueError("Invalid featureMethod {0}".format(featureMethod))

                    if convertToInt is not None and convertToInt is True:
                        if 'convertToIntMax' in featureParameters['feature parameters']:
                            maxInt = featureParameters['feature parameters']['convertToIntMax']
                            fXsubMax = np.max(fXsub)
                            fXsubMultiplier = maxInt / fXsubMax
                            fXsub *= fXsubMultiplier
                        elif 'convertToIntMultiplier' in featureParameters['feature parameters']:
                            fXsubMultiplier = featureParameters['feature parameters']['convertToIntMultiplier']
                            fXsub *= fXsubMultiplier
                        fXsub = np.array(fXsub, dtype=np.int64)

                    # Get metadata from iPhone and Wahoo files and calculated parameters
                    metadataArray = determineMetadata(filePather, fXsub.shape[0], windowTimeLength, burstMethod=None)

                    with pd.HDFStore(featureStorePath, 'a') as featureStore:
                        print("Save stats in {0}".format(featureStorePath))
                        featureStore['X'] = pd.DataFrame(fXsub)
                        featureStore['metadata'] = pd.DataFrame(metadataArray)

                    gc.collect()  # try to free some memory because i just saved off the stats to the hard disk
    if processedFilesCount > maxFiles:
        print ("***************\n Failed to do all files due to maxFile limit of {0}\n***************".format(maxFiles))
    timer.toc()
    return not processedFilesCount > maxFiles  # 0 means i failed to do all files 1 means I'm done


def getSetNameForFile(filename, defaultSetName, fileNamesNumbersToSets):
    """
    :param filename: name of the base file we want to assign a set to
    :param defaultSetName: the default name for the set if this filename isn't in the fileNamesNumberstoSets array
    :param fileNamesNumbersToSets: array of tuples that give set names
        to the filename [(setName, fileBaseName, fileNumber), ... ]
    :return: the set name for this file
    """
    setNames = []
    matcher = re.match(r'(?P<baseName>[\d\w]+)(?P<fileNumber>[\d]{5,10})\.(wav|hf)', filename)
    if matcher:
        filename = matcher.group("baseName")
        for setNameAndNumbers in fileNamesNumbersToSets:
            if filename == setNameAndNumbers[1]:
                fileNumber = int(matcher.group("fileNumber"))
                if fileNumber in setNameAndNumbers[2]:
                    setNames.append(setNameAndNumbers[0])
    else:
        print ("file didn't match problem {0}".format(filename))
    if len(setNames) == 0:
        setNames.append(defaultSetName)
    return setNames


def getRowsForFileDueToSequence(thisFileRows, timestepsPerSequence, offsetBetweenSequences,
                                fractionOfRandomToUse, padSequenceWithZeros):
    if timestepsPerSequence is not None and offsetBetweenSequences is not None:
        if not padSequenceWithZeros:
            totalPossibleUniqueSamples = int(
                (thisFileRows - timestepsPerSequence) / float(abs(offsetBetweenSequences)))
        else:
            totalPossibleUniqueSamples = np.ceil(thisFileRows / float(abs(offsetBetweenSequences)))
        # if the offset is 0 or negative use the fraction instead
        if offsetBetweenSequences > 0:
            actualNewSamples = totalPossibleUniqueSamples
        else:
            actualNewSamples = int(totalPossibleUniqueSamples * fractionOfRandomToUse)
        return actualNewSamples
    else:
        # if eiter were none then the sequence length is all the samples and thus there is one row
        return 1


def updateOutputFinalWithOutputTemp(outputLabelsFinal, outputLabelsTemp):
    # consolidate all outputLabels into the final array

    # Old Way
    # outputLabelsTempTuple = [tuple(output) for output in outputLabelsTemp]
    # for output in tqdm(outputLabelsTempTuple, "Labels Loop"):
    #     if not outputLabelsFinal or output not in set(outputLabelsFinal):
    #         outputLabelsFinal.append(output)

    # New Way
    outputLabelsTempSet = set(outputLabelsTemp)
    outputLabelsFinalSet = set(outputLabelsFinal)
    sameLabels = outputLabelsFinalSet.intersection(outputLabelsTempSet)
    newLabels = outputLabelsTempSet.difference(sameLabels)
    outputLabelsFinal += (list(newLabels))
    return outputLabelsFinal


def getYColumnSize(datasetParameters):
    yValueType = datasetParameters['yValueType']

    # GPS parameters
    if 'y value parameters' in datasetParameters:
        includeAltitude = datasetParameters['y value parameters']['y value by gps Parameters']['includeAltitude']
    else:
        includeAltitude = False

    # Sequence Parameters
    makeSequences = datasetParameters['makeSequences'] if 'makeSequences' in datasetParameters else False
    timestepsPerSequence = datasetParameters[
        'timestepsPerSequence'] if 'timestepsPerSequence' in datasetParameters else 100
    timeDistributedY = datasetParameters['timeDistributedY'] if 'timeDistributedY' in datasetParameters else False

    totalyColumns = 1
    if yValueType in CreateDatasetConfig.yValueContinuousTypes:
        if yValueType in CreateDatasetConfig.yValueGPSTypes:
            if includeAltitude:
                totalyColumns = 3
            else:
                totalyColumns = 2
    if makeSequences and timeDistributedY:
        totalyColumns *= int(timestepsPerSequence)
    return totalyColumns


def deterineYValuesGridByGPSArrayInLocalLevelCoords(gpsDataArrayLocalLevelCoords, gridSize):
    gridSize = np.array(gridSize[0:gpsDataArrayLocalLevelCoords.shape[1]])
    gridArg = np.ceil(np.array(gpsDataArrayLocalLevelCoords / gridSize)) * gridSize[None, :]
    fYTempArg = gridArg - gridSize/2.0
    return fYTempArg


def sliceUpArray(arrayToSlice, startSample, endSample, timeDistributed, flipCurrentSlice):
    if startSample > endSample:
        if flipCurrentSlice:
            part1 = arrayToSlice[startSample:]
            part2 = np.flipud(arrayToSlice[-endSample:])
            curSlice = np.concatenate([part1, part2]).flatten()
        else:
            part1 = arrayToSlice[startSample:]
            part2 = arrayToSlice[:endSample]
            curSlice = np.concatenate([part1, part2]).flatten()
    else:
        if flipCurrentSlice:
            curSlice = np.flipud(arrayToSlice[-endSample:-startSample]).flatten()
        else:
            curSlice = arrayToSlice[startSample:endSample].flatten()
    return curSlice


def getIndexOfLabels(dataArray, currentLabelsList):
    # get the unique parts from this fYTemp array
    # timer.tic("Getting unique stuff")
    fYTempWithRowView = np.ascontiguousarray(dataArray).view(
        np.dtype((np.void, dataArray.dtype.itemsize * dataArray.shape[1])))
    _, idx, unique_inverse = np.unique(fYTempWithRowView, return_index=True, return_inverse=True)
    uniquefYTemp = dataArray[idx]
    # timer.toc()
    # timer.tic("Getting new labels")
    # make them labels now
    outputLabelsTemp = []
    uniqueAsStrings = np.array(uniquefYTemp, dtype=str)
    for yIndex in range(uniqueAsStrings.shape[0]):
        newLabel = ','.join(yItem for yItem in uniqueAsStrings[yIndex])
        outputLabelsTemp.append(newLabel)
    # timer.toc()
    # timer.tic("Update current labels")
    # add the new temp lables to the end of the larger final space removing duplicates
    currentLabelsList = updateOutputFinalWithOutputTemp(currentLabelsList, outputLabelsTemp)
    # timer.toc()
    # timer.tic("Map current labels to new labels")
    indexOfNewLabels = np.zeros_like(unique_inverse)
    for newOutputLabel, indexNumber in zip(outputLabelsTemp, range(len(outputLabelsTemp))):
        indexOfNewLabels[indexNumber == unique_inverse] = currentLabelsList.index(newOutputLabel)
    # timer.toc()
    return indexOfNewLabels, currentLabelsList


def getXYTempAndLabelsFromFile(featureStorePath, datasetParameters, imageShape, useMetadata, metadataList, yValueType,
                               outputLabelsFinal=None, rowPackagingMetadataFinal=None):
    filterXToUpperDiagonal = datasetParameters[
        'filterXToUpperDiagonal'] if 'filterXToUpperDiagonal' in datasetParameters else False

    # GPS parameters
    if 'y value parameters' in datasetParameters:
        includeAltitude = datasetParameters['y value parameters']['y value by gps Parameters']['includeAltitude']
        gridSize = datasetParameters['y value parameters']['y value by gps Parameters']['gridSize'] \
            if 'gridSize' in datasetParameters['y value parameters']['y value by gps Parameters'] else None
        localLevelOriginInECEF = datasetParameters['y value parameters']['y value by gps Parameters'][
            'localLevelOriginInECEF']
    else:
        includeAltitude = False
        gridSize = (10, 10, 1000)
        localLevelOriginInECEF = [506052.051626, -4882162.055080, 4059778.630410]

    # Sequence Parameters
    makeSequences = datasetParameters['makeSequences'] if 'makeSequences' in datasetParameters else False
    timestepsPerSequence = datasetParameters[
        'timestepsPerSequence'] if 'timestepsPerSequence' in datasetParameters else 100
    # offsetBetweenSequences =  -1 will mean use random start locations
    offsetBetweenSequences = datasetParameters[
        'offsetBetweenSequences'] if 'offsetBetweenSequences' in datasetParameters else 1
    fractionOfRandomToUse = datasetParameters[
        'fractionOfRandomToUse'] if 'fractionOfRandomToUse' in datasetParameters else 0.5
    padSequenceWithZeros = datasetParameters[
        'padSequenceWithZeros'] if 'padSequenceWithZeros' in datasetParameters else False
    repeatSequenceBeginningAtEnd = datasetParameters['repeatSequenceBeginningAtEnd'] \
        if 'repeatSequenceBeginningAtEnd' in datasetParameters else False
    repeatSequenceEndingAtEnd = datasetParameters[
        'repeatSequenceEndingAtEnd'] if 'repeatSequenceEndingAtEnd' in datasetParameters else False
    timeDistributedY = datasetParameters['timeDistributedY'] if 'timeDistributedY' in datasetParameters else False

    # packaging data
    rowPackagingStyle = datasetParameters['rowPackagingStyle'] if 'rowPackagingStyle' in datasetParameters else None

    with pd.HDFStore(featureStorePath, 'r') as featureStore:
        print (featureStorePath)
        fXTemp = featureStore['X']
        assert 'metadata' in featureStore, "There is no metadata in file {0}".format(
            featureStorePath)
        metadataFull = featureStore['metadata']
        if useMetadata:
            metadataArray = metadataFull[metadataList]
            fXTemp = (pd.concat([fXTemp, metadataArray], axis=1))
        fXTemp = fXTemp.as_matrix()

    #########################
    # Features X Matrix     #
    #########################
    # timer.tic("X Values")
    # remove any NANs and make them 0
    if np.isnan(fXTemp).any():
        print("Found {0} nan(s) in X matrix".format(np.sum(np.isnan(fXTemp))))
        fXTemp[np.isnan(fXTemp)] = 0

    # filter the X to the upper triangle
    if filterXToUpperDiagonal:
        rowColMatrix = np.zeros(imageShape[1:3])
        for i in range(rowColMatrix.shape[0]):
            for j in range(rowColMatrix.shape[1]):
                if i == 0:
                    rowColMatrix[i, j] = True
                else:
                    rowColMatrix[i, j] = j / float(i) >= rowColMatrix.shape[1] / float(
                        rowColMatrix.shape[0])
        flatMask = rowColMatrix.flatten() == 1
        fXTemp = fXTemp[:, flatMask]
    # timer.toc()

    ###################
    # Target Y Values #
    ###################
    # timer.tic("Making Y Values")
    if yValueType in CreateDatasetConfig.yValueGPSTypes:
        fYTemp = metadataFull[['LatitudeiPhone', 'LongitudeiPhone', 'AltitudeiPhone']].as_matrix()
        # turn Lat Lon into radians from degrees
        fYTemp[:, 0:2] = fYTemp[:, 0:2] * np.pi / 180.0
        ecefCoords = CoordinateTransforms.LlhToEcef(fYTemp)
        fYTemp = CoordinateTransforms.EcefToLocalLevel(localLevelOriginInECEF, ecefCoords)
        if not includeAltitude:
            fYTemp = fYTemp[:, 0:2]
        if yValueType == 'gpsPolar':
            fYTemp[:, 0] = np.sqrt(
                np.multiply(fYTemp[:, 0], fYTemp[:, 0]) + np.multiply(fYTemp[:, 1], fYTemp[:, 1]))
            fYTemp[:, 1] = np.arctan2(fYTemp[:, 1], fYTemp[:, 0]) * 180.0 / np.pi
        if yValueType == 'gpsD':
            # force the Y output to be a number of a grid location
            fYTemp = deterineYValuesGridByGPSArrayInLocalLevelCoords(fYTemp, gridSize)
    elif yValueType == 'time':
        fYTemp = metadataFull['ElapsedSeconds']
    elif yValueType == 'file':
        fYTemp = metadataFull['BaseFileName']
    else:
        raise ValueError("Invalid yValueType: {0}".format(yValueType))
    # remove any nans
    if np.isnan(fYTemp).any():
        print("Found {0} nan(s) in Y matrix".format(np.sum(np.isnan(fYTemp))))
        fYTemp[np.isnan(fYTemp)] = 0
    # timer.toc()

    ##########
    # Labels #
    ##########
    # timer.tic("Making Labels")
    if yValueType in CreateDatasetConfig.yValueDiscreteTypes:
        labelIndices, outputLabelsFinal = getIndexOfLabels(fYTemp, outputLabelsFinal)
        fYTemp = labelIndices
        fYTemp = fYTemp[:, None]
    # timer.toc()

    ######################
    # Package Data #######
    ######################
    # timer.tic("Making Package Data")
    if rowPackagingStyle == 'BaseFileNameWithNumber':
        dataArray = metadataFull['BaseFileNameWithNumber'].as_matrix()
        dataArray = dataArray.astype(str)
        dataArray = dataArray[:, None]
        rowPackagingMetadata, rowPackagingMetadataFinal = getIndexOfLabels(dataArray, rowPackagingMetadataFinal)
        rowPackagingMetadata = rowPackagingMetadata[:, None]
    elif rowPackagingStyle == 'gpsD':
        gridSizePackage = datasetParameters['gridSizePackage'] if 'gridSizePackage' in datasetParameters else (1, 1, 1)
        gpsCoords = metadataFull[['LatitudeiPhone', 'LongitudeiPhone', 'AltitudeiPhone']].as_matrix()
        # turn Lat Lon into radians from degrees
        gpsCoords[:, 0:2] = gpsCoords[:, 0:2] * np.pi / 180.0
        ecefCoords = CoordinateTransforms.LlhToEcef(gpsCoords)
        localLevelCoords = CoordinateTransforms.EcefToLocalLevel(localLevelOriginInECEF, ecefCoords)
        localLevelCoords = localLevelCoords[:, 0:2]
        newGridCoordinates = deterineYValuesGridByGPSArrayInLocalLevelCoords(localLevelCoords, gridSizePackage)
        rowPackagingMetadata, rowPackagingMetadataFinal = getIndexOfLabels(newGridCoordinates,
                                                                           rowPackagingMetadataFinal)
        rowPackagingMetadata = rowPackagingMetadata[:, None]
    else:
        rowPackagingMetadata = None
    # timer.toc()

    ######################
    # Sequence Creator ###
    ######################
    if makeSequences:
        # timer.tic("Make Sequences")
        actualNewSamples = getRowsForFileDueToSequence(fXTemp.shape[0],
                                                       timestepsPerSequence,
                                                       offsetBetweenSequences,
                                                       fractionOfRandomToUse,
                                                       padSequenceWithZeros)
        # indicies based on the actual file size
        offsetIndices = np.arange(actualNewSamples)

        if len(fYTemp.shape) < 2:
            fYTemp = numpy.expand_dims(fYTemp, axis=1)

        sequencefX = np.zeros((int(actualNewSamples), int(fXTemp.shape[1] * timestepsPerSequence)))
        sequencefY = np.zeros((int(actualNewSamples), int(fYTemp.shape[1] * timestepsPerSequence)))
        sequenceRowPackagingMetadata = np.zeros(
            (int(actualNewSamples), int(rowPackagingMetadata.shape[1] * timestepsPerSequence)))
        for offsetIndex in offsetIndices:
            # Determine start and ending
            if repeatSequenceBeginningAtEnd or repeatSequenceEndingAtEnd:
                startSample = np.mod(int(offsetIndex * offsetBetweenSequences), fXTemp.shape[0])
                endSample = np.mod(int(offsetIndex * offsetBetweenSequences + timestepsPerSequence),
                                   fXTemp.shape[0] + 1)
                cutSize = int(endSample - startSample) if endSample > startSample else int(
                    endSample + (fXTemp.shape[0] - startSample))
                flipCurrentSlice = repeatSequenceEndingAtEnd and np.mod(
                    int(offsetIndex * offsetBetweenSequences + timestepsPerSequence) / fXTemp.shape[0], 2) == 1
            else:
                startSample = int(offsetIndex * offsetBetweenSequences)
                endSample = int(
                    min(offsetIndex * offsetBetweenSequences + timestepsPerSequence,
                        fXTemp.shape[0] - 1))
                cutSize = endSample - startSample

            # Slice up X
            curSliceX = sliceUpArray(fXTemp, startSample, endSample, True, flipCurrentSlice)
            sequencefX[int(offsetIndex), 0:int(cutSize * fXTemp.shape[1])] = curSliceX

            # Slice up y
            curSliceY = sliceUpArray(fYTemp, startSample, endSample, True, flipCurrentSlice)
            sequencefY[int(offsetIndex), 0:int(cutSize * fYTemp.shape[1])] = curSliceY

            # Slice up rowPackagingMetadata
            curSliceRPM = sliceUpArray(rowPackagingMetadata, startSample, endSample, True, flipCurrentSlice)
            sequenceRowPackagingMetadata[int(offsetIndex), 0:int(cutSize * rowPackagingMetadata.shape[1])] = curSliceRPM

        if not timeDistributedY:
            sequencefY = sequencefY[:, 0]
        fXTemp = sequencefX
        fYTemp = sequencefY
        rowPackagingMetadata = sequenceRowPackagingMetadata
        # timer.toc()
    return fXTemp, fYTemp, outputLabelsFinal, rowPackagingMetadata, rowPackagingMetadataFinal


def finalFilteringSets(setDictArg, datasetParameters, filterFitSets, yValueType, totalXColumns, totalYColumns):
    shuffleFinalSamples = datasetParameters[
        'shuffleFinalSamples'] if 'shuffleFinalSamples' in datasetParameters else False

    # Sequence Parameters
    makeSequences = datasetParameters['makeSequences'] if 'makeSequences' in datasetParameters else False
    timestepsPerSequence = datasetParameters[
        'timestepsPerSequence'] if 'timestepsPerSequence' in datasetParameters else 100

    # Filter parameters
    filterPCA = datasetParameters['filterPCA'] if 'filterPCA' in datasetParameters else False
    filterPCAn_components = datasetParameters[
        'filterPCAn_components'] if 'filterPCAn_components' in datasetParameters else None
    filterPCAwhiten = datasetParameters['filterPCAwhiten'] if 'filterPCAwhiten' in datasetParameters else False
    yScaleFactor = datasetParameters['y value parameters']['yScaleFactor'] if 'yScaleFactor' in datasetParameters[
        'y value parameters'] else 1.0
    yBias = datasetParameters['y value parameters']['yBias'] if 'yBias' in datasetParameters[
        'y value parameters'] else 0.0
    yNormalized = datasetParameters['y value parameters']['yNormalized'] if 'yNormalized' in datasetParameters[
        'y value parameters'] else False
    xScaleFactor = datasetParameters['xScaleFactor'] if 'xScaleFactor' in datasetParameters else 1.0
    xBias = datasetParameters['xBias'] if 'xBias' in datasetParameters else 0.0
    xNormalized = datasetParameters['xNormalized'] if 'xNormalized' in datasetParameters else False

    filterFitX = np.empty((0, totalXColumns))
    filterFitY = np.empty((0, totalYColumns))
    for setName, setValue in setDictArg.iteritems():
        if setName in filterFitSets:
            filterFitX = np.vstack((filterFitX, setValue[0]))
            filterFitY = np.vstack((filterFitY, setValue[1]))

    if makeSequences:
        filterFitX = np.reshape(filterFitX,
                                (int(filterFitX.shape[0] * timestepsPerSequence),
                                 int(filterFitX.shape[1] / timestepsPerSequence)))
        filterFitY = np.reshape(filterFitY,
                                (int(filterFitY.shape[0] * timestepsPerSequence),
                                 int(filterFitY.shape[1] / timestepsPerSequence)))
    if filterPCA:
        if filterPCAn_components is None:
            filterPCAn_components = filterFitX.shape[1]
        pca = decomposition.PCA(n_components=filterPCAn_components, whiten=filterPCAwhiten, copy=True)
        pca.fit(filterFitX)
    else:
        pca = None

    if xNormalized:
        xBias = -np.min(filterFitX, axis=0)
        xScaleFactor = 1.0 / (np.max(filterFitX, axis=0) + xBias)
    if yNormalized:
        yBias = - np.min(filterFitY, axis=0)
        yScaleFactor = 1.0 / (np.max(filterFitY, axis=0) + yBias)

    for setName, setValue in setDictArg.iteritems():
        if yValueType in CreateDatasetConfig.yValueDiscreteTypes:
            # remove rows I rejected by giving a y value of -1
            goodRows = np.array(setValue[1] >= 0)
            goodRows = goodRows.flatten()
            setValue[0] = setValue[0][goodRows, :]
            setValue[1] = setValue[1][goodRows]
        else:
            # Y scaling
            # reshape out of sequences
            if makeSequences:
                filterY = np.reshape(setValue[1],
                                     (int(setValue[1].shape[0] * timestepsPerSequence),
                                      int(setValue[1].shape[1] / timestepsPerSequence)))
            else:
                filterY = setValue[1]
            filterY += yBias
            filterY *= yScaleFactor
            # reshape back to sequences
            if makeSequences:
                setValue[1] = np.reshape(filterY,
                                         (int(filterY.shape[0] / timestepsPerSequence),
                                          int(filterY.shape[1] * timestepsPerSequence)))
            else:
                setValue[1] = filterY
        # X filter and scaling changes
        # reshape out of sequences
        if makeSequences:
            filterX = np.reshape(setValue[0],
                                 (int(setValue[0].shape[0] * timestepsPerSequence),
                                  int(setValue[0].shape[1] / timestepsPerSequence)))
        else:
            filterX = setValue[0]
        # now I can work on a sample by sample basis not sequence by sequence
        # do some biasing and scaling
        filterX += xBias
        filterX *= xScaleFactor
        # do the PCA transform
        if filterPCA:
            filterX = pca.transform(filterX)
        # reshape back to sequences
        if makeSequences:
            setValue[0] = np.reshape(filterX,
                                     (int(filterX.shape[0] / timestepsPerSequence),
                                      int(filterX.shape[1] * timestepsPerSequence)))
        else:
            setValue[0] = filterX

        if shuffleFinalSamples:
            timer.tic("Shuffle {0} Set".format(setName))
            if not doScaryShuffle:
                (setValue[0], setValue[1]) = shuffle_in_unison_inplace(setValue[0], setValue[1])
            else:
                shuffle_in_unison_scary(setValue[0], setValue[1])
            timer.toc()
    return setDictArg, xScaleFactor, xBias, yScaleFactor, yBias


def plotY(yWorking, yNormalized):
    plt.figure(1)
    xer = yWorking[0::2].flatten()
    yer = yWorking[1::2].flatten()
    plt.plot(yer, xer)
    plt.scatter(yer, xer, c=range(xer.size), cmap=plt.cm.get_cmap('spectral'))
    plt.ylim([-800, 600])
    plt.xlim([-800, 600])
    if yNormalized:
        plt.ylim([0, 1])
        plt.xlim([0, 1])
    plt.show()


def plotYKeras(yWorking, stepper, seqs, yNormalized):
    ranger = np.arange(seqs, yWorking.shape[0], stepper)
    filterRange = ranger
    print filterRange
    xer = yWorking[filterRange, 0::2].flatten()
    yer = yWorking[filterRange, 1::2].flatten()
    plt.plot(yer, xer)
    plt.scatter(yer, xer, c=range(xer.size), cmap=plt.cm.get_cmap('spectral'))
    plt.ylim([-800, 600])
    plt.xlim([-800, 600])
    if yNormalized:
        plt.ylim([0, 1])
        plt.xlim([0, 1])
    plt.show()


def repackageSets(setDictArg, datasetParameters, rowProcessingMetadataDictArg):
    rowPackagingStyle = datasetParameters['rowPackagingStyle'] if 'rowPackagingStyle' in datasetParameters else None
    padRowPackageWithZeros = datasetParameters[
        'padRowPackageWithZeros'] if 'padRowPackageWithZeros' in datasetParameters else False
    repeatRowPackageBeginningAtEnd = datasetParameters[
        'repeatRowPackageBeginningAtEnd'] if 'repeatRowPackageBeginningAtEnd' in datasetParameters else False
    repeatRowPackageEndingAtEnd = datasetParameters[
        'repeatRowPackageEndingAtEnd'] if 'repeatRowPackageEndingAtEnd' in datasetParameters else False
    alternateRowsForKeras = datasetParameters[
        'alternateRowsForKeras'] if 'alternateRowsForKeras' in datasetParameters else False
    timestepsPerKerasBatchRow = datasetParameters[
        'timestepsPerKerasBatchRow'] if 'timestepsPerKerasBatchRow' in datasetParameters else 1
    allSetsSameRows = datasetParameters['allSetsSameRows'] if 'allSetsSameRows' in datasetParameters else False

    packagedRowsPerSetDict = {}
    kerasRowMultiplier = 1
    if rowPackagingStyle is not None:
        newRowsMax = 0
        totalColumnCountMax = 0
        for setName, setValue in setDictArg.iteritems():
            rowMetadata = rowProcessingMetadataDictArg[setName]
            uniqueRows, uniqueCounts = np.unique(rowMetadata, return_counts=True)
            newRows = uniqueRows.shape[0]
            newRowsMax = newRows if newRows > newRowsMax else newRowsMax
            totalColumnCount = int(np.max(uniqueCounts) if padRowPackageWithZeros else np.min(uniqueCounts))
            totalColumnCountMax = totalColumnCount if totalColumnCount > totalColumnCountMax else totalColumnCountMax

        if alternateRowsForKeras:
            if padRowPackageWithZeros:
                kerasRowMultiplier = int(np.ceil(totalColumnCountMax / float(timestepsPerKerasBatchRow)))
            else:
                kerasRowMultiplier = int(np.floor(totalColumnCountMax / float(timestepsPerKerasBatchRow)))
            totalColumnCount = kerasRowMultiplier * timestepsPerKerasBatchRow
        else:
            totalColumnCount = totalColumnCountMax

        for setName, setValue in setDictArg.iteritems():
            rowMetadata = rowProcessingMetadataDictArg[setName]
            uniqueRows, uniqueCounts = np.unique(rowMetadata, return_counts=True)
            if allSetsSameRows is True:
                newRows = newRowsMax
            else:
                newRows = uniqueRows.shape[0]
            packagedRowsPerSetDict[setName] = newRows
            totalColumnsX = setValue[0].shape[1]
            totalColumnsY = setValue[1].shape[1]
            tempX = np.zeros((newRows, totalColumnsX * totalColumnCount))
            tempY = np.zeros((newRows, totalColumnsY * totalColumnCount))
            tempSets = [tempX, tempY]
            for uniqueRowIndex in range(uniqueRows.shape[0]):
                uniqueRow = uniqueRows[uniqueRowIndex]
                uniqueCount = uniqueCounts[uniqueRowIndex]
                for totalColumnsSet, setNumber in zip([totalColumnsX, totalColumnsY], [0, 1]):
                    tempSet = tempSets[setNumber]
                    sliceTo = min(uniqueCount, totalColumnCount)
                    tempSet[uniqueRowIndex, :sliceTo * totalColumnsSet] = \
                        setValue[setNumber][rowMetadata.squeeze() == uniqueRow, :].flatten()[:sliceTo * totalColumnsSet]
                    if uniqueCount < totalColumnCount and (
                                repeatRowPackageBeginningAtEnd or repeatRowPackageEndingAtEnd):
                        repeatTimes = totalColumnCount / uniqueCount
                        for repeatTime in range(repeatTimes):

                            startSample = int((repeatTime + 1) * uniqueCount * totalColumnsSet)
                            endSample = min(int((repeatTime + 2) * uniqueCount * totalColumnsSet), tempSet.shape[1])

                            sliceStart = 0
                            sliceEnd = endSample - startSample

                            if repeatRowPackageBeginningAtEnd or repeatTime % 2 == 1:
                                tempSet[uniqueRowIndex, startSample:endSample] = \
                                    tempSet[uniqueRowIndex, :sliceEnd - sliceStart]
                            elif repeatRowPackageEndingAtEnd:
                                newSliceStart = uniqueCount * totalColumnsSet - sliceEnd + sliceStart
                                newSliceEnd = uniqueCount * totalColumnsSet
                                newshape = ((sliceEnd - sliceStart) / totalColumnsSet, totalColumnsSet)
                                tempSet[uniqueRowIndex, startSample:endSample] = \
                                    np.flipud(
                                        np.reshape(tempSet[uniqueRowIndex, newSliceStart:newSliceEnd],
                                                   newshape=newshape)).flatten()
                    tempSets[setNumber] = tempSet
                    # if setNumber == 1:
                    #     print("Before keras Packaging Set Name {0} Row Index {1}".format(setName,uniqueRowIndex))
                    #     plotY(tempSet[uniqueRowIndex, :], True)
            if alternateRowsForKeras:
                for setNumber, totalColumnsSet in zip([0, 1], [totalColumnsX, totalColumnsY]):
                    tempSet = tempSets[setNumber]
                    # tempSet.shape = (run x timesteps x data dim)
                    tempSet = tempSet.reshape(
                        (newRows, kerasRowMultiplier, timestepsPerKerasBatchRow * totalColumnsSet), order='C')
                    # tempSet.shape = (run x slice of run x timestep data dim)
                    tempSet = tempSet.reshape(
                        (newRows * kerasRowMultiplier, timestepsPerKerasBatchRow * totalColumnsSet), order='F')
                    # tempSet.shape = (run count in timestep x timestep data dim)
                    tempSets[setNumber] = tempSet
                    # if setNumber == 1:
                    #     for runNumber in range(newRows):
                    #         print("After Keras packaging setName {0} runNumber {1}".format(setName, runNumber))
                    #         plotYKeras(tempSet, newRows, runNumber, True)
            setValue[0] = tempSets[0]
            setValue[1] = tempSets[1]
    return setDictArg, packagedRowsPerSetDict, kerasRowMultiplier


def buildDataSet(datasetParameters, featureParameters, forceRefreshDataset=False):
    # feature variables
    featureDataFolder = convertPathToThisOS(featureParameters['featureDataFolder'])
    featureSetName = featureParameters['featureSetName']
    imageShape = featureParameters['imageShape']

    # general dataset variables
    rawDataFolder = convertPathToThisOS(datasetParameters['rawDataFolder'])
    processedDataFolder = convertPathToThisOS(datasetParameters['processedDataFolder'])
    datasetName = datasetParameters['datasetName']

    # which files to use variables
    allBaseFileNames = datasetParameters['allBaseFileNames']
    removeFileNumbers = datasetParameters['removeFileNumbers'] if 'removeFileNumbers' in datasetParameters else {}
    onlyFileNumbers = datasetParameters['onlyFileNumbers'] if 'onlyFileNumbers' in datasetParameters else {}

    # Set variables
    fileNamesNumbersToSets = datasetParameters[
        'fileNamesNumbersToSets'] if 'fileNamesNumbersToSets' in datasetParameters else []
    setFractions = datasetParameters['setFractions'] if 'setFractions' in datasetParameters else []
    defaultSetName = datasetParameters['defaultSetName'] if 'defaultSetName' in datasetParameters else "train"
    shuffleSamplesPerFile = datasetParameters['shuffleSamplesPerFile']
    assert shuffleSamplesPerFile is False, "Shuffle Sample Per File not allowed with separate sets"
    shuffleFinalSamples = datasetParameters['shuffleFinalSamples']
    maxSamplesPerFile = datasetParameters['maxSamplesPerFile']
    assert maxSamplesPerFile < 1, "Max samples Per file not allowed with separate sets"

    # X variables
    useMetadata = datasetParameters['useMetadata'] if 'useMetadata' in datasetParameters else False
    metadataList = datasetParameters['metadataList'] if 'metadataList' in datasetParameters else []

    # Y variables
    yValueType = datasetParameters['yValueType']
    overlapTime = datasetParameters['overlapTime'] if 'overlapTime' in datasetParameters else None
    assert yValueType != 'time' and (overlapTime is None or overlapTime is False), "Overlap time feature was removed"

    # Sequence Parameters
    makeSequences = datasetParameters['makeSequences'] if 'makeSequences' in datasetParameters else False
    timestepsPerSequence = datasetParameters[
        'timestepsPerSequence'] if 'timestepsPerSequence' in datasetParameters else 100
    # offsetBetweenSequences =  -1 will mean use random start locations
    offsetBetweenSequences = datasetParameters[
        'offsetBetweenSequences'] if 'offsetBetweenSequences' in datasetParameters else 1
    fractionOfRandomToUse = datasetParameters[
        'fractionOfRandomToUse'] if 'fractionOfRandomToUse' in datasetParameters else 0.5
    padSequenceWithZeros = datasetParameters[
        'padSequenceWithZeros'] if 'padSequenceWithZeros' in datasetParameters else False

    # packaging data
    rowPackagingStyle = datasetParameters['rowPackagingStyle'] if 'rowPackagingStyle' in datasetParameters else None

    # Filter Variables
    filterFitSets = datasetParameters['filterFitSets'] if 'filterFitSets' in datasetParameters else None

    rngSeed = datasetParameters['rngSeed']
    np.random.seed(rngSeed)

    datasetFile = os.path.join(processedDataFolder, featureSetName + '.hf')
    if not os.path.exists(datasetFile) or forceRefreshDataset:
        timer.tic("Build dataset {0} for feature set {1}".format(datasetName, featureSetName))
        timer.tic("Extract features from stored files")

        outputLabelsFinal = []
        rowPackagingMetadataFinal = []

        setDict = {}
        rowProcessingMetadataDict = {}

        totalXColumns = None
        totalMetadataPackagingColumns = 1

        totalRowsInSetDict = {}
        totalFilesInSetDict = {}

        rowsPerFile = {}
        rowsPerBaseFileDict = {}
        maxRowsOfAllFiles = 0
        maxSamplesOfAllFiles = 0

        allIncludedFiles = []
        timer.tic("Get sizes of files")
        for baseFileName in allBaseFileNames:
            files = np.array([f2 for f2 in sorted(os.listdir(rawDataFolder)) if
                              re.match(re.escape(baseFileName) + r'\d*\.(wav|hf)', f2)])
            files = filterFilesByFileNumber(files, baseFileName, removeFileNumbers=removeFileNumbers,
                                            onlyFileNumbers=onlyFileNumbers)
            if len(files) > 0:
                for fileName in files:
                    allIncludedFiles.append(str(fileName))
                    featureStorePath = os.path.join(featureDataFolder, fileName + ".h5")
                    with pd.HDFStore(featureStorePath, 'r') as featureStore:
                        # Get the number of rows and columns in this file
                        (thisFileRows, thisFileColumns) = featureStore['X'].shape
                        thisFileSamples = thisFileRows
                        if useMetadata:
                            thisFileColumns += len(metadataList)
                        # it better be the same as the rest
                        if totalXColumns is None:
                            totalXColumns = thisFileColumns
                        assert totalXColumns == thisFileColumns, "One of the datasets has a different number of columns"
                        # If using sequences get how many X rows we would get form this file
                        if makeSequences:
                            thisFileRows = getRowsForFileDueToSequence(thisFileRows,
                                                                       timestepsPerSequence,
                                                                       offsetBetweenSequences,
                                                                       fractionOfRandomToUse,
                                                                       padSequenceWithZeros)
                        # update the total Rows
                        setNames = getSetNameForFile(fileName, defaultSetName, fileNamesNumbersToSets)
                        for setName in setNames:
                            if setName in totalRowsInSetDict:
                                totalRowsInSetDict[setName] += thisFileRows
                                totalFilesInSetDict[setName] += 1
                            else:
                                totalRowsInSetDict[setName] = thisFileRows
                                totalFilesInSetDict[setName] = 1
                        # update base name rows
                        rowsPerFile[featureStorePath] = thisFileRows
                        if baseFileName in rowsPerBaseFileDict:
                            rowsPerBaseFileDict[baseFileName] += thisFileRows
                        else:
                            rowsPerBaseFileDict[baseFileName] = thisFileRows

                        # keep track of the biggest file
                        if thisFileRows > maxRowsOfAllFiles:
                            maxRowsOfAllFiles = thisFileRows
                        if thisFileSamples > maxSamplesOfAllFiles:
                            maxSamplesOfAllFiles = thisFileSamples

        timer.toc()
        if timestepsPerSequence is None or offsetBetweenSequences is None:
            datasetParameters['timestepsPerSequence'] = maxSamplesOfAllFiles
            timestepsPerSequence = maxSamplesOfAllFiles
            datasetParameters['offsetBetweenSequences'] = timestepsPerSequence

        # if we are making sequences the actual total X columns is multiplied by the timesteps
        if makeSequences:
            totalXColumns *= timestepsPerSequence
        # get the total y columns
        totalyColumns = getYColumnSize(datasetParameters)

        for (setName, rowsInSet) in totalRowsInSetDict.iteritems():
            setDict[setName] = [np.zeros((rowsInSet, totalXColumns)), np.zeros((rowsInSet, totalyColumns))]
            if rowPackagingStyle is not None:
                rowProcessingMetadataDict[setName] = np.zeros((rowsInSet, totalMetadataPackagingColumns))
        setsRowsProcessedTotal = {}

        for baseFileName in allBaseFileNames:
            files = np.array([f2 for f2 in sorted(os.listdir(rawDataFolder)) if
                              re.match(re.escape(baseFileName) + r'\d*\.(wav|hf)', f2)])
            files = filterFilesByFileNumber(files, baseFileName, removeFileNumbers=removeFileNumbers,
                                            onlyFileNumbers=onlyFileNumbers)
            if len(files) > 0:
                for fileName in files:
                    featureStorePath = os.path.join(featureDataFolder, fileName + ".h5")
                    (fXTemp,
                     fYTemp,
                     outputLabelsFinal,
                     rowPackagingMetadata,
                     rowPackagingMetadataFinal) = getXYTempAndLabelsFromFile(featureStorePath=featureStorePath,
                                                                             datasetParameters=datasetParameters,
                                                                             imageShape=imageShape,
                                                                             useMetadata=useMetadata,
                                                                             metadataList=metadataList,
                                                                             yValueType=yValueType,
                                                                             outputLabelsFinal=outputLabelsFinal,
                                                                             rowPackagingMetadataFinal=rowPackagingMetadataFinal)

                    setNames = getSetNameForFile(fileName, defaultSetName, fileNamesNumbersToSets)
                    for setName in setNames:
                        if setName not in setsRowsProcessedTotal:
                            setsRowsProcessedTotal[setName] = 0
                        rowsProcessedTotal = setsRowsProcessedTotal[setName]
                        setDict[setName][0][rowsProcessedTotal:(rowsProcessedTotal + fXTemp.shape[0]), :] = fXTemp
                        setDict[setName][1][rowsProcessedTotal:(rowsProcessedTotal + fYTemp.shape[0]), :] = fYTemp
                        rowProcessingMetadataDict[setName][rowsProcessedTotal:(rowsProcessedTotal + fYTemp.shape[0]),
                        :] = rowPackagingMetadata
                        setsRowsProcessedTotal[setName] += fXTemp.shape[0]
                    gc.collect()
                gc.collect()

        timer.toc()  # extract features from files

        timer.tic("Filtering Sets")
        if filterFitSets is None:
            filterFitSets = [defaultSetName]
        (setDict, xScaleFactor, xBias, yScaleFactor, yBias) = finalFilteringSets(setDict,
                                                                                 datasetParameters,
                                                                                 filterFitSets,
                                                                                 yValueType,
                                                                                 totalXColumns,
                                                                                 totalyColumns)
        timer.toc()

        # repackage the sets for processing
        timer.tic("Repackaging sets")
        setDict, packagedRowsPerSetDict, kerasRowMultiplier = repackageSets(setDict, datasetParameters,
                                                                            rowProcessingMetadataDict)
        timer.toc()

        numberOfSamples = 0
        for setName, setValue in setDict.iteritems():
            numberOfSamples += setValue[0].shape[0]
        print("Total samples to work with {0}".format(numberOfSamples))
        if len(setFractions) > 0:
            timer.tic("Split sets apart to make new sets")
            if numberOfSamples > 0:
                for (fromSet, toSet, fractionOrTotalrows) in setFractions:
                    if fractionOrTotalrows < 1.0:
                        fromSetEnd = int(totalRowsInSetDict[fromSet] * fractionOrTotalrows)
                    else:
                        fromSetEnd = fractionOrTotalrows
                    assert fromSetEnd < setDict[0][fromSet].shape[0], \
                        "Tried to get too many rows from set {fromSet}".format(fromSet=fromSet)
                    (setDict[toSet][0], setDict[fromSet][0]) = np.split(setDict[fromSet][0], [fromSetEnd])
                    (setDict[toSet][1], setDict[fromSet][1]) = np.split(setDict[fromSet][1], [fromSetEnd])
                    totalFilesInSetDict[toSet] = totalFilesInSetDict[fromSet]
            timer.toc()

        for setName, setValue in setDict.iteritems():
            print("{setName} set has {samples} samples".format(setName=setName, samples=setValue[0].shape[0]))
        print("Features per set {0}".format(totalXColumns))
        print("Output Dim per set {0}".format(totalyColumns))

        timer.tic("Save datasets and labels")
        with pd.HDFStore(datasetFile, 'w') as datasetStore:
            for setName, setValue in setDict.iteritems():
                datasetStore[setName + '_set_x'] = pd.DataFrame(setDict[setName][0])
                datasetStore[setName + '_set_y'] = pd.DataFrame(setDict[setName][1])
            datasetStore['labels'] = pd.DataFrame(outputLabelsFinal)
        timer.toc()

        datasetParametersToDump = dict(datasetParameters)
        datasetParametersToDump['allIncludedFiles'] = allIncludedFiles
        datasetParametersToDump['filesInSets'] = totalFilesInSetDict
        datasetParametersToDump['packagedRowsPerSetDict'] = packagedRowsPerSetDict
        datasetParametersToDump['kerasRowMultiplier'] = kerasRowMultiplier
        datasetParametersToDump['totalXColumns'] = totalXColumns
        datasetParametersToDump['totalyColumns'] = totalyColumns
        datasetParametersToDump['xScaleFactor'] = xScaleFactor
        datasetParametersToDump['xBias'] = xBias
        datasetParametersToDump['y value parameters']['yScaleFactor'] = yScaleFactor
        datasetParametersToDump['y value parameters']['yBias'] = yBias
        configFileName = os.path.join(processedDataFolder, "dataset parameters.yaml")
        with open(configFileName, 'w') as myDatasetConfigFile:
            yaml.dump(datasetParametersToDump, myDatasetConfigFile, default_flow_style=False, width=1000)
        timer.toc()  # overall time for whole function
        print ("######## Make Dataset Complete ############")


class DictDiffer(object):
    """
    Calculate the difference between two dictionaries as:
    (1) items added
    (2) items removed
    (3) keys same in both but changed values
    (4) keys same in both and unchanged values
    """

    def __init__(self, current_dict, past_dict):
        self.current_dict, self.past_dict = current_dict, past_dict
        self.set_current, self.set_past = set(current_dict.keys()), set(past_dict.keys())
        self.intersect = self.set_current.intersection(self.set_past)

    def added(self):
        return self.set_current - self.intersect

    def removed(self):
        return self.set_past - self.intersect

    def changed(self):
        outSet = set()
        for o in self.intersect:
            if type(self.past_dict[o]).__module__ == np.__name__ or type(
                    self.current_dict[o]).__module__ == np.__name__:
                if np.array((self.past_dict[o] != self.current_dict[o])).any():
                    outSet.add(o)
            elif self.past_dict[o] != self.current_dict[o]:
                outSet.add(o)
        # return set(o for o in self.intersect if self.past_dict[o] != self.current_dict[o])
        return outSet

    def unchanged(self):
        outSet = set()
        for o in self.intersect:
            if type(self.past_dict[o]).__module__ == np.__name__ or type(
                    self.current_dict[o]).__module__ == np.__name__:
                if np.array((self.past_dict[o] == self.current_dict[o])).all():
                    outSet.add(o)
            elif self.past_dict[o] == self.current_dict[o]:
                outSet.add(o)
        # return set(o for o in self.intersect if self.past_dict[o] == self.current_dict[o])
        return outSet

    def printAll(self):
        return """Added:     {added}
Removed:   {removed}
Changed:   {changed}
Unchanged: {unchanged}""".format(added=self.added(), removed=self.removed(), changed=self.changed(),
                                 unchanged=self.unchanged())

    def printAllDiff(self):
        return """Added:     {added}
Removed:   {removed}
Changed:   {changed}""".format(added=self.added(), removed=self.removed(), changed=self.changed(),
                               unchanged=self.unchanged())


# *********************************************************************************************************************
# ******************************************************MAIN START*****************************************************
# *********************************************************************************************************************
if __name__ == '__main__':
    # per run variables
    forceRefreshStatsDefault = False
    showFiguresDefault = False  # watch out this reduces the number of time windows to 1!!!!

    rawDataFolderMain = r"E:\\Users\\Joey\Documents\\Virtual Box Shared Folder\\"
    featureSetNameMain = 'DefaultPatchFeatures'
    datasetNameMain = 'staticLocations'

    featureDataFolderMain = os.path.join(rawDataFolderMain, "Processed Data Features", featureSetNameMain)
    featureConfigFileName = os.path.join(featureDataFolderMain, "feature parameters.yaml")
    with open(featureConfigFileName, 'r') as myConfigFile:
        featureParametersDefault = yaml.load(myConfigFile)

    processedDataFolderMain = os.path.join(rawDataFolderMain, "Processed Data Datasets", datasetNameMain)
    datasetConfigFileName = os.path.join(processedDataFolderMain, "dataset parameters.yaml")
    with open(datasetConfigFileName, 'r') as myConfigFile:
        datasetParametersDefault = yaml.load(myConfigFile)

    buildFeatures(featureParametersDefault, forceRefreshFeatures=forceRefreshStatsDefault,
                  showFigures=showFiguresDefault)
    buildDataSet(datasetParametersDefault, featureParametersDefault)
