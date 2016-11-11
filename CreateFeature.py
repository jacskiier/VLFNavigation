# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:29:50 2015

@author: jacsk
"""
import os
import re
import gc
import yaml

import numpy as np
import numpy.random

import matplotlib.pylab as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm

import wave
import struct

import scipy.stats
import scipy.io.wavfile
import scipy.signal
import scipy.interpolate

import gpxpy
import gpxpy.gpx

import pandas as pd
from tqdm import tqdm

from python_speech_features import mfcc

import zipfile
import datetime
import dateutil
import dateutil.parser
import pytz

import tictoc
import CreateUtils

timer = tictoc.tictoc()

gpxpy.gpx.DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'


def everyOther(v, offset=0):
    return [v[i] for i in range(offset, len(v), 2)]


def everyOtherX(v, x, offset=0):
    return [v[i] for i in range(offset, len(v), x)]


def find_nearestValue(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearestIndex(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearestIndexValue(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


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
                                            noverlap=2 ** (fftPowerArg - 1), cmap=cm.get_cmap("gist_heat"))

        # plt.pcolormesh(bins,freqs,10*np.log10(Pxx),cmap=cm.get_cmap("gist_heat"))
        # plt.imshow(10*np.log10(Pxx),interpolation = 'nearest',cmap = cm.get_cmap("gist_heat"),aspect = 'auto')
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
                plt.imshow(10 * np.log10(PxxWindow), interpolation='nearest', cmap=cm.get_cmap("gist_heat"),
                           aspect='auto')

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
                    plt.imshow(10 * np.log10(PxxSubWindow), interpolation='nearest', cmap=cm.get_cmap("gist_heat"),
                               aspect='auto')

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
                plt.imshow(statArray[0][:].reshape((NFArg, NTArg)), interpolation='nearest', cmap=cm.get_cmap("gist_heat"),
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


def collectRawAmplitudeFromWAVFile(filePatherArg,
                                   windowTimeLength,
                                   poolingSize=None,
                                   poolingType=None,
                                   maxTime=None):
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

    # window out the data
    samplesPerWindow = int(windowTimeLength * float(samplingRate * poolingSize))
    samplesInFinal = int(samplesPerWindow * np.floor(actualDataY.shape[0] / samplesPerWindow))
    actualDataY = np.reshape(actualDataY[:samplesInFinal], (int(actualDataY.shape[0] / samplesPerWindow), samplesPerWindow))

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

    totalMetadataColumns = len(CreateUtils.allMetadataNamesList)
    zeroFillArray = np.zeros((numOfSamples, totalMetadataColumns)).transpose()
    finalMetadataArray = np.core.records.fromarrays(zeroFillArray, dtype=CreateUtils.allMetadataDtyesList)

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
    namesAll = [CreateUtils.cadenceNames, CreateUtils.speedNames,
                CreateUtils.locationWahooNames, CreateUtils.heartNames,
                CreateUtils.footpadNames, CreateUtils.maNames,
                CreateUtils.accelerometerNames, CreateUtils.gyroscopeNames,
                CreateUtils.locationiPhoneNames, CreateUtils.headingNames]
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

def buildFeatures(featureParameters, forceRefreshFeatures=False, showFigures=False, maxFiles=10000,
                  allBaseFileNames=None, removeFileNumbers=(), onlyFileNumbers=()):
    processedFilesCount = 0
    rawDataFolder = CreateUtils.convertPathToThisOS(featureParameters['rawDataFolder'])
    featureDataFolder = CreateUtils.convertPathToThisOS(featureParameters['featureDataFolder'])
    if allBaseFileNames is None:
        allBaseFileNames = CreateUtils.getAllBaseFileNames(rawDataFolder)
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
        files = CreateUtils.filterFilesByFileNumber(files, baseFileName, removeFileNumbers=removeFileNumbers,
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
                                                               windowTimeLength,
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


################################
# Main Start           #########
################################
def runMain():
    ################################
    # Run Parameters Begin ########
    ################################
    runNow = False
    overwriteConfigFile = True
    forceRefreshFeatures = True  # rewrite existing files
    rebuildFromConfig = True  # rebuild current feature name
    rebuildAllFromConfig = False  # rebuild all current features
    removeFileNumbers = {}
    onlyFileNumbers = {}
    removeFeatureSetNames = []
    maxFiles = 100

    featureSetName = 'RawAmplitude'
    featureMethod = 'RawAmplitude'
    signalSource = 'Loop Antenna With iPhone 4'
    # signalSource = '3-Axis Dipole With SRI Receiver'
    samplingRate = None

    if os.name == 'nt':
        if featureMethod in CreateUtils.featureMethodNamesRebuildValid:
            if signalSource == 'Loop Antenna With iPhone 4':
                rawDataFolder = r"E:\\Users\\Joey\Documents\\Virtual Box Shared Folder\\"
                samplingRate = 44100
            elif signalSource == '3-Axis Dipole With SRI Receiver':
                rawDataFolder = r"E:\\Users\\Joey\\Documents\\DataFolder\\"
                samplingRate = 200000
            else:
                raise ValueError("Signal Source of {0} is not supported".format(signalSource))
        elif featureMethod == 'MNIST':
            rawDataFolder = r"E:\\Users\\Joey\\Documents\\Python Scripts\\Spyder\\deeplearningfiles\\mnist raw data folder\\"
        elif featureMethod == 'THoR':
            rawDataFolder = r"E:\\Users\\Joey\\Documents\\Python Scripts\\parse NHL\\"
        else:  # featureMethod == "Test"
            rawDataFolder = r"E:\\Users\\Joey\\Documents\\Python Scripts\\Spyder\\deeplearningfiles\\test raw data folder\\"
    elif os.name == 'posix':
        if featureMethod in CreateUtils.featureMethodNamesRebuildValid:
            if signalSource == 'Loop Antenna With iPhone 4':
                rawDataFolder = r"/media/sena/Greed Island/Users/Joey/Documents/Virtual Box Shared Folder/"
                samplingRate = 44100
            elif signalSource == '3-Axis Dipole With SRI Receiver':
                rawDataFolder = r"/media/sena/Greed Island/Users/Joey/Documents/DataFolder/"
                samplingRate = 200000
            else:
                raise ValueError("Signal Source of {0} is not supported".format(signalSource))
        elif featureMethod == 'MNIST':
            rawDataFolder = r"/media/sena/Greed Island/Users/Joey/Documents/Python Scripts/Spyder/deeplearningfiles/mnist raw data folder/"
        elif featureMethod == 'THoR':
            rawDataFolder = r"/media/sena/Greed Island/Users/Joey/Documents/Python Scripts/parse NHL/"
        else:  # featureMethod == "Test"
            rawDataFolder = r"/media/sena/Greed Island/Users/Joey/Documents/Python Scripts/Spyder/deeplearningfiles/test raw data folder/"
    else:
        raise ValueError("This OS is not allowed")

    ##################################
    # Feature specific parameters ###
    ##################################

    featureParametersDefault = {}
    featureDataFolder = os.path.join(rawDataFolder, "Processed Data Features", featureSetName)
    configFileName = os.path.join(featureDataFolder, "feature parameters.yaml")
    if (not os.path.exists(configFileName)) or overwriteConfigFile:
        if featureMethod == 'Patch':
            # x matrix parameters
            fftPower = 10
            windowTimeLength = 0.1  # in seconds
            burstMethod = False
            windowFreqBounds = [0, 22500]  # the start and stop frequencies in Hz
            # patch specific
            NT = 1  # blocks in time of each patch
            NF = 50  # blocks in frequency of each patch
            useOverallWindowStat = False  # should the last set of stat features be the whole window
            # mean variance, standard deviation, skewness kurtosis
            statOrder = [1, 3, 4, 0]
            normalizeStats = True  # done on a patch by patch basis

            # calculated stats
            totalStats = len(statOrder)  # how many stats to use from each patch
            NTF = NT * NF  # total blocks in a patch
            totalFeatures = (NT * NF + useOverallWindowStat) * totalStats  # length of an fP row

            # shape of input features as C style reshaping
            imageShape = (NF, NT, totalStats)
            # image shape order says the order you have to shuffle to get the image shape as (channels,width,height)
            imageShapeOrder = (2, 0, 1)

            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'feature parameters': {
                    'featureMethod': featureMethod,
                    'fftPower': fftPower,
                    'windowTimeLength': windowTimeLength,
                    'burstMethod': burstMethod,
                    'windowFreqBounds': windowFreqBounds,
                    'NT': NT,
                    'NF': NF,
                    'totalStats': totalStats,
                    'NTF': NTF,
                    'totalFeatures': totalFeatures,
                    'useOverallWindowStat': useOverallWindowStat,
                    'statOrder': statOrder,
                    'normalizeStats': normalizeStats,
                },
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }
        elif featureMethod == 'Covariance':

            # x matrix parameters
            fftPower = 5
            windowTimeLength = 1.0  # in seconds
            burstMethod = False
            windowFreqBounds = [0, 22500]  # the start and stop frequencies in Hz

            imageShapeOrder = (0, 1, 2)
            imageShape = (1, 2 ** (fftPower - 1) + 1, 2 ** (fftPower - 1) + 1)

            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'feature parameters': {
                    'featureMethod': featureMethod,
                    'fftPower': fftPower,
                    'windowTimeLength': windowTimeLength,
                    'burstMethod': burstMethod,
                    'windowFreqBounds': windowFreqBounds,
                },
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }
        elif featureMethod == 'MFCC':

            fftPower = 5

            winlen = 0.1
            winstep = 0.01
            numcep = 100
            nfilt = numcep * 2
            lowfreq = 0
            highfreq = 6000  # use none to get samplingRate/2
            windowFreqBounds = (lowfreq, highfreq)

            convertToInt = True
            convertToIntMax = 10000
            convertToIntMultiplier = None

            imageShapeOrder = (0, 1, 2)
            imageShape = (1, numcep, 1)

            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'feature parameters': {
                    'featureMethod': featureMethod,
                    'fftPower': fftPower,
                    'windowTimeLength': winlen,
                    'winstep': winstep,
                    'numcep': numcep,
                    'nfilt': nfilt,
                    'windowFreqBounds': windowFreqBounds,
                    'convertToInt': convertToInt,
                    'convertToIntMax': convertToIntMax,
                    'convertToIntMultiplier': convertToIntMultiplier,
                },
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }
        elif featureMethod == 'FFT':
            # x matrix parameters
            fftPower = 10
            burstMethod = False
            windowFreqBounds = [0, 22500]  # the start and stop frequencies in Hz

            # Determine the frequencies for this run so I can make the output shape for this frequency
            freqs = np.fft.fftfreq(2 ** fftPower, 1.0 / samplingRate)

            windowFreqIndex = [find_nearestIndex(freqs, windowFreqBounds[0]),
                               find_nearestIndex(freqs, windowFreqBounds[1])]

            windowTimeLength = (2 ** fftPower * (1.0 / samplingRate)) / 2.0
            # in seconds ## OK technically the window length isn't /2 but I'm really looking for window split which is this

            imageShape = (1 + int(windowFreqIndex[1] - windowFreqIndex[0]),)
            imageShapeOrder = (0,)

            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'feature parameters': {
                    'featureMethod': featureMethod,
                    'fftPower': fftPower,
                    'windowTimeLength': windowTimeLength,
                    'burstMethod': burstMethod,
                    'windowFreqBounds': windowFreqBounds,
                },
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }
        elif featureMethod == 'RawAmplitude':
            poolingSize = 1
            poolingType = None

            if poolingType == 'none' or poolingType is None:
                poolingSize = 1
            samplesPerWindow = int(samplingRate / 10.0)
            windowTimeLength = samplesPerWindow / float(samplingRate * poolingSize)

            maxTime = 5 * 60  # in seconds

            imageShape = (samplesPerWindow,)
            imageShapeOrder = (0,)

            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'feature parameters': {
                    'samplesPerWindow': samplesPerWindow,
                    'featureMethod': featureMethod,
                    'windowTimeLength': windowTimeLength,
                    'poolingSize': poolingSize,
                    'poolingType': poolingType,
                    'maxTime': maxTime,
                },
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }
        elif featureMethod == 'MNIST':
            imageShapeOrder = (0, 1, 2)
            imageShape = (1, 28, 28)
            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }
        elif featureMethod == "THoR":
            imageShapeOrder = (0, 1, 2)
            imageShape = (1, 10000, 1)
            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }
        elif featureMethod == "Test":
            imageShapeOrder = (0, 1, 2)
            imageShape = (1, 1, 2)
            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }

        ################################
        # Parameters End   ############
        ################################

        if not os.path.exists(featureDataFolder):
            os.makedirs(featureDataFolder)
        doesFileAlreadyExist = os.path.exists(configFileName)
        if not overwriteConfigFile:
            assert not doesFileAlreadyExist, 'do you want to overwirte the config file?'
        with open(configFileName, 'w') as myConfigFile:
            print("Set Name: {0}".format(featureSetName))
            yaml.dump(featureParametersDefault, myConfigFile, default_flow_style=False, width=1000)
            if doesFileAlreadyExist:
                print("Overwrote Feature config file {0}".format(configFileName))
            else:
                print("Wrote Feature config file {0}".format(configFileName))
        featureSets = [featureSetName]
    else:
        if rebuildAllFromConfig:
            featureDataFolderRoot = os.path.join(rawDataFolder, "Processed Data Features")
            featureSets = [fileIterator for fileIterator in os.listdir(featureDataFolderRoot) if
                           os.path.isdir(os.path.join(featureDataFolderRoot,
                                                      fileIterator)) and fileIterator not in removeFeatureSetNames]
        elif rebuildFromConfig:
            featureSets = [featureSetName]
        else:
            featureSets = []

    if runNow:
        for featureSetFolder in featureSets:
            featureConfigFileName = os.path.join(rawDataFolder, "Processed Data Features", featureSetFolder,
                                                 "feature parameters.yaml")
            with open(featureConfigFileName, 'r') as myConfigFile:
                featureParametersDefault = yaml.load(myConfigFile)
            # didProcessAllFiles = False
            # while not didProcessAllFiles:
            didProcessAllFiles = buildFeatures(featureParametersDefault,
                                               forceRefreshFeatures=forceRefreshFeatures, maxFiles=maxFiles,
                                               removeFileNumbers=removeFileNumbers,
                                               onlyFileNumbers=onlyFileNumbers)
            print ("Did process all files? {0}".format(didProcessAllFiles))


if __name__ == '__main__':
    runMain()
