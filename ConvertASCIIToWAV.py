import os
import re
import scipy.io.wavfile
import numpy as np
import dateutil
import dateutil.parser
import matplotlib.dates
import datetime

from scipy.interpolate import interp1d
import matplotlib.pylab as plt

import gpxpy
import gpxpy.gpx
import pandas as pd
from tqdm import tqdm

gpxpy.gpx.DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'

import geocoder


def convertASCIIToWAVandGPX(dataCollectFolder, dataCollectBaseName, rawDataFolder, maxRows=None, showFigures=False, windowSize=0.05,
                            forceSampleRate=None, makeAndReadHF=True, saveToWAV=True, skipHighSpeedFile=False):
    if not skipHighSpeedFile:
        highSpeedFile = os.path.join(dataCollectFolder, dataCollectBaseName + '.hsa')
        highSpeedFilehf = os.path.join(dataCollectFolder, dataCollectBaseName + '.hf')

        if os.path.exists(highSpeedFilehf) and makeAndReadHF:
            print("Reading {0} into array from hf file".format(highSpeedFilehf))
            with pd.HDFStore(highSpeedFilehf, 'r') as featureStore:
                highSpeedArray = featureStore['hsa'].as_matrix()
            highSpeedArrayValues = highSpeedArray[:, 1:4]
            highSpeedArrayDates = highSpeedArray[:, 0]
        else:
            print("Reading {0} into array from hsa file using genfromtxt".format(highSpeedFile))
            # num_linesHighSpeedFile = sum(1 for line in open(highSpeedFile))
            # print('number of lines in high speed = {0}'.format(num_linesHighSpeedFile))
            highSpeedArray = np.genfromtxt(highSpeedFile, converters={0: dateutil.parser.parse}, delimiter='\t', max_rows=maxRows)
            highSpeedArrayDates = highSpeedArray['f0']
            highSpeedArrayValues = highSpeedArray[['f1', 'f2', 'f3']].view(np.float64).reshape(highSpeedArray[['f1', 'f2', 'f3']].shape + (-1,))

            if makeAndReadHF:
                print("Writing hsa array to hf file {0}".format(highSpeedFilehf))
                with pd.HDFStore(highSpeedFilehf, 'a') as datasetStore:
                    datasetStore['hsa'] = pd.DataFrame(highSpeedArray)

        samplePeriod = (highSpeedArrayDates[1] - highSpeedArrayDates[0]).total_seconds()
        sampleRate = int(np.round((1 / samplePeriod)))
        maxValue = np.max(highSpeedArrayValues, 0)
        print('Dataset starts at {0}'.format(highSpeedArrayDates[0]))
        print('Dataset elapsed Time is {0}'.format(highSpeedArrayDates[-1] - highSpeedArrayDates[0]))
        print('Measured sampling rate is {0}. Largest Value on each channel is {1}'.format(sampleRate, maxValue))

        if forceSampleRate is not None:
            sampleRate = forceSampleRate
            samplePeriod = 1.0 / sampleRate

        print("Making Gap Indices arrays")
        gapIndices = np.where(np.diff(highSpeedArrayDates) > datetime.timedelta(seconds=windowSize * 2))[0]
        gapIndicesStarts = gapIndices[:-1] + 1
        gapIndicesEnds = gapIndices[1:] + 1
        gapStartAndStop = np.vstack((gapIndicesStarts, gapIndicesEnds)).transpose()

        print("Making Time diff and time ranges arrays")
        totalTimeDifference = (highSpeedArrayDates[-1] - highSpeedArrayDates[0]).total_seconds()
        timeRanges = np.array([highSpeedArrayDates[0] + datetime.timedelta(microseconds=i * samplePeriod * 1e6) for i in
                               xrange(1 + int(totalTimeDifference * sampleRate))])

        if showFigures:
            plt.figure(1)
            plt.plot([b.total_seconds() for b in np.diff(highSpeedArrayDates)])
            plt.title("Time Difference Between Samples")
            plt.xlabel("Sample")
            plt.ylabel("Time (s)")
            plt.show()

        def convertToSecondsSince(datetimeArray, startTime):
            return [(thisdatetime - startTime).total_seconds() for thisdatetime in datetimeArray]

        print("Making In Seconds Arrays")
        highSpeedArrayDatesInSeconds = np.array(convertToSecondsSince(highSpeedArrayDates, highSpeedArrayDates[0]), dtype=np.float64)
        timeRangesInSeconds = np.arange(start=0, stop=timeRanges.shape[0] * samplePeriod, step=samplePeriod, dtype=np.float64)

        burstStartTimes = np.zeros((gapStartAndStop.shape[0],))
        newValues = np.zeros((timeRanges.shape[0], highSpeedArrayValues.shape[1]))
        timeRangesGapIndices = np.zeros((gapStartAndStop.shape[0], 2))
        for dim in range(highSpeedArrayValues.shape[1]):
            for burstNumber in tqdm(range(gapStartAndStop.shape[0]), "Channel {0}".format(dim)):
                burstStartIndex = gapStartAndStop[burstNumber, 0]
                burstEndIndex = gapStartAndStop[burstNumber, 1]
                gapFiller = interp1d(highSpeedArrayDatesInSeconds[burstStartIndex:burstEndIndex],
                                     highSpeedArrayValues[burstStartIndex:burstEndIndex, dim], kind='linear', assume_sorted=True)
                timeRangeStartIndex = np.argmin(np.abs(timeRangesInSeconds - highSpeedArrayDatesInSeconds[burstStartIndex])) + 1
                timeRangeEndIndex = np.argmin(np.abs(timeRangesInSeconds - highSpeedArrayDatesInSeconds[burstEndIndex - 1])) - 1
                burstStartTimes[burstNumber] = timeRangesInSeconds[timeRangeStartIndex]
                newValues[timeRangeStartIndex:timeRangeEndIndex, dim] = gapFiller(timeRangesInSeconds[timeRangeStartIndex:timeRangeEndIndex])
                timeRangesGapIndices[burstNumber, :] = np.array([timeRangeStartIndex, timeRangeEndIndex])

        firstBurstStart = np.argmin(np.abs(timeRangesInSeconds - highSpeedArrayDatesInSeconds[gapStartAndStop[0, 0]])) + 1
        lastBurstEnd = np.argmin(np.abs(timeRangesInSeconds - highSpeedArrayDatesInSeconds[gapStartAndStop[-1, 1] - 1])) - 1
        highSpeedArrayValues = newValues[firstBurstStart:lastBurstEnd, :]
        timeRangesInSeconds = timeRangesInSeconds[firstBurstStart:lastBurstEnd]
        # wavFileStartTime = timeRanges[gapStartAndStop[0, 0]]
        wavFileStartTime = timeRanges[int(timeRangesGapIndices[0, 0]), 0]

        if showFigures:
            plt.figure(2)
            plt.plot(timeRangesInSeconds, highSpeedArrayValues)
            # plt.xticks(np.arange(min(timeRangesInSeconds), max(timeRangesInSeconds)+1, 0.001))
            plt.title("Actual Signal at Constant Sampling Frequency")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude ")
            plt.show()

            plt.figure(3)
            plt.plot(np.diff(burstStartTimes))
            plt.title("Difference between Start of each burst")
            plt.xlabel("Burst Number Diff")
            plt.ylabel("Time Between Burst Starts (burst period)")
            plt.show()

        if saveToWAV:
            highSpeedWAVFile = os.path.join(rawDataFolder, dataCollectBaseName + '.wav')
            print("Writing WAV file {0}".format(highSpeedWAVFile))
            scipy.io.wavfile.write(highSpeedWAVFile, sampleRate, highSpeedArrayValues)
        else:
            highSpeedHFFile = os.path.join(rawDataFolder, dataCollectBaseName + '.hf')
            print("Writing HF file {0}".format(highSpeedHFFile))
            with pd.HDFStore(highSpeedHFFile, 'w') as datasetStore:
                datasetStore['hsav'] = pd.DataFrame(highSpeedArrayValues)
                datasetStore['samplingRate'] = pd.DataFrame(np.array([sampleRate]))
                datasetStore['timestamps'] = pd.DataFrame(timeRanges)
                datasetStore['gapStartStopIndices'] = pd.DataFrame(timeRangesGapIndices)
    else:
        highSpeedHFFile = os.path.join(rawDataFolder, dataCollectBaseName + '.hf')
        print("Reading HF file {0}".format(highSpeedHFFile))
        with pd.HDFStore(highSpeedHFFile, 'r') as datasetStore:
            timeRanges = datasetStore['timestamps'].as_matrix()
            timeRangesGapIndices = datasetStore['gapStartStopIndices'].as_matrix()
            wavFileStartTime = timeRanges[int(timeRangesGapIndices[0, 0]), 0]
            wavFileStartTime = datetime.datetime.utcfromtimestamp((wavFileStartTime - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))

    gpsFile = os.path.join(dataCollectFolder, dataCollectBaseName + '.gps')
    # num_linesGPSFile = sum(1 for line in open(gpsFile))
    # print('number of lines in high speed = {0}'.format(num_linesGPSFile))

    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    print("Building GPX object from {0}".format(gpsFile))
    firstTime = True
    with open(gpsFile, 'r') as gpsFileObject:
        for line in tqdm(gpsFileObject.readlines(), "Lines in GPS Message File"):
            initialDatePart = line[:23]
            gprmcMessage = line[26:]
            gprmcSplit = gprmcMessage.split(',')

            timeStamp = dateutil.parser.parse(initialDatePart)
            if timeStamp > wavFileStartTime:
                latMatcher = re.match('''(?P<degrees>\d*)(?P<minutes>\d{2})\.?(?P<decimalMinutes>\d{2,4})''', gprmcSplit[3])
                lonMatcher = re.match('''(?P<degrees>\d*)(?P<minutes>\d{2})\.?(?P<decimalMinutes>\d{2,4})''', gprmcSplit[5])
                if latMatcher is None or lonMatcher is None:
                    print ("Bad Message {0}".format(gprmcMessage))
                    continue
                lat = (float(lonMatcher.group('degrees')) + float(latMatcher.group("minutes") + "." + latMatcher.group("decimalMinutes")) / 60.0) * (
                -1 if gprmcSplit[4] == 'S' else 1)
                lon = (float(lonMatcher.group('degrees')) + float(lonMatcher.group("minutes") + "." + lonMatcher.group("decimalMinutes")) / 60.0) * (
                -1 if gprmcSplit[6] == 'W' else 1)

                # lat = (float(gprmcSplit[3][:2]) + float(gprmcSplit[3][2:])/60) * (-1 if gprmcSplit[4] == 'S' else 1)
                # lon = (float(gprmcSplit[5][:3]) + float(gprmcSplit[5][3:])/60) * (-1 if gprmcSplit[6] == 'W' else 1)
                # # clip them back to real bounds if they are bogus....
                # lat %= 90
                # lon = ((lon + 180) % 360) - 180

                # We will bail if they are bogus
                if lat < 0 or lat > 90 or lon < -180 or lon > 180:
                    print("Message {0}".format(gprmcMessage))
                    print(" Bad latitude {lat} or longitude {lon}".format(lat=lat, lon=lon))
                    continue

                ele = geocoder.google([lat, lon], method='elevation').meters
                coordDict = {
                    'latitude': lat,
                    'longitude': lon,
                    "altitude": ele,
                    "speed": float(gprmcSplit[7]),
                    "horizontalAccuracy": None,
                    "verticalAccuracy": None
                }

                if firstTime:
                    print('GPX file starts at {0}'.format(timeStamp))
                    firstTime = False
                    gps_trackPoint = gpxpy.gpx.GPXTrackPoint(float(coordDict["latitude"]), float(coordDict["longitude"]),
                                                             elevation=coordDict["altitude"], time=wavFileStartTime,
                                                             horizontal_dilution=coordDict["horizontalAccuracy"],
                                                             vertical_dilution=coordDict["verticalAccuracy"],
                                                             speed=coordDict["speed"])
                    gpx_segment.points.append(gps_trackPoint)

                gps_trackPoint = gpxpy.gpx.GPXTrackPoint(float(coordDict["latitude"]), float(coordDict["longitude"]), elevation=coordDict["altitude"],
                                                         time=timeStamp,
                                                         horizontal_dilution=coordDict["horizontalAccuracy"],
                                                         vertical_dilution=coordDict["verticalAccuracy"], speed=coordDict["speed"])
                gpx_segment.points.append(gps_trackPoint)

    gpsFilePath = os.path.join(rawDataFolder, dataCollectBaseName + '.gpx')
    print("Writing GPX file to {0}".format(gpsFilePath))
    with open(gpsFilePath, 'w') as f:
        f.write(gpx.to_xml())


if __name__ == '__main__':
    if os.name == 'nt':
        rawDataFolderMain = os.path.join("E:", "Users", "Joey", "Documents", "DataFolder")  # 3 Axis VLF Antenna signals raw data folder
        dataCollectFolderMain = os.path.join("L:", "Thesis Files", "afitdata")
    elif os.name == 'posix':
        rawDataFolder = r"/media/sena/Greed Island/Users/Joey/Documents/DataFolder/"  # 3 Axis VLF Antenna signals raw data folder
        dataCollectFolderMain = os.path.join("media", "sena", "blue", "Thesis Files", "afitdata")
    else:
        raise ValueError("This OS is not allowed")

    dataCollectBaseNames = ['afittest00000', 'cartest00000', 'cartest00001', 'hometest00000', 'hometest00001', 'hometest00002', 'hometest00003',
                            'carneighborhood00000', 'carneighborhood00001']
    dataCollectBaseNames = ['carneighborhood00003']

    for dataCollectBaseNameMain in dataCollectBaseNames:
        print("Starting base file {0}".format(dataCollectBaseNameMain))
        convertASCIIToWAVandGPX(dataCollectFolderMain, dataCollectBaseNameMain, rawDataFolderMain, maxRows=None, showFigures=False,
                                makeAndReadHF=False, saveToWAV=False, skipHighSpeedFile=True)
