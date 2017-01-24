import os
import re
import gc
import numpy as np
import numpy.random

import matplotlib.pylab as plt

import sklearn
from sklearn import decomposition
from sklearn.externals import joblib

from scipy.spatial.distance import cdist

import pandas as pd
from tqdm import tqdm

from CreateFeature import getFileStatisticsOfFile
from CreateFeature import buildFeatures

import CreateUtils
import CoordinateTransforms
import tictoc

timer = tictoc.tictoc()

doScaryShuffle = True


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
    if yValueType in CreateUtils.yValueContinuousTypes:
        if yValueType in CreateUtils.yValueGPSTypes:
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
    fYTempArg = gridArg - gridSize / 2.0
    return fYTempArg


def sliceUpArray(arrayToSlice, startSample, endSample, flipCurrentSlice):
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


def getXYTempAndLabelsFromFile(featureStorePath,
                               datasetParameters,
                               imageShape,
                               useMetadata,
                               metadataList,
                               yValueType,
                               outputLabelsFinal=None,
                               rowPackagingMetadataFinal=None):
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
    minForClassTransition = datasetParameters['minForClassTransition'] if 'minForClassTransition' in datasetParameters else 1

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
    if yValueType in CreateUtils.yValueGPSTypes:
        fYTemp = metadataFull[['LatitudeiPhone', 'LongitudeiPhone', 'AltitudeiPhone']].as_matrix()
        # turn Lat Lon into radians from degrees
        fYTemp[:, 0:2] = fYTemp[:, 0:2] * np.pi / 180.0
        ecefCoords = CoordinateTransforms.LlhToEcef(fYTemp)
        fYTemp = CoordinateTransforms.EcefToLocalLevel(localLevelOriginInECEF, ecefCoords)
        if not includeAltitude:
            fYTemp = fYTemp[:, 0:2]

        # yValueType specific
        if yValueType == 'gpsPolar':
            fYTemp[:, 0] = np.sqrt(
                np.multiply(fYTemp[:, 0], fYTemp[:, 0]) + np.multiply(fYTemp[:, 1], fYTemp[:, 1]))
            fYTemp[:, 1] = np.arctan2(fYTemp[:, 1], fYTemp[:, 0]) * 180.0 / np.pi
        if yValueType == 'gpsD':
            # force the Y output to be a number of a grid location
            fYTemp = deterineYValuesGridByGPSArrayInLocalLevelCoords(fYTemp, gridSize)
        if yValueType == 'particle':
            particleFilePath = datasetParameters['y value parameters']['particleFilePath']
            # the file should be in North x East with the same local level coord
            particleArray = np.genfromtxt(particleFilePath, delimiter=',', skip_header=1)
            distArray = cdist(fYTemp, particleArray, metric='euclidean')
            fYTemp = np.argmin(distArray, axis=1)
            fYTemp = particleArray[fYTemp, :]
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
    if yValueType in CreateUtils.yValueDiscreteTypes:
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
        # pull out variables for packing
        gridSizePackage = datasetParameters['gridSizePackage'] if 'gridSizePackage' in datasetParameters else (1, 1, 1)

        # calculate the coordinate positions
        gpsCoords = metadataFull[['LatitudeiPhone', 'LongitudeiPhone', 'AltitudeiPhone']].as_matrix()
        # turn Lat Lon into radians from degrees
        gpsCoords[:, 0:2] = gpsCoords[:, 0:2] * np.pi / 180.0
        ecefCoords = CoordinateTransforms.LlhToEcef(gpsCoords)
        localLevelCoords = CoordinateTransforms.EcefToLocalLevel(localLevelOriginInECEF, ecefCoords)
        localLevelCoords = localLevelCoords[:, 0:2]

        # make them discrete to a grid
        newGridCoordinates = deterineYValuesGridByGPSArrayInLocalLevelCoords(localLevelCoords, gridSizePackage)

        # turn the positions into class indices
        rowPackagingMetadata, rowPackagingMetadataFinal = getIndexOfLabels(newGridCoordinates,
                                                                           rowPackagingMetadataFinal)
        rowPackagingMetadata = rowPackagingMetadata[:, None]
    elif rowPackagingStyle == 'particle':
        # pull out variables for packing
        particlePackFilePath = datasetParameters['particlePackFilePath']
        # the file should be in North x East with the same local level coord
        particleArray = np.genfromtxt(particlePackFilePath, delimiter=',', skip_header=1)

        # calculate the coordinate positions
        gpsCoords = metadataFull[['LatitudeiPhone', 'LongitudeiPhone', 'AltitudeiPhone']].as_matrix()
        # turn Lat Lon into radians from degrees
        gpsCoords[:, 0:2] = gpsCoords[:, 0:2] * np.pi / 180.0
        ecefCoords = CoordinateTransforms.LlhToEcef(gpsCoords)
        localLevelCoords = CoordinateTransforms.EcefToLocalLevel(localLevelOriginInECEF, ecefCoords)
        localLevelCoords = localLevelCoords[:, 0:2]

        # make coords discrete to particles
        distArray = cdist(localLevelCoords, particleArray, metric='euclidean')
        newGridCoordinates = np.argmin(distArray, axis=1)
        newGridCoordinates = particleArray[newGridCoordinates, :]

        # turn the positions into class indices
        rowPackagingMetadata, rowPackagingMetadataFinal = getIndexOfLabels(newGridCoordinates, rowPackagingMetadataFinal)
        rowPackagingMetadata = rowPackagingMetadata[:, None]
    elif rowPackagingStyle in ['class', 'classWithClassTransitions']:
        # turn the positions into class indices
        rowPackagingMetadata, rowPackagingMetadataFinal = getIndexOfLabels(fYTemp, rowPackagingMetadataFinal)
        if rowPackagingStyle == 'classWithClassTransitions':
            currentMinLength = 1
            assert rowPackagingMetadata.size > minForClassTransition, "min is too big for data"
            while currentMinLength < minForClassTransition:
                indicesOfChanges = np.where(np.concatenate(([True], rowPackagingMetadata[:-1] != rowPackagingMetadata[1:], [True])))[0]
                lengthOfChanges = np.diff(indicesOfChanges)
                currentMin = np.min(lengthOfChanges)
                if currentMin > minForClassTransition:
                    break
                if currentMin > currentMinLength:
                    currentMinLength = currentMin

                indicesOfChanges = indicesOfChanges[:-1]
                startNumbers = rowPackagingMetadata[indicesOfChanges[:-1]]

                indicesOfThisSize = indicesOfChanges[lengthOfChanges == currentMinLength]
                startNumbersOfThisSizeIndices = np.where(lengthOfChanges == currentMinLength)[0]
                for i, indexOfThisSize in enumerate(indicesOfThisSize):
                    starter = startNumbers[startNumbersOfThisSizeIndices[i] - 1] if startNumbersOfThisSizeIndices[i] - 1 > 0 else startNumbers[0]
                    rowPackagingMetadata[indexOfThisSize:indexOfThisSize + currentMinLength] = starter

            indicesOfChanges = np.where(np.concatenate(([True], rowPackagingMetadata[:-1] != rowPackagingMetadata[1:], [True])))[0]
            lengthOfChanges = np.diff(indicesOfChanges)
            indicesOfChanges = indicesOfChanges[:-1]
            startNumbers = rowPackagingMetadata[indicesOfChanges[:-1]]
            endNumbers = rowPackagingMetadata[indicesOfChanges[1:]]

            transitionLabels = ["{0}->{1}".format(startClass, endClass) for (startClass, endClass) in zip(startNumbers, endNumbers)]

            startEndIndices = np.vstack((indicesOfChanges[:-2], indicesOfChanges[1:-1]))
            transitionIndices = np.array(np.floor(np.median(startEndIndices, axis=0)), dtype=np.int64)
            transitionStartEndLengths = np.vstack((lengthOfChanges[:-1], lengthOfChanges[1:]))
            transitionLengths = np.array(np.floor(np.mean(transitionStartEndLengths, axis=0)), dtype=np.int64)

            classTransitionMask = np.zeros_like(rowPackagingMetadata, dtype='|S8')
            classTransitionMask[:transitionIndices[0]] = transitionLabels[0]
            classTransitionMask[transitionIndices[-1]:] = transitionLabels[-1]
            for transitionIndex, transitionLength, transitionLabel in zip(transitionIndices, transitionLengths, transitionLabels):
                classTransitionMask[transitionIndex:transitionIndex + transitionLength] = transitionLabel
            classTransitionMask = classTransitionMask[:, None]
            rowPackagingMetadataTransitions, rowPackagingMetadataFinal = getIndexOfLabels(classTransitionMask, rowPackagingMetadataFinal)
            # Stack together both rowPackagingMetadata arrays
            rowPackagingMetadata = np.hstack((rowPackagingMetadata, rowPackagingMetadataTransitions))
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
            curSliceX = sliceUpArray(fXTemp, startSample, endSample, flipCurrentSlice)
            sequencefX[int(offsetIndex), 0:int(cutSize * fXTemp.shape[1])] = curSliceX

            # Slice up y
            curSliceY = sliceUpArray(fYTemp, startSample, endSample, flipCurrentSlice)
            sequencefY[int(offsetIndex), 0:int(cutSize * fYTemp.shape[1])] = curSliceY

            # Slice up rowPackagingMetadata
            curSliceRPM = sliceUpArray(rowPackagingMetadata, startSample, endSample, flipCurrentSlice)
            sequenceRowPackagingMetadata[int(offsetIndex), 0:int(cutSize * rowPackagingMetadata.shape[1])] = curSliceRPM

        if not timeDistributedY:
            sequencefY = sequencefY[:, 0]
        fXTemp = sequencefX
        fYTemp = sequencefY
        rowPackagingMetadata = sequenceRowPackagingMetadata
        # timer.toc()
    return fXTemp, fYTemp, outputLabelsFinal, rowPackagingMetadata, rowPackagingMetadataFinal


def finalFilteringSets(setDictArg, datasetParameters, filterFitSets, yValueType, totalXColumns, totalYColumns, storeFilterFile):
    shuffleFinalSamples = datasetParameters['shuffleFinalSamples'] if 'shuffleFinalSamples' in datasetParameters else False

    # Sequence Parameters
    makeSequences = datasetParameters['makeSequences'] if 'makeSequences' in datasetParameters else False
    timestepsPerSequence = datasetParameters['timestepsPerSequence'] if 'timestepsPerSequence' in datasetParameters else 100

    # Saved Filter parameters
    useSavedFilter = datasetParameters['useSavedFilter'] if 'useSavedFilter' in datasetParameters else False
    savedFilterFile = datasetParameters['savedFilterFile'] if 'savedFilterFile' in datasetParameters else ''
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

    if useSavedFilter:
        savedFilterDict = joblib.load(savedFilterFile)
        pca = savedFilterDict['pca']
        xBias = savedFilterDict['xBias']
        yBias = savedFilterDict['yBias']
        xScaleFactor = savedFilterDict['xScaleFactor']
        yScaleFactor = savedFilterDict['yScaleFactor']
    else:
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
        if yValueType in CreateUtils.yValueDiscreteTypes:
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
        if pca is not None:
            filterX = pca.transform(filterX)
        # reshape back to sequences
        if makeSequences:
            setValue[0] = np.reshape(filterX,
                                     (int(filterX.shape[0] / timestepsPerSequence),
                                      int(filterX.shape[1] * timestepsPerSequence)))
        else:
            setValue[0] = filterX

        savedFilterDict = {
            'pca': pca,
            'xBias': xBias,
            'yBias': yBias,
            'xScaleFactor': xScaleFactor,
            'yScaleFactor': yScaleFactor,
        }
        joblib.dump(savedFilterDict, storeFilterFile)

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
    shuffleSamplesAfterRepackaging = datasetParameters['shuffleSamplesAfterRepackaging'] if 'shuffleSamplesAfterRepackaging' in datasetParameters else False

    packagedRowsPerSetDict = {}
    packagedColumnsPerSetDict = {}
    trueRowsPerSetDict = {}
    trueColumnsPerSetDict = {}
    kerasRowMultiplier = 1
    if rowPackagingStyle is not None:
        newRowsMax = 0
        totalColumnCountMax = 0
        for setName, setValue in setDictArg.iteritems():
            rowMetadata = rowProcessingMetadataDictArg[setName]
            uniqueRows, uniqueCounts = np.unique(rowMetadata, return_counts=True)
            # filter out numbers less than 0
            uniqueCounts = uniqueCounts[uniqueRows > -1]
            uniqueRows = uniqueRows[uniqueRows > -1]
            # get the total row and column count for this set
            newRows = uniqueRows.shape[0]
            trueRowsPerSetDict[setName] = newRows
            newRowsMax = newRows if newRows > newRowsMax else newRowsMax
            totalColumnCount = int(np.max(uniqueCounts) if padRowPackageWithZeros else np.min(uniqueCounts))
            trueColumnsPerSetDict[setName] = totalColumnCount
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
            packagedColumnsPerSetDict[setName] = totalColumnCount
            rowMetadata = rowProcessingMetadataDictArg[setName]
            uniqueRows, uniqueCounts = np.unique(rowMetadata, return_counts=True)
            # filter out numbers less than 0
            uniqueCounts = uniqueCounts[uniqueRows > -1]
            uniqueRows = uniqueRows[uniqueRows > -1]
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
            for uniqueRowIndex in tqdm(range(uniqueRows.shape[0]), desc="Rows for Set Name: {setName}".format(setName=setName)):
                uniqueRow = uniqueRows[uniqueRowIndex]
                uniqueCount = uniqueCounts[uniqueRowIndex]
                for totalColumnsSet, setNumber in zip([totalColumnsX, totalColumnsY], [0, 1]):
                    tempSet = tempSets[setNumber]
                    sliceTo = min(uniqueCount, totalColumnCount)
                    setIndexer = np.mod(np.where(rowMetadata.squeeze() == uniqueRow)[0], setValue[setNumber].shape[0])
                    # now mask out the samples I want from setValue then flatten it and set it to the temp set
                    tempSet[uniqueRowIndex, :sliceTo * totalColumnsSet] = setValue[setNumber][setIndexer, :].flatten()[:sliceTo * totalColumnsSet]
                    if uniqueCount < totalColumnCount and (repeatRowPackageBeginningAtEnd or repeatRowPackageEndingAtEnd):
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
            if shuffleSamplesAfterRepackaging:
                timer.tic("Shuffle After Repackaging {0} Set".format(setName))
                if not doScaryShuffle:
                    (tempSets[0], tempSets[1]) = shuffle_in_unison_inplace(tempSets[0], tempSets[1])
                else:
                    shuffle_in_unison_scary(tempSets[0], tempSets[1])
                timer.toc()
            setValue[0] = tempSets[0]
            setValue[1] = tempSets[1]
    return setDictArg, trueRowsPerSetDict, trueColumnsPerSetDict, packagedRowsPerSetDict, packagedColumnsPerSetDict, kerasRowMultiplier


def buildDataSet(datasetParameters, featureParameters, forceRefreshDataset=False):
    # feature variables
    featureDataFolder = CreateUtils.getProcessedFeaturesFolder(featureParameters['featureSetName'])
    featureSetName = featureParameters['featureSetName']
    imageShape = featureParameters['imageShape']

    # general dataset variables
    rawDataFolder = CreateUtils.getRawDataFolder()
    datasetName = datasetParameters['datasetName']
    processedDataFolder = CreateUtils.getProcessedDataDatasetsFolder(datasetName=datasetName)

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

    datasetFile = CreateUtils.getDatasetFile(featureSetName=featureSetName, datasetName=datasetName, checkExistence=False)
    if not os.path.exists(datasetFile) or forceRefreshDataset:
        timer.tic("Build dataset {0} for feature set {1}".format(datasetName, featureSetName))
        timer.tic("Extract features from stored files")

        setDict = {}
        rowProcessingMetadataDict = {}

        allIncludedFiles = []

        totalXColumns = None
        totalMetadataPackagingColumns = 1
        metadataMultiplier = CreateUtils.metadataMultipliersFromRowPackagingStyle[rowPackagingStyle]

        rowPackagingMetadataFinal = []
        rowsPerFile = {}
        rowsPerBaseFileDict = {}
        totalRowsInSetDict = {}
        totalFilesInSetDict = {}
        maxRowsOfAllFiles = 0
        maxSamplesOfAllFiles = 0

        outputLabelsFinal = []
        if yValueType == 'particle':
            particleFilePath = datasetParameters['y value parameters']['particleFilePath']
            # the file should be in North x East with the same local level coord
            particleArray = np.genfromtxt(particleFilePath, delimiter=',', skip_header=1)
            labelIndices, outputLabelsFinal = getIndexOfLabels(particleArray, outputLabelsFinal)

        timer.tic("Get sizes of files")
        for baseFileName in allBaseFileNames:
            files = np.array([f2 for f2 in sorted(os.listdir(rawDataFolder)) if
                              re.match(re.escape(baseFileName) + r'\d*\.(wav|hf)', f2)])
            files = CreateUtils.filterFilesByFileNumber(files, baseFileName, removeFileNumbers=removeFileNumbers, onlyFileNumbers=onlyFileNumbers)
            if len(files) > 0:
                for fileName in tqdm(files, desc="File Loop"):
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
                        setNames = CreateUtils.getSetNameForFile(fileName, defaultSetName, fileNamesNumbersToSets)
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
            rowProcessingMetadataDict[setName] = np.zeros((rowsInSet * metadataMultiplier, totalMetadataPackagingColumns))
        setsRowsProcessedTotal = {}

        for baseFileName in allBaseFileNames:
            files = np.array([f2 for f2 in sorted(os.listdir(rawDataFolder)) if
                              re.match(re.escape(baseFileName) + r'\d*\.(wav|hf)', f2)])
            files = CreateUtils.filterFilesByFileNumber(files, baseFileName, removeFileNumbers=removeFileNumbers,
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

                    setNames = CreateUtils.getSetNameForFile(fileName, defaultSetName, fileNamesNumbersToSets)
                    for setName in setNames:
                        if setName not in setsRowsProcessedTotal:
                            setsRowsProcessedTotal[setName] = 0
                        rowsProcessedTotal = setsRowsProcessedTotal[setName]
                        setDict[setName][0][rowsProcessedTotal:(rowsProcessedTotal + fXTemp.shape[0]), :] = fXTemp
                        setDict[setName][1][rowsProcessedTotal:(rowsProcessedTotal + fYTemp.shape[0]), :] = fYTemp
                        for mult in range(metadataMultiplier):
                            totalRows = totalRowsInSetDict[setName]
                            rowSlicer = slice(rowsProcessedTotal + (totalRows * mult), (rowsProcessedTotal + fYTemp.shape[0] + (totalRows * mult)))
                            rowSlicer2 = slice(fYTemp.shape[0] * mult, fYTemp.shape[0] * (mult + 1))
                            rowProcessingMetadataDict[setName][rowSlicer, :] = rowPackagingMetadata[rowSlicer2, :]
                        setsRowsProcessedTotal[setName] += fXTemp.shape[0]
                    gc.collect()
                gc.collect()

        timer.toc()  # extract features from files

        timer.tic("Filtering Sets")
        if filterFitSets is None:
            filterFitSets = [defaultSetName]
        savedFilterFile = os.path.join(processedDataFolder, "SavedFilters", featureSetName + '(sklearn-' + sklearn.__version__ + ').pkl')
        (setDict, xScaleFactor, xBias, yScaleFactor, yBias) = finalFilteringSets(setDict,
                                                                                 datasetParameters,
                                                                                 filterFitSets,
                                                                                 yValueType,
                                                                                 totalXColumns,
                                                                                 totalyColumns,
                                                                                 savedFilterFile)
        timer.toc()

        # repackage the sets for processing
        timer.tic("Repackaging sets")
        (setDict,
         trueRowsPerSetDict,
         trueColumnsPerSetDict,
         packagedRowsPerSetDict,
         packagedColumnsPerSetDict,
         kerasRowMultiplier) = repackageSets(setDict,
                                             datasetParameters,
                                             rowProcessingMetadataDict)
        timer.toc()

        # Start set breakout by fractions
        numberOfSamples = 0
        for setName, setValue in setDict.iteritems():
            numberOfSamples += setValue[0].shape[0]
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

        print("Total samples in all sets {0}".format(numberOfSamples))
        print("Samples Per Set:")
        for setName, setValue in setDict.iteritems():
            print("\t{setName}: {samples} samples".format(setName=setName, samples=setValue[0].shape[0]))
        print("Features per set {0}".format(totalXColumns))
        print("Output Dim per set {0}".format(totalyColumns))
        if yValueType in CreateUtils.yValueDiscreteTypes:
            print("Output Classes: {0}".format(len(outputLabelsFinal)))
        if rowPackagingStyle is not None:
            outputString = '\n'.join(['\t{setName}: {rows}'.format(setName=setName, rows=rows) for setName, rows in trueRowsPerSetDict.iteritems()])
            print("True Rows Per Set\n{0}".format(outputString))
            print("Keras Row Multiplier: {kerasRowMultiplier}".format(kerasRowMultiplier=kerasRowMultiplier))
            bytesPerArrayEntry = 4
            bytesPerBatchX = bytesPerArrayEntry * np.prod(setDict[defaultSetName][0].shape) / kerasRowMultiplier
            bytesPerBatchY = bytesPerArrayEntry * np.prod(setDict[defaultSetName][1].shape) / kerasRowMultiplier
            formatString = "Bytes per batch:\n\tX:     {0:>8} \n\tY:     {1:>8} \n\tTotal: {2:>8}"
            print(formatString.format(CreateUtils.sizeof_fmt(bytesPerBatchX),
                                      CreateUtils.sizeof_fmt(bytesPerBatchY),
                                      CreateUtils.sizeof_fmt(
                                          bytesPerBatchX + bytesPerBatchY)))

        formatString = '{setName:>15}|{xbytes:>8}|{ybytes:>8}|{totalbytes:>8}'
        outputString = '\n'.join(
            [formatString.format(setName=setName,
                                 xbytes=CreateUtils.sizeof_fmt(setValue[0].nbytes),
                                 ybytes=CreateUtils.sizeof_fmt(setValue[1].nbytes),
                                 totalbytes=CreateUtils.sizeof_fmt(setValue[0].nbytes + setValue[1].nbytes))
             for setName, setValue in setDict.iteritems()])
        formatString = "Size of each set: \n{setName:^15}|{xbytes:^8}|{ybytes:^8}|{totalbytes:^8}\n{outputString}"
        print(formatString.format(setName="Set Name",
                                  xbytes="X",
                                  ybytes="Y",
                                  totalbytes="Total",
                                  outputString=outputString))

        timer.tic("Save datasets and labels")
        # no package
        # set_x.shape = (sample x seqTimestep x data_dim)
        # if packaged
        # set_x.shape = (packageRow x packageTimestep x seqTimestep x data_dim)
        # if packaged and alternate rows
        # set_x.shape = (packageRow x sliceOfPackageRow x packageTimestep x seqTimestep x data_dim)
        with pd.HDFStore(datasetFile, 'w') as datasetStore:
            for setName, setValue in setDict.iteritems():
                datasetStore[setName + '_set_x'] = pd.DataFrame(setValue[0])
                datasetStore[setName + '_set_y'] = pd.DataFrame(setValue[1])
            datasetStore['labels'] = pd.DataFrame(outputLabelsFinal)
        timer.toc()

        datasetParametersToDump = dict(datasetParameters)
        datasetParametersToDump['allIncludedFiles'] = allIncludedFiles
        datasetParametersToDump['setNames'] = setDict.keys()
        datasetParametersToDump['filesInSets'] = totalFilesInSetDict
        datasetParametersToDump['trueRowsPerSetDict'] = trueRowsPerSetDict
        datasetParametersToDump['packagedRowsPerSetDict'] = packagedRowsPerSetDict
        datasetParametersToDump['trueColumnsPerSetDict'] = trueColumnsPerSetDict
        datasetParametersToDump['packagedColumnsPerSetDict'] = packagedColumnsPerSetDict
        datasetParametersToDump['kerasRowMultiplier'] = kerasRowMultiplier
        datasetParametersToDump['totalXColumns'] = totalXColumns
        datasetParametersToDump['totalyColumns'] = totalyColumns
        datasetParametersToDump['xScaleFactor'] = xScaleFactor
        datasetParametersToDump['xBias'] = xBias
        datasetParametersToDump['y value parameters']['yScaleFactor'] = yScaleFactor
        datasetParametersToDump['y value parameters']['yBias'] = yBias
        configFileName = CreateUtils.getDatasetConfigFileName(datasetName)
        CreateUtils.makeConfigFile(configFileName, datasetParametersToDump)
        timer.toc()  # overall time for whole function
        print ("######## Make Dataset Complete ############")


def getStatisticsOnSet(datasetParameters, featureParameters, allBaseFileNames=None, removeFileNumbers=(),
                       onlyFileNumbers=(), showFiguresArg=False):
    # feature parameters
    rawDataFolder = CreateUtils.getRawDataFolder()
    featureSetName = featureParameters['featureSetName']
    imageShape = featureParameters['imageShape']
    windowTimeLength = featureParameters['feature parameters']['windowTimeLength']
    if allBaseFileNames is None:
        allBaseFileNames = CreateUtils.getAllBaseFileNames(rawDataFolder)

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
        files = CreateUtils.filterFilesByFileNumber(files, baseFileName, removeFileNumbers=removeFileNumbers,
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
            setNames = CreateUtils.getSetNameForFile(filename, defaultSetName, fileNamesNumbersToSets)
            labels.append(','.join(setNames))

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 1, label, ha='center', va='bottom')
        plt.show()


def mainRun():
    # this line sets the root folder for future calls
    CreateUtils.getRootDataFolder(featureMethod="SignalPlaceholder")

    # Run Parameters   ############
    runNow = True
    forceRefreshFeatures = False
    overwriteConfigFile = True
    forceRefreshDataset = True
    rebuildFromConfig = True
    rebuildAllFromConfig = False
    removeDatasetNames = []
    removeFileNumbers = {}
    onlyFileNumbers = {}
    removeFeatureSetNames = []
    onlyThisFeatureSetNames = ['FFTWindowDefault']
    showFigures = True

    # Parameters Begin ############
    rngSeed = 0
    shuffleFinalSamples = False  # done after sequencing and filtering but before repackaging
    shuffleSamplesAfterRepackaging = True  # done after repackaging
    # per file parameters
    shuffleSamplesPerFile = False
    maxSamplesPerFile = 0

    setFractions = []
    # setFractions = [("train", "valid", 1.0 / 7), ("train", "test", 1.0 / 7)]
    timeDistributedY = True

    # metadata features ###############################################
    useMetadata = False
    metadataList = ['CadenceBike', 'CrankRevolutions', 'SpeedInstant', 'WheelRevolutions', 'DayPercent']
    metadataShape = (len(metadataList),)
    assert len(set(metadataList).union(CreateUtils.allMetadataNamesSet)) == len(
        CreateUtils.allMetadataNamesList), "You are using a metadata name not in the master list"

    # y Value type Specific variables #################################
    # gps variables
    includeAltitude = False
    decimalPrecision = (4, 4, -1)  # precision for lat, lon, and altitude to round to a grid
    gridSize = (20, 20, 1000)  # size of the grid used for gps discrete blocks
    # localLevelOriginInECEF = [506052.051626,-4882162.055080,4059778.630410] # AFIT
    localLevelOriginInECEF = [507278.89822834, -4884824.02376298, 4056425.76820216]  # Neighborhood Center

    # particle variables
    particleFilePath = os.path.join(CreateUtils.getImageryFolder(), "bikeneighborhoodPackFileNormParticleTDMparticleLocationsFromDataset.csv")

    # Sequences ########################################################
    makeSequences = False
    timestepsPerSequence = 1000  # if None use longest file timesteps
    offsetBetweenSequences = 1000  # negative or 0 will mean use random start locations
    fractionOfRandomToUse = 1.0
    padSequenceWithZeros = True
    # if False will use only full sequences, if True will extend last sequence and pad with zeros
    repeatSequenceBeginningAtEnd = True
    # if padSequenceWithZeros is True this will replace the zeros with the beginning of the sequence
    repeatSequenceEndingAtEnd = False
    # if padSequenceWithZeros is True this will replace the zeros with the end of the sequence backwards
    assert not (repeatSequenceBeginningAtEnd and repeatSequenceEndingAtEnd), \
        "you can't have both repeatSequenceBeginningAtEnd and repeatSequenceEndingAtEnd"

    # filter features ###################################################
    useSavedFilter = False
    savedFilterFile = os.path.join(CreateUtils.getProcessedDataDatasetsFolder('bikeneighborhoodPackFileNormParticle'),
                                   'SavedFilters',
                                   'FFTWindowDefault.pkl')

    filterPCA = True
    filterFitSets = ["normal"]  # names of the sets you want to use to filter
    filterPCAn_components = None
    filterPCAwhiten = True
    filterXToUpperDiagonal = False

    # x scaling
    xScaleFactor = 1.0
    xBias = 0.0
    xNormalized = False

    # y scale factor for continuous sets
    yScaleFactor = 1.0
    yBias = 0.0
    yNormalized = False

    # Packaging ########################################################
    rowPackagingStyle = 'BaseFileNameWithNumber'  # None, 'BaseFileNameWithNumber', 'gpsD', 'particle', 'class', 'classWithClassTransitions'
    padRowPackageWithZeros = True
    repeatRowPackageBeginningAtEnd = False
    repeatRowPackageEndingAtEnd = True
    assert not (repeatRowPackageBeginningAtEnd and repeatRowPackageEndingAtEnd), \
        "you can't have both repeatRowPackageBeginningAtEnd and repeatRowPackageEndingAtEnd"
    allSetsSameRows = True
    # packing parameters based on type of pack
    #   gpsD
    gridSizePackage = (100, 100, 1000)
    #   particle
    particlePackFilePath = os.path.join(CreateUtils.getImageryFolder(), "bikeneighborhoodPackFileNormParticleTDMparticleLocationsFromDataset.csv")
    #   classWithClassTransition
    minForClassTransition = 50
    # keras packaging
    alternateRowsForKeras = True
    timestepsPerKerasBatchRow = 100
    assert not (not allSetsSameRows and alternateRowsForKeras), "You must keep all sets same rows for keras packging to work in keras"

    # Set Definitions ################################################
    # region Other Datasets ###
    ######################

    # datasetName = 'MNIST'
    # allBaseFileNames = ['MNIST']
    # yValueType = 'Written Number'

    # datasetName = 'Test'
    # allBaseFileNames = ['Test']
    # yValueType = 'Multiple outputs'

    # datasetName = 'THoR'
    # allBaseFileNames = ["xAll"]
    # yValueType = 'Goal Value'
    # Static Datasets  ###
    ########################

    # datasetName = 'staticLocationsSequence'
    # allBaseFileNames = ['642lobby','646lobby','library','640244class','enyconf','einstein','jimsoffice',
    # 'dolittles','kennylobby','642lobbytop','antoffice','cube','640sideroom',
    # 'algebra','engconf','640338class','640220lab']
    # ##allBaseFileNames =  readwav.getAllBaseFileNames(CreateUtils.getRawDataFolder(), nomatch = ["afitwalk","afitwalkFail",
    # "afitinsidewalk",'mixer','microwave','transformer', 'kitchentable'])
    # yValueType = 'file'

    # datasetName = 'staticLocations'
    # allBaseFileNames = ['afittest', 'hometest']
    # yValueType = 'file'

    # datasetName = 'threeClassProblemSequence'
    # allBaseFileNames = ['algebra', 'cube', 'jimsoffice']
    # yValueType = 'file'

    # datasetName = 'threeClassProblemgpsC'
    # allBaseFileNames = ['algebra', 'cube', 'jimsoffice']
    # yValueType = 'gpsC'

    # datasetName = 'fourClassProblem'
    # allBaseFileNames = ['kitchentable','algebra', 'cube', 'jimsoffice']
    # yValueType = 'file'
    # Walking Datasets ###
    ########################

    # datasetName = 'vlfsignalsAfitInsideWalk'
    # allBaseFileNames = ["afitinsidewalk"]
    # yValueType = 'time'

    # datasetName = 'vlfsignalsAfitWalkAll'
    # allBaseFileNames = ["afitwalk", "afitwalkFail"]
    # yValueType = 'time'

    # datasetName = 'vlfsignalsAfitWalkSequenceC'
    # allBaseFileNames = ["afitwalk"]
    # yValueType = 'gpsC'

    # datasetName = 'vlfsignalsAfitWalkC'
    # allBaseFileNames = ["afitwalk"]
    # yValueType = 'gpsC'

    # datasetName = 'vlfsignalsAfitWalkSorted'
    # allBaseFileNames = ["afitwalk"]
    # yValueType = 'gpsC'
    # Car Datasets ###
    ####################

    # datasetName = 'toAFITandBackSequenceCTest'
    # allBaseFileNames = ["carafittohome","cartoafit"]
    # yValueType = 'gpsC'

    # datasetName = 'toYogaandBackSequenceC'
    # allBaseFileNames = ["carhometoyoga","caryogatohome"]
    # yValueType = 'gpsC'

    # datasetName = 'allcar'
    # allBaseFileNames = ["carhometoyoga","caryogatohome","carafittohome","cartoafit",
    # "carrandom", "carrandomback", "carneighborhood"]
    # yValueType = 'gpsC'

    # datasetName = 'carneighborhoodDiscrete'
    # allBaseFileNames = ["carneighborhood"]
    # yValueType = 'gpsD'
    # removeFileNumbers = {"carneighborhood":[0]}

    # datasetName = 'carneighborhoodNewSequenceC'
    # allBaseFileNames = ["carneighborhood"]
    # yValueType = 'gpsC'
    # removeFileNumbers = {"carneighborhood":[0,1]}
    # # onlyFileNumbers = {"carneighborhood":[4,5]}

    # datasetName = 'carneighborhoodTestSequenceC'
    # allBaseFileNames = ["carneighborhood"]
    # yValueType = 'gpsC'
    # # removeFileNumbers = {"carneighborhood":[0,1]}
    # onlyFileNumbers = {"carneighborhood": [11]}

    # datasetName = 'carAFITParkingLot'
    # allBaseFileNames = ["carafitparkinglot"]
    # yValueType = 'gpsC'
    # Bike Datasets ###
    #####################

    # datasetName = 'bikeneighborhoodC'
    # allBaseFileNames = ["bikeneighborhood"]
    # yValueType = 'gpsC'

    # datasetName = 'bikeneighborhoodSequenceC'
    # allBaseFileNames = ["bikeneighborhood"]
    # yValueType = 'gpsC'

    # endregion

    # datasetName = 'walkneighborhoodPackClassNormParticle'
    # allBaseFileNames = ["walkneighborhood"]
    # yValueType = 'particle'
    # onlyFileNumbers = {"walkneighborhood": []}
    # removeFileNumbers = {"walkneighborhood": [0]}
    # defaultSetName = "normal"
    # fileNamesNumbersToSets = [("crazy", "walkneighborhood", [1])]

    datasetName = 'bikeneighborhoodExamPackFileNormParticleShuffle'
    allBaseFileNames = ["bikeneighborhood"]
    yValueType = 'particle'
    onlyFileNumbers = {"bikeneighborhood": [31, ]}
    removeFileNumbers = {"bikeneighborhood": []}
    defaultSetName = "normal"
    # fileNamesNumbersToSets = [("crazy", "bikeneighborhood", [32])]
    fileNamesNumbersToSets = []

    # datasetName = 'bikeneighborhoodPackFileNormParticle'
    # allBaseFileNames = ["bikeneighborhood"]
    # yValueType = 'particle'
    # onlyFileNumbers = {"bikeneighborhood": range(33)}
    # removeFileNumbers = {"bikeneighborhood": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 29, 8, 31, 32]}
    # defaultSetName = "train"
    # fileNamesNumbersToSets = [("valid", "bikeneighborhood", [12, 14, 16, 18, 20, 22, 24, 26, 28]), ]

    ################################
    # Parameters End   ############
    ################################

    processedDataFolder = CreateUtils.getProcessedDataDatasetsFolder(datasetName)
    datasetConfigFileName = CreateUtils.getDatasetConfigFileName(datasetName)

    # region: Make Dicts
    datasetParametersToDump = {
        'datasetName': datasetName,
        'yValueType': yValueType,
        'removeFileNumbers': removeFileNumbers,
        'onlyFileNumbers': onlyFileNumbers,
        'y value parameters': {
            'yScaleFactor': yScaleFactor,
            'yBias': yBias,
            'yNormalized': yNormalized,
        },
        'allBaseFileNames': allBaseFileNames,
        'shuffleFinalSamples': shuffleFinalSamples,
        'shuffleSamplesAfterRepackaging': shuffleSamplesAfterRepackaging,
        'shuffleSamplesPerFile': shuffleSamplesPerFile,
        'rngSeed': rngSeed,
        'maxSamplesPerFile': maxSamplesPerFile,
        'filterXToUpperDiagonal': filterXToUpperDiagonal,

        'xScaleFactor': xScaleFactor,
        'xBias': xBias,
        'xNormalized': xNormalized,

        'defaultSetName': defaultSetName,
        'fileNamesNumbersToSets': fileNamesNumbersToSets,
        'setFractions': setFractions,
    }
    if makeSequences:
        sequencesDict = {
            'makeSequences': makeSequences,
            'timestepsPerSequence': timestepsPerSequence,
            'offsetBetweenSequences': offsetBetweenSequences,
            'fractionOfRandomToUse': fractionOfRandomToUse,
            'padSequenceWithZeros': padSequenceWithZeros,

            'timeDistributedY': timeDistributedY,
        }
        if padSequenceWithZeros:
            if repeatSequenceBeginningAtEnd:
                sequencesDict.update({'repeatSequenceBeginningAtEnd': repeatSequenceBeginningAtEnd, })
            elif repeatSequenceEndingAtEnd:
                sequencesDict.update({'repeatSequenceEndingAtEnd': repeatSequenceEndingAtEnd, })
        datasetParametersToDump.update(sequencesDict)
    if rowPackagingStyle is not None:
        packageDict = {
            'rowPackagingStyle': rowPackagingStyle,
            'padRowPackageWithZeros': padRowPackageWithZeros,
            'repeatRowPackageBeginningAtEnd': repeatRowPackageBeginningAtEnd,
            'repeatRowPackageEndingAtEnd': repeatRowPackageEndingAtEnd,
        }
        if rowPackagingStyle == 'gpsD':
            gridDict = {'gridSizePackage': gridSizePackage}
            packageDict.update(gridDict)
        if rowPackagingStyle == 'particle':
            particleFilePathDict = {'particlePackFilePath': particlePackFilePath}
            packageDict.update(particleFilePathDict)
        if alternateRowsForKeras:
            kerasDict = {
                'alternateRowsForKeras': alternateRowsForKeras,
                'timestepsPerKerasBatchRow': timestepsPerKerasBatchRow,
                'allSetsSameRows': allSetsSameRows,
            }
            packageDict.update(kerasDict)
        if rowPackagingStyle == 'classWithClassTransitions':
            minForClassDict = {'minForClassTransition': minForClassTransition}
            packageDict.update(minForClassDict)
        if rowPackagingStyle in ['class', 'classWithClassTransitions']:
            assert yValueType in CreateUtils.yValueDiscreteTypes, "class packing only valid with discrete"
        datasetParametersToDump.update(packageDict)
    if useSavedFilter:
        filterDict = {
            'useSavedFilter': useSavedFilter,
            'savedFilterFile': savedFilterFile
        }
        datasetParametersToDump.update(filterDict)
    elif filterPCA:
        filterDict = {
            'filterPCA': filterPCA,
            'filterFitSets': filterFitSets,
            'filterPCAn_components': filterPCAn_components,
            'filterPCAwhiten': filterPCAwhiten,
        }
        datasetParametersToDump.update(filterDict)
    if useMetadata:
        metadataDict = {
            'useMetadata': useMetadata,
            'metadataList': metadataList,
            'metadataShape': metadataShape
        }
        datasetParametersToDump.update(metadataDict)
    if yValueType == 'particle':
        datasetParametersToDump['y value parameters'].update({'particleFilePath': particleFilePath})
    if yValueType in CreateUtils.yValueGPSTypes:
        yValueByGPS_parameters_dict = {'y value by gps Parameters': {
            'includeAltitude': includeAltitude,
            'decimalPrecision': decimalPrecision,
            'gridSize': gridSize,
            'localLevelOriginInECEF': localLevelOriginInECEF
        }}
        datasetParametersToDump['y value parameters'].update(yValueByGPS_parameters_dict)
    # endregion

    if (not os.path.exists(datasetConfigFileName)) or overwriteConfigFile:
        if not os.path.exists(processedDataFolder):
            os.makedirs(processedDataFolder)
        savedFilterFile = os.path.join(processedDataFolder, 'SavedFilters')
        if not os.path.exists(savedFilterFile):
            os.makedirs(savedFilterFile)
        configFileName = CreateUtils.getDatasetConfigFileName(datasetName)
        if not overwriteConfigFile:
            assert not os.path.exists(configFileName), 'do you want to overwirte the config file?'
        CreateUtils.makeConfigFile(configFileName, datasetParametersToDump)
        datasets = [datasetName]
    else:
        if rebuildAllFromConfig:
            processedDataFolderMain = CreateUtils.getProcessedDataDatasetsFolder()
            datasets = [fileIterator for fileIterator in os.listdir(processedDataFolderMain) if
                        os.path.isdir(os.path.join(processedDataFolderMain,
                                                   fileIterator)) and fileIterator not in removeDatasetNames]
        elif rebuildFromConfig:
            datasets = [datasetName]
        else:
            datasets = []

    if runNow:
        featureDataFolderRoot = CreateUtils.getProcessedFeaturesFolder()
        if len(onlyThisFeatureSetNames) == 0:
            featureSetNamesAll = os.listdir(featureDataFolderRoot)
            featureSetNames = list(featureSetNamesAll)
            for featureSetName in featureSetNamesAll:
                if featureSetName in removeFeatureSetNames:
                    featureSetNames.remove(featureSetName)
        else:
            featureSetNames = list(onlyThisFeatureSetNames)
        for featureSetName in featureSetNames:
            featureConfigFileName = CreateUtils.getFeatureConfigFileName(featureSetName)
            featureParametersDefault = CreateUtils.loadConfigFile(featureConfigFileName)
            for datasetName in datasets:
                datasetConfigFileName = CreateUtils.getDatasetConfigFileName(datasetName)
                datasetParameters = CreateUtils.loadConfigFile(datasetConfigFileName)
                if overwriteConfigFile:
                    dictDiffer = CreateUtils.DictDiffer(datasetParameters, datasetParametersToDump)
                    print(dictDiffer.printAllDiff())
                removeFileNumbers = datasetParameters['removeFileNumbers'] \
                    if 'removeFileNumbers' in datasetParameters else {}
                onlyFileNumbers = datasetParameters['onlyFileNumbers'] \
                    if 'onlyFileNumbers' in datasetParameters else {}
                buildFeatures(featureParametersDefault,
                              allBaseFileNames=allBaseFileNames,
                              removeFileNumbers=removeFileNumbers,
                              onlyFileNumbers=onlyFileNumbers,
                              forceRefreshFeatures=forceRefreshFeatures)
                buildDataSet(datasetParameters,
                             featureParametersDefault,
                             forceRefreshDataset=forceRefreshDataset)
                getStatisticsOnSet(datasetParameters,
                                   featureParameters=featureParametersDefault,
                                   allBaseFileNames=allBaseFileNames,
                                   removeFileNumbers=removeFileNumbers,
                                   onlyFileNumbers=onlyFileNumbers,
                                   showFiguresArg=showFigures)


if __name__ == '__main__':
    mainRun()
