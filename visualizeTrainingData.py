# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:05:37 2015

@author: jacsk
"""
import os
import yaml
import StringIO
import time

import numpy as np

import matplotlib.collections
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.patches
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

import pandas as pd
import pandas.tools.plotting

import sklearn.manifold
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition

from scipy.spatial import Voronoi

import CreateUtils
import ClassificationUtils
import RegressionUtils
import CreateDataset
from KerasClassifiers import getPredictedClasses_Values_TrueClasses_Labels
from ClassificationUtils import voronoi_finite_polygons_2d
from ClassificationUtils import plotVeroni

if os.name == 'nt':
    plt.rcParams['animation.ffmpeg_path'] = 'E:\\Program Files\\ffmpeg\\ffmpeg-20160512-git-cd244fa-win64-static\\bin\\ffmpeg.exe'
# elif os.name == 'posix':
#     plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'


rawDataFolder = CreateUtils.getRawDataFolder()

featureSetName = 'PatchShortTallAllFreq'
datasetName = "bikeneighborhoodPackFileNormParticleTDM"

whichSet = 1
whichSetName = ['train', 'valid', 'test'][whichSet]
downsample = None

# X visuals
showImages = False
makeAnimation = False

# Y Visuals
makeYScatterPlot = False
makeYScatterPlotColorOnY = False
showPathPerRowOfPackagedFile = False
gpsGrid = False

# Calculate Stats
calculatex_t0andP_t0 = False
kMeansOnRegressionY = False

# Prediction Visuals
videoClassProbability = True

# transforms of X or Y
makeDBSCAN = False
makeMiniBatchKMeans = False
clusterScatterKMeans = False
makePairsPlot = False
makePCAAnalysis = False
makePCAAnalysisRegression = False
makeIsomapX = False
makeIsomapYRegression = False
makeSpectralEmbeddingYRegression = False
makeLocallyLinearEmbeddingYRegression = False
makeTSNEPlotX = False
makeTSNEPlotYRegression = False

# Load all the config files
featureDataFolderMain = os.path.join(rawDataFolder, "Processed Data Features", featureSetName)
featureConfigFileName = os.path.join(featureDataFolderMain, "feature parameters.yaml")
with open(featureConfigFileName, 'r') as myConfigFile:
    featureParameters = yaml.load(myConfigFile)

processedDataFolderMain = os.path.join(rawDataFolder, "Processed Data Datasets", datasetName)
datasetConfigFileName = os.path.join(processedDataFolderMain, "dataset parameters.yaml")
with open(datasetConfigFileName, 'r') as myConfigFile:
    datasetParameters = yaml.load(myConfigFile)

if os.path.exists(os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'] + '.hf')):
    datasetFile = os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'] + '.hf')
else:
    datasetFile = os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'],
                               datasetParameters['datasetName'] + '.pkl.gz')

if datasetParameters['yValueType'] != 'gpsC':
    datasets, inputs, numParticles, max_batch_size = ClassificationUtils.load_data(datasetFile,
                                                                                   rogueClasses=(),
                                                                                   makeSharedData=False)
else:
    datasets, inputs, numParticles, max_batch_size = RegressionUtils.load_data(datasetFile,
                                                                               rogueClasses=(),
                                                                               makeSharedData=False)
outputLabels, outputLabelsRaw = ClassificationUtils.getLabelsForDataset(processedDataFolderMain, datasetFile,
                                                                        includeRawLabels=True)

train_set, valid_set, test_set = datasets
imageShape = featureParameters['imageShape']
timeDistributedY = datasetParameters['timeDistributedY'] if 'timeDistributedY' in datasetParameters else False
timestepsPerSequence = datasetParameters[
    'timestepsPerSequence'] if 'timestepsPerSequence' in datasetParameters else None
rowPackagingStyle = datasetParameters['rowPackagingStyle'] if 'rowPackagingStyle' in datasetParameters else None
packagedRowsPerSetDict = datasetParameters['packagedRowsPerSetDict'] \
    if 'packagedRowsPerSetDict' in datasetParameters else {}
packagedRows = packagedRowsPerSetDict[whichSetName]
oneSequencePerFile = datasetParameters['oneSequencePerFile'] if 'oneSequencePerFile' in datasetParameters else False

totalXColumns = datasetParameters['totalXColumns']
totalyColumns = datasetParameters['totalyColumns']

yValueType = datasetParameters['yValueType']

xNormalized = datasetParameters['xNormalized'] if 'xNormalized' in datasetParameters else False
yNormalized = datasetParameters['y value parameters']['yNormalized'] if 'yNormalized' in datasetParameters[
    'y value parameters'] else False
yBias = datasetParameters['y value parameters']['yBias'] if 'yBias' in datasetParameters[
    'y value parameters'] else 0.0
yScaleFactor = datasetParameters['y value parameters']['yScaleFactor'] if 'yScaleFactor' in datasetParameters[
    'y value parameters'] else 1.0

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

alternateRowsForKeras = datasetParameters[
    'alternateRowsForKeras'] if 'alternateRowsForKeras' in datasetParameters else False

print("Train set has {0} samples".format(train_set[0].shape[0]))
print("Valid set has {0} samples".format(valid_set[0].shape[0]))
print("Test  set has {0} samples".format(test_set[0].shape[0]))
print("Features per set {0}".format(test_set[0].shape[1]))
print("Output Dim per set {0}".format(test_set[1].shape[1] if len(test_set[1].shape) > 1 else 1))
print("Image Shape is {0}".format(imageShape))

XRaw = datasets[whichSet][0][0:downsample, :]
XRaw[np.isnan(XRaw)] = 0
X = XRaw
yRaw = datasets[whichSet][1][0:downsample, :]
y = yRaw
makeSequences = datasetParameters['makeSequences'] if 'makeSequences' in datasetParameters else False
# undo packaging
if rowPackagingStyle is not None:
    X = np.reshape(X,
                   (int(X.shape[0] * X.shape[1] / totalXColumns),
                    totalXColumns))
    y = np.reshape(y,
                   (int(y.shape[0] * y.shape[1] / totalyColumns),
                    totalyColumns))
# undo sequences
if makeSequences:
    X = np.reshape(X,
                   (int(X.shape[0] * timestepsPerSequence),
                    int(X.shape[1] / timestepsPerSequence)))
    if timeDistributedY:
        y = np.reshape(y,
                       (int(y.shape[0] * timestepsPerSequence),
                        int(y.shape[1] / timestepsPerSequence)))

##############
# Images for X
##############
if showImages or makeAnimation:
    imageList = []
    plotsX = 1
    plotsY = 1490
    totalPlots = plotsX * plotsY
    sampleOffset = 0  # so you can slice other then the first set of samples

    for sampleNum in tqdm(range(totalPlots * sampleOffset, totalPlots * (sampleOffset + 1)), "Creating Image"):

        originalFlatFeatures = X[sampleNum][:]
        # pull flat array out into correct image shape
        workingImageData = np.reshape(originalFlatFeatures, imageShape, order='C')
        # now reorder that shape into channel,width,height order
        imageShapeOrder = featureParameters['imageShapeOrder']
        workingImageData = CreateDataset.shuffleDimensions(workingImageData, imageShapeOrder)
        # but i actually want width,height, channel so do some swapping
        workingImageData = np.swapaxes(workingImageData, 0, 2)
        # this swap is just so i can plot each channel in the sub plot
        multiChannelImageData = np.swapaxes(workingImageData, 0, 1)

        # i want to see the channel change over the image
        plt.figure(1)
        plt.subplot(plotsX, plotsY, sampleNum % totalPlots + 1)
        for x in range(multiChannelImageData.shape[2]):
            plotData = multiChannelImageData[:, :, x].flatten()
            plt.plot(plotData)

        # in actuality the image should not have the axis swap so but it back
        multiChannelImageData = workingImageData
        # prep for showing total image
        # squeeze out any stupid dimensions ie if channel was 1
        if multiChannelImageData.shape[2] == 1:
            multiChannelImageData = np.squeeze(workingImageData, axis=2)
        # if we have more then 3 channels we need to limit it to 3
        if len(multiChannelImageData.shape) > 2 and multiChannelImageData.shape[2] > 1:
            threeChannelImageData = multiChannelImageData[:, :, :3]
        else:
            threeChannelImageData = multiChannelImageData
        imageList.append(threeChannelImageData)

    if showImages:
        # now plot the figure of all the channels on one image
        plt.figure(2)
        for sampleNum, threeChannelImageData in zip(range(len(imageList)), imageList):
            ax = plt.subplot(plotsX, plotsY, sampleNum % totalPlots + 1)
            extent = [0, featureParameters['feature parameters']['windowTimeLength'],
                      featureParameters['feature parameters']['windowFreqBounds'][1],
                      featureParameters['feature parameters']['windowFreqBounds'][0]]
            plt.imshow(threeChannelImageData, aspect=extent[1] / extent[2], interpolation='nearest', vmin=0, vmax=1,
                       cmap=cm.get_cmap('gray'), extent=extent)
            plt.title(str(y[sampleNum]))
            plt.xlabel('Window Time (s)')
            plt.ylabel('Frequency (Hz)', rotation=270)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        plt.show()

    if makeAnimation:
        fps = 50
        # def make_frame(t):
        #     return imageList[int(t*fps)]
        #
        # animation = VideoClip(make_frame, duration = len(imageList)/fps)
        #
        # #one of these should really work...
        # featureFolderName = "animationFiles/{0}".format(featureSetName)
        # fileBaseName = "animationFiles/{0}/{1}".format(featureSetName,datasetName)
        # if not os.path.exists(fileBaseName):
        #     os.mkdir(featureFolderName)
        # animation.write_videofile(fileBaseName + ".mp4",codec = 'mpeg4',  fps = fps)
        # #animation.write_gif(fileBaseName + ".gif", fps = fps) # export as GIF (slow)

        writer = animation.FFMpegWriter(fps=fps)
        fig = plt.figure()
        videoSavePath = os.path.join(rawDataFolder, "Imagery", datasetName + ".mp4")
        with writer.saving(fig, videoSavePath, 100):
            for threeChannelImageData in tqdm(imageList, "Making Animation Per Image"):
                extent = [0, featureParameters['feature parameters']['windowTimeLength'],
                          featureParameters['feature parameters']['windowFreqBounds'][1],
                          featureParameters['feature parameters']['windowFreqBounds'][0]]
                plt.imshow(threeChannelImageData, aspect=extent[1] / extent[2], interpolation='nearest', vmin=0, vmax=1,
                           cmap=cm.get_cmap('gray'), extent=extent)
                ax = plt.gca()
                plt.xlabel('Window Time (s)')
                plt.ylabel('Frequency (Hz)', rotation=270)
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                writer.grab_frame()


def getFactorsClosestToSquareRoot(numberToFactorOriginal):
    bestPair = (1, numberToFactorOriginal)

    for numberToFactor in range(numberToFactorOriginal, numberToFactorOriginal + int(np.sqrt(numberToFactorOriginal))):
        for lowerFactor in range(2, int(np.sqrt(numberToFactor)) + 1):
            higherFactor = numberToFactor / lowerFactor
            if higherFactor * lowerFactor == numberToFactor:
                if (bestPair[1] - bestPair[0]) > (higherFactor - lowerFactor):
                    bestPair = (lowerFactor, higherFactor)
    return bestPair


def convertBatchSequencesToFullSequences(yArg, batch_sizeArg):
    slicesPerBatch = len(np.arange(batch_sizeArg, yArg.shape[0], batch_sizeArg))
    finalOut = np.zeros((batch_sizeArg, slicesPerBatch * yArg.shape[1]))
    for seqsArg in range(0, batch_sizeArg):
        rangerArg = np.arange(seqsArg, slicesPerBatch * batch_sizeArg, batch_sizeArg)
        filterRangeArg = rangerArg
        oneSequence = yArg[filterRangeArg, :].flatten()
        finalOut[seqsArg, :] = oneSequence
    return finalOut


#################
# Images for Y
#################
if makeYScatterPlot:
    assert yValueType == 'gpsC', "only gpsC allowed for scatter plot"
    setForNorthEastMap = True
    XWorking = X
    yWorking = y
    if makeSequences:
        XWorking = np.reshape(X,
                              (int(X.shape[0] / timestepsPerSequence),
                               int(X.shape[1] * timestepsPerSequence)))
        if timeDistributedY:
            yWorking = np.reshape(y,
                                  (int(y.shape[0] / timestepsPerSequence),
                                   int(y.shape[1] * timestepsPerSequence)))
        if oneSequencePerFile:
            sliceTo = packagedRows
    plt.figure(1)
    totalObservations = y.shape[0]
    if setForNorthEastMap:
        plt.scatter(y[0:totalObservations, 1], y[0:totalObservations, 0])
    else:
        plt.scatter(y[0:totalObservations, 0], y[0:totalObservations, 1])
    plt.axis('equal')
    plt.show()

if makeYScatterPlotColorOnY:
    assert yValueType == 'gpsC', "only gpsC allowed for scatter plot"
    setForNorthEastMap = True
    plt.figure(1)
    totalObservations = y.shape[0]
    xDim = X.shape[1]

    for plotNumber in range(xDim):
        plt.subplot(1, xDim, plotNumber + 1)
        if setForNorthEastMap:
            plt.scatter(y[0:totalObservations, 1], y[0:totalObservations, 0], c=X[:, plotNumber],
                        cmap=cm.get_cmap('coolwarm'))
        else:
            plt.scatter(y[0:totalObservations, 0], y[0:totalObservations, 1])
        plt.axis('equal')
    plt.show()

if showPathPerRowOfPackagedFile:
    assert yValueType == 'gpsC', "can only show path for gpsC"
    assert rowPackagingStyle is not None, "you didn't package any rows"
    yWorking = yRaw

    if alternateRowsForKeras:
        yWorking = convertBatchSequencesToFullSequences(yWorking, packagedRows)

    # yWorking = yWorking / yScaleFactor - yBias
    print ('packagedRows {0}'.format(packagedRows))
    totalRuns = 1
    plotsY, plotsX = getFactorsClosestToSquareRoot(packagedRows)
    for seqs in range(packagedRows):
        plt.figure(1)
        # plt.subplot(plotsX, plotsY, seqs + 1)
        ranger = np.arange(seqs, yWorking.shape[0], packagedRows)
        filterRange = ranger
        print filterRange
        xer = yWorking[filterRange, 0::2].flatten()
        yer = yWorking[filterRange, 1::2].flatten()
        # plt.plot(yer, xer)
        thisCmap = cm.get_cmap('prism')
        thisColor = thisCmap(seqs / float(packagedRows))
        plt.scatter(yer, xer, c=thisColor, marker='o', edgecolors=thisColor)
        ax = plt.gca()
        miner = -800
        maxer = 600
        major_ticks = np.arange(start=gridSize[1] * int(miner / gridSize[1]),
                                stop=gridSize[1] * int(maxer / gridSize[1]),
                                step=gridSize[1])
        major_ticksX = major_ticks
        major_ticksY = major_ticks
        if yNormalized:
            miner = 0
            maxer = 1
            major_ticksX = yScaleFactor[1] * (major_ticksX + yBias[1])
            major_ticksY = yScaleFactor[0] * (major_ticksY + yBias[0])

        plt.ylim([miner, maxer])
        plt.xlim([miner, maxer])
        ax.set_xticks(major_ticksX)
        ax.set_yticks(major_ticksY)
        ax.grid(which='both')

        ax.grid(which='major', alpha=0.5)
    plt.show()

if gpsGrid:
    assert yValueType == 'gpsC' or yValueType == 'gpsD', "only gpsC and gpsD are allowed"
    yWorking = y
    if yValueType == 'gpsC':
        yWorking = yWorking / yScaleFactor - yBias
        yWorkingDiscrete = CreateDataset.deterineYValuesGridByGPSArrayInLocalLevelCoords(yWorking, gridSize)
        indexOfNewLabels, currentLabelList = CreateDataset.getIndexOfLabels(yWorkingDiscrete, [])
        outputString = '\n'.join(currentLabelList)
        outputLabelsAsArray = np.genfromtxt(StringIO.StringIO(outputString), delimiter=',')
        hist, bin_edges = np.histogram(indexOfNewLabels, bins=len(currentLabelList))
    else:
        outputString = '\n'.join(outputLabelsRaw)
        outputLabelsAsArray = np.genfromtxt(StringIO.StringIO(outputString), delimiter=',')
        hist, bin_edges = np.histogram(yWorking, bins=len(outputLabelsRaw))

    rects = []
    plt.figure(1)
    ax = plt.gca()
    diffArray = np.diff(np.sort(outputLabelsAsArray[:, 0]))
    gridSizeX = np.min(diffArray[diffArray > 0])
    diffArray = np.diff(np.sort(outputLabelsAsArray[:, 1]))
    gridSizeY = np.min(diffArray[diffArray > 0])
    for outputLabelsAsNumbers in outputLabelsAsArray:
        xMid = outputLabelsAsNumbers[1]
        yMid = outputLabelsAsNumbers[0]
        gridSizeScaled = yScaleFactor * (gridSize[:2] + yBias)

        gridSizeScaled = gridSize
        thisPatch = matplotlib.patches.Rectangle((xMid - gridSizeScaled[1] / 2.0, yMid - gridSizeScaled[0] / 2.0),
                                                 gridSizeScaled[1],
                                                 gridSizeScaled[0])
        rects.append(thisPatch)
        plt.text(x=xMid - gridSizeScaled[1] / 2.0,
                 y=yMid,
                 s=','.join([str(number) for number in outputLabelsAsNumbers]),
                 size=12,
                 zorder=2,
                 color='k')

    colorMap = cm.get_cmap('coolwarm')
    p = matplotlib.collections.PatchCollection(patches=rects, cmap=colorMap)
    p.set_array(hist)
    ax.add_collection(p)
    plt.xlim([np.min(outputLabelsAsArray[:, 1]) - gridSize[1], np.max(outputLabelsAsArray[:, 1]) + gridSize[1]])
    plt.ylim([np.min(outputLabelsAsArray[:, 0]) - gridSize[0], np.max(outputLabelsAsArray[:, 0]) + gridSize[0]])
    if yValueType == 'gpsC':
        plt.scatter(yWorking[:, 1], yWorking[:, 0], c='w', edgecolors='w')
    plt.show()

###################
# Stats for dataset
###################
if calculatex_t0andP_t0:
    assert yValueType == 'gpsC', "can't get starting coords for non gpsC"
    yWorking = yRaw
    if alternateRowsForKeras:
        yWorking = convertBatchSequencesToFullSequences(yWorking, packagedRows)

    startingCoords = yWorking[:, 0:2]

    x_t0 = np.mean(startingCoords, axis=0)
    P_t0 = np.var(startingCoords, axis=0)
    print ("starting coords")
    print (startingCoords)
    print ("x_t0")
    print (x_t0)
    print ("P_t0")
    print (P_t0)

if kMeansOnRegressionY:
    assert yValueType == 'gpsC', "can't do regression on Y with non gpsC"
    yWorking = y

    yWorking = yWorking / yScaleFactor - yBias
    numParticles = 100
    batch_size = 100
    random_state = 1
    mbk = MiniBatchKMeans(n_clusters=numParticles,
                          max_no_improvement=10,
                          batch_size=batch_size,
                          init='k-means++',
                          n_init=10,
                          compute_labels=True,
                          random_state=random_state,
                          verbose=1)

    mbk.fit(yWorking)
    mbk_means_labels = mbk.labels_
    mbk_means_cluster_centers = mbk.cluster_centers_
    mbk_means_labels_unique = np.unique(mbk_means_labels)

    plt.figure()
    colors = plt.cm.get_cmap("spectral")(np.linspace(0, 1, numParticles))

    for k, col in zip(range(numParticles), colors):
        my_members = mbk_means_labels == k
        cluster_center = mbk_means_cluster_centers[k]
        plt.scatter(yWorking[my_members, 1], yWorking[my_members, 0], color=col, marker='.')
        plt.scatter(cluster_center[1], cluster_center[0], color=col, marker="*", s=300, edgecolors=['k'])
    plt.axis('equal')
    plt.title('MiniBatchKMeans')
    saveCenters = True
    if saveCenters:
        filePath = os.path.join(rawDataFolder, "Imagery", datasetName + "RegressionYkMeansClusters.csv")
        np.savetxt(filePath, mbk_means_cluster_centers, fmt='%6.2f', delimiter=',', header="North, East")
    plt.show()

########################
# Prediction Visuals
########################
if videoClassProbability:
    classifierType = "LSTM"
    classifierSetName = "ClassificationAllClasses2LPlus2MLPStatefulAutoBatchDropReg2RlrRMSPropTD"
    experimentsFolder = os.path.join(rawDataFolder, "Data Experiments", featureSetName, datasetName, classifierType,
                                     classifierSetName)
    valueMethod = 0
    modelStoreNameType = 'best'
    modelDataFolder = os.path.join(rawDataFolder, "Processed Data Models", classifierType, classifierSetName)
    modelConfigFileName = os.path.join(modelDataFolder, "model set parameters.yaml")
    with open(modelConfigFileName, 'r') as myConfigFile:
        classifierParameters = yaml.load(myConfigFile)

    tupleOutputTemp = getPredictedClasses_Values_TrueClasses_Labels(datasetFileName=datasetFile,
                                                                    experimentStoreFolder=experimentsFolder,
                                                                    valueMethod=valueMethod,
                                                                    whichSetArg=whichSet,
                                                                    datasetParameters=datasetParameters,
                                                                    classifierParameters=classifierParameters,
                                                                    modelStoreNameType=modelStoreNameType,
                                                                    runByShape=True,
                                                                    returnClassProbabilities=True)
    (predicted_class_master, predicted_values_master, true_class_master, classLabelsMaster, predicted_probabilities) = tupleOutputTemp
    whichRuns = [0, 1, 2]

    outputString = '\n'.join(outputLabelsRaw)
    outputLabelsAsArray = np.genfromtxt(StringIO.StringIO(outputString), delimiter=',')
    totalTimesteps = predicted_probabilities.shape[1]

    dobywhat = 'fpsandtotaltime'
    if dobywhat == 'totaltimeandskip':
        totalTimeInSeconds = 60
        skipTimesteps = 50

        totalTimestepsToDo = totalTimesteps / skipTimesteps
        fps = max(totalTimestepsToDo / totalTimeInSeconds, 1)
    elif dobywhat == 'fpsandtotaltime':
        totalTimeInSeconds = 60
        fps = 30

        totalTimestepsToDo = fps * totalTimeInSeconds
        skipTimesteps = totalTimesteps / totalTimestepsToDo
    else:
        raise ValueError("pick a do by what that is valid")

    print ("fps {fps}, total time steps {totalTimesteps}".format(fps=fps, totalTimesteps=predicted_probabilities.shape[1]))
    print ("Total timesteps to do {totalTimestepsToDo} skip timesteps {skipTimesteps}".format(totalTimestepsToDo=totalTimestepsToDo,
                                                                                              skipTimesteps=skipTimesteps))
    writer = animation.FFMpegWriter(fps=fps)
    for whichRun in whichRuns:
        fig = plt.figure()
        fileName = "ProbabilityVideo_{datasetName}_{whichSetName}_run{run}_fps{fps}.mp4".format(fps=fps, whichSetName=whichSetName, run=whichRun,
                                                                                                datasetName=datasetName)
        videoSavePath = os.path.join(rawDataFolder, "Imagery", fileName)
        dpi = 100
        with writer.saving(fig, videoSavePath, 100):
            ax = plt.gca()
            vor = Voronoi(points=outputLabelsAsArray)
            regions, vertices = voronoi_finite_polygons_2d(vor)
            vertices = np.hstack((vertices[:, 1][:, None], vertices[:, 0][:, None]))
            for timestep in tqdm(range(0, totalTimesteps, skipTimesteps), desc="Video Frame Loop"):
                thisStepProbabilities = predicted_probabilities[whichRun, timestep, :]
                plotVeroni(ax, regions, vertices, thisStepProbabilities)
                edgeBuffer = 100
                plt.scatter(outputLabelsAsArray[:, 1], outputLabelsAsArray[:, 0], marker='*')
                # plt.xlim([np.min(outputLabelsAsArray[:, 1]) - edgeBuffer, np.max(outputLabelsAsArray[:, 1]) + edgeBuffer])
                # plt.ylim([np.min(outputLabelsAsArray[:, 0]) - edgeBuffer, np.max(outputLabelsAsArray[:, 0]) + edgeBuffer])
                plt.xlim(vor.min_bound[1] - edgeBuffer, vor.max_bound[1] + edgeBuffer)
                plt.ylim(vor.min_bound[0] - edgeBuffer, vor.max_bound[0] + edgeBuffer)
                plt.scatter(outputLabelsAsArray[:, 1], outputLabelsAsArray[:, 0], marker='*', c='r')
                plt.scatter(outputLabelsAsArray[int(true_class_master[whichRun, timestep, 0]), 1],
                            outputLabelsAsArray[int(true_class_master[whichRun, timestep, 0]), 0],
                            marker='*',
                            s=300,
                            c='g')
                writer.grab_frame()

########################
# X or Y Transformations
########################
if makeDBSCAN:
    labels_true = y
    db = DBSCAN(eps=1.0, min_samples=20).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    if len(set(labels)) > 1:
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, labels))

    ##############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.get_cmap("spectral")(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

    plt.figure()
    unique_true_labels = set(labels_true)
    colors = plt.cm.get_cmap("spectral")(np.linspace(0, 1, len(unique_true_labels)))
    for k, col in zip(unique_true_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels_true == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('True number of classes: %d' % len(unique_true_labels))
    plt.show()

if makeMiniBatchKMeans:
    print ("Doing KMeans")
    batch_size = min(45, max_batch_size)
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=numParticles, batch_size=batch_size,
                          n_init=10, max_no_improvement=10, verbose=0)
    t0 = time.time()
    mbk.fit(X)
    t_mini_batch = time.time() - t0
    mbk_means_labels = mbk.labels_
    mbk_means_cluster_centers = mbk.cluster_centers_
    mbk_means_labels_unique = np.unique(mbk_means_labels)

    fig = plt.figure(figsize=(8, 3))

    # MiniBatchKMeans
    ax = fig.add_subplot(1, 3, 2)
    colors = plt.cm.get_cmap("spectral")(np.linspace(0, 1, numParticles))
    for k, col in zip(range(numParticles), colors):
        my_members = mbk_means_labels == k
        cluster_center = mbk_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
    ax.set_title('MiniBatchKMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %
             (t_mini_batch, mbk.inertia_))
    plt.show()

if makePairsPlot:
    df = pd.DataFrame(X)
    print(df.columns)
    df = df.loc[:, 0:10]
    axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2)
    plt.tight_layout()
    plt.show()

if makePCAAnalysis:
    featuresToAnalyze = (0, 1, 2)
    np.random.seed(5)

    n_components = 0.9  # X.shape[1]
    pca = decomposition.PCA(n_components=n_components, whiten=True, copy=True)
    pca.fit(X)
    X = pca.transform(X)

    plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('Feautre Number')
    plt.ylabel('Percentage of variance explained')
    plt.subplot(1, 2, 2)
    plt.plot(pca.explained_variance_)
    plt.xlabel('Feautre Number')
    plt.ylabel('Amount of variance explained (in feature units)')

    cov = pca.get_covariance()
    cov = np.abs(cov)
    plt.figure(2)
    plt.imshow(cov, aspect='equal', interpolation='nearest', cmap=cm.get_cmap('coolwarm'))

    plt.figure(1)
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    for name, label in zip(outputLabels, range(len(outputLabels))):
        ax.text3D(X[y == label, featuresToAnalyze[0]].mean(),
                  X[y == label, featuresToAnalyze[1]].mean() + 1.5,
                  X[y == label, featuresToAnalyze[2]].mean(),
                  name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    # y = np.choose(y, [1, 2, 0]).astype(np.float)

    ax.scatter(X[:, featuresToAnalyze[0]], X[:, featuresToAnalyze[1]], X[:, featuresToAnalyze[2]], c=y,
               cmap=plt.cm.get_cmap("spectral"))

    x_surf = [X[:, featuresToAnalyze[0]].min(), X[:, featuresToAnalyze[0]].max(),
              X[:, featuresToAnalyze[0]].min(), X[:, featuresToAnalyze[0]].max()]
    y_surf = [X[:, featuresToAnalyze[0]].max(), X[:, featuresToAnalyze[0]].max(),
              X[:, featuresToAnalyze[0]].min(), X[:, featuresToAnalyze[0]].min()]
    x_surf = np.array(x_surf)
    y_surf = np.array(y_surf)
    v0 = pca.transform(pca.components_[[0]])
    v0 /= v0[-1]
    v1 = pca.transform(pca.components_[[1]])
    v1 /= v1[-1]

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.figure(4)
    comp = pca.components_
    xcomp = int(np.sqrt(comp.shape[0])) + 1
    ycomp = xcomp

    for compNumber in range(comp.shape[0]):
        plt.subplot(xcomp, ycomp, compNumber + 1)
        imager = np.reshape(comp[compNumber, :], imageShape[1:3])
        plt.imshow(imager, interpolation='nearest')
        plt.title('Component {0}'.format(compNumber))

    plt.show()

if clusterScatterKMeans:
    numParticles = 2
    print ("Doing KMeans")

    batch_size = min(45, max_batch_size)
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=numParticles, batch_size=batch_size,
                          n_init=10, max_no_improvement=10, verbose=0)
    t0 = time.time()
    mbk.fit(X)
    t_mini_batch = time.time() - t0
    mbk_means_labels = mbk.labels_
    mbk_means_cluster_centers = mbk.cluster_centers_
    mbk_means_labels_unique = np.unique(mbk_means_labels)

    plt.figure()
    # MiniBatchKMeans
    colors = plt.cm.get_cmap("spectral")(np.linspace(0, 1, numParticles))
    print("Found {0} output clusters".format(numParticles))
    for k, col in zip(range(numParticles), colors):
        my_members = mbk_means_labels == k
        cluster_center = mbk_means_cluster_centers[k]
        plt.scatter(y[my_members, 1], y[my_members, 0], color=col, marker='.')
        plt.axis('equal')
    plt.title('MiniBatchKMeans')
    plt.show()

if makePCAAnalysisRegression:
    featuresToAnalyze = (0, 1, 2)
    np.random.seed(5)

    pca = decomposition.PCA(n_components=0.9)
    pca.fit(X)
    X = pca.transform(X)

    plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('Feautre Number')
    plt.ylabel('Percentage of variance explained')
    plt.subplot(1, 2, 2)
    plt.plot(pca.explained_variance_)
    plt.xlabel('Feautre Number')
    plt.ylabel('Amount of variance explained (in feature units)')

    cov = pca.get_covariance()
    cov = np.abs(cov)
    plt.figure(2)
    plt.imshow(cov, aspect='equal', interpolation='nearest', cmap=cm.get_cmap('coolwarm'))

    plt.figure(1)
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # for name, label in zip(outputLabels,range(len(outputLabels))):
    #     ax.text3D(X[y == label, featuresToAnalyze[0]].mean(),
    #               X[y == label, featuresToAnalyze[1]].mean() + 1.5,
    #               X[y == label, featuresToAnalyze[2]].mean(),
    #               name,
    #               horizontalalignment='center',
    #               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    # y = np.choose(y, [1, 2, 0]).astype(np.float)
    y0normalized = (y[:, 0] - np.min(y[:, 0])) / np.max(y[:, 0])
    y1normalized = (y[:, 1] - np.min(y[:, 1])) / np.max(y[:, 1])
    ax.scatter(X[:, featuresToAnalyze[0]], X[:, featuresToAnalyze[1]], X[:, featuresToAnalyze[2]], c=y1normalized,
               cmap=plt.cm.get_cmap("spectral"))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.figure(4)
    comp = pca.components_
    xcomp = int(np.sqrt(comp.shape[0])) + 1
    ycomp = xcomp

    for compNumber in range(comp.shape[0]):
        plt.subplot(xcomp, ycomp, compNumber + 1)
        imager = np.reshape(comp[compNumber, :], imageShape[1:3])
        plt.imshow(imager, interpolation='nearest')
        plt.title('Component {0}'.format(compNumber))
    plt.show()


def makePlotOfy1andy0WithRespectToX(xTransformArg, yarg):
    y0normalizedArg = (yarg[:, 0] - np.min(yarg[:, 0])) / np.max(yarg[:, 0])
    y1normalizedArg = (yarg[:, 1] - np.min(yarg[:, 1])) / np.max(yarg[:, 1])

    plt.figure(1)
    plt.scatter(yarg[:, featuresToAnalyze[1]], yarg[:, featuresToAnalyze[0]], c=y0normalizedArg,
                cmap=plt.cm.get_cmap("spectral"))
    plt.figure(2)
    plt.scatter(xTransformArg[:, featuresToAnalyze[1]], xTransformArg[:, featuresToAnalyze[0]], c=y0normalizedArg,
                cmap=plt.cm.get_cmap("spectral"))
    plt.figure(3)
    plt.scatter(yarg[:, featuresToAnalyze[1]], yarg[:, featuresToAnalyze[0]], c=y1normalizedArg,
                cmap=plt.cm.get_cmap("spectral"))
    plt.figure(4)
    plt.scatter(xTransformArg[:, featuresToAnalyze[1]], xTransformArg[:, featuresToAnalyze[0]], c=y1normalizedArg,
                cmap=plt.cm.get_cmap("spectral"))
    plt.show()


if makeIsomapX:
    isomap = sklearn.manifold.Isomap(n_neighbors=5, n_components=2)
    xTransform = isomap.fit_transform(X, y)

    featuresToAnalyze = (0, 1, 2)

    makePlotOfy1andy0WithRespectToX(xTransformArg=xTransform, yarg=y)

if makeIsomapYRegression:
    assert yValueType == 'gpsC', "This only works for gpsC"
    isomap = sklearn.manifold.Isomap(n_neighbors=5, n_components=2)
    isomap.fit(y)
    yTransform = isomap.transform(y)

    featuresToAnalyze = (0, 1, 2)

    y0normalized = (y[:, 0] - np.min(y[:, 0])) / np.max(y[:, 0])
    y1normalized = (y[:, 1] - np.min(y[:, 1])) / np.max(y[:, 1])

    plt.figure(1)
    plt.scatter(y[:, featuresToAnalyze[1]], y[:, featuresToAnalyze[0]], c=y1normalized,
                cmap=plt.cm.get_cmap("spectral"))

    plt.figure(2)
    plt.scatter(yTransform[:, featuresToAnalyze[1]], yTransform[:, featuresToAnalyze[0]], c=y1normalized,
                cmap=plt.cm.get_cmap("spectral"))
    # plt.scatter(yTransform[:, featuresToAnalyze[0]], np.zeros( yTransform[:, featuresToAnalyze[0]].shape ) ,
    #  c=y1normalized, cmap=plt.cm.get_cmap("spectral"))
    plt.show()

if makeSpectralEmbeddingYRegression:
    assert yValueType == 'gpsC', "This only works for gpsC"
    isomap = sklearn.manifold.SpectralEmbedding(n_components=2)
    yTransform = isomap.fit_transform(y)

    featuresToAnalyze = (0, 1, 2)

    y0normalized = (y[:, 0] - np.min(y[:, 0])) / np.max(y[:, 0])
    y1normalized = (y[:, 1] - np.min(y[:, 1])) / np.max(y[:, 1])

    plt.figure(1)
    plt.scatter(y[:, featuresToAnalyze[1]], y[:, featuresToAnalyze[0]], c=y1normalized,
                cmap=plt.cm.get_cmap("spectral"))

    plt.figure(2)
    plt.scatter(yTransform[:, featuresToAnalyze[1]], yTransform[:, featuresToAnalyze[0]], c=y1normalized,
                cmap=plt.cm.get_cmap("spectral"))
    # plt.scatter(yTransform[:, featuresToAnalyze[0]], np.zeros( yTransform[:, featuresToAnalyze[0]].shape ) ,
    #  c=y1normalized, cmap=plt.cm.get_cmap("spectral"))
    plt.show()

if makeLocallyLinearEmbeddingYRegression:
    assert yValueType == 'gpsC', "This only works for gpsC"
    lle = sklearn.manifold.LocallyLinearEmbedding()
    yTransform = lle.fit_transform(y)

    featuresToAnalyze = (0, 1, 2)

    y0normalized = (y[:, 0] - np.min(y[:, 0])) / np.max(y[:, 0])
    y1normalized = (y[:, 1] - np.min(y[:, 1])) / np.max(y[:, 1])

    plt.figure(1)
    plt.scatter(y[:, featuresToAnalyze[1]], y[:, featuresToAnalyze[0]], c=y1normalized,
                cmap=plt.cm.get_cmap("spectral"))

    plt.figure(2)
    plt.scatter(yTransform[:, featuresToAnalyze[1]], yTransform[:, featuresToAnalyze[0]], c=y1normalized,
                cmap=plt.cm.get_cmap("spectral"))
    # plt.scatter(yTransform[:, featuresToAnalyze[0]], np.zeros( yTransform[:, featuresToAnalyze[0]].shape ) ,
    # c=y1normalized, cmap=plt.cm.get_cmap("spectral"))
    plt.show()

if makeTSNEPlotX:
    print("Making TSNE")
    tsne = sklearn.manifold.TSNE(n_components=2, perplexity=30.0, random_state=0)

    xTransform = tsne.fit_transform(X, y)

    featuresToAnalyze = (0, 1, 2)

    y0normalized = (y[:, 0] - np.min(y[:, 0])) / np.max(y[:, 0])
    y1normalized = (y[:, 1] - np.min(y[:, 1])) / np.max(y[:, 1])

    makePlotOfy1andy0WithRespectToX(xTransformArg=xTransform, yarg=y)

if makeTSNEPlotYRegression:
    assert yValueType == 'gpsC', "This only works for gpsC"
    print("Making TSNE")
    tsne = sklearn.manifold.TSNE(n_components=2, perplexity=30.0, random_state=0)

    yTransform = tsne.fit_transform(y)

    featuresToAnalyze = (0, 1, 2)

    y0normalized = (y[:, 0] - np.min(y[:, 0])) / np.max(y[:, 0])
    y1normalized = (y[:, 1] - np.min(y[:, 1])) / np.max(y[:, 1])

    plt.figure(1)
    plt.scatter(y[:, featuresToAnalyze[1]], y[:, featuresToAnalyze[0]], c=y1normalized,
                cmap=plt.cm.get_cmap("spectral"))

    plt.figure(2)
    plt.scatter(yTransform[:, featuresToAnalyze[1]], yTransform[:, featuresToAnalyze[0]], c=y1normalized,
                cmap=plt.cm.get_cmap("spectral"))
    # plt.scatter(yTransform[:, featuresToAnalyze[0]], np.zeros( yTransform[:, featuresToAnalyze[0]].shape ) ,
    # c=y1normalized, cmap=plt.cm.get_cmap("spectral"))
    plt.show()
