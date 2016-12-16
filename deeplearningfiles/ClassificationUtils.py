import os
import re
import gzip
import cPickle
import yaml

import numpy as np

import theano
import theano.tensor as T

import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib
import matplotlib.colors
import matplotlib.patches
import matplotlib.collections

from sklearn.preprocessing import normalize

from scipy.stats import gaussian_kde
from scipy.spatial import Voronoi

import pandas as pd
import StringIO
from tqdm import tqdm

valueMethodNames = ['Probability', 'Probability Ratio', 'Probability Difference']


# from https://gist.github.com/pv/8036995
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plotVeroni(ax, regions, vertices, colorScaleNumbers):
    polygons = []
    for polygonIndex in range(len(regions)):
        region = regions[polygonIndex]
        polygon = vertices[region]
        thisPatch = matplotlib.patches.Polygon(polygon)
        polygons.append(thisPatch)

    colorMap = cm.get_cmap('coolwarm')
    p = matplotlib.collections.PatchCollection(patches=polygons, cmap=colorMap, norm=matplotlib.colors.NoNorm())
    p.set_array(colorScaleNumbers)
    ax.add_collection(p)
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title("Counts and True Positive percentage for each grid location")


def load_data(datasetFileName,
              rogueClasses=(),
              makeSharedData=True,
              makeSequenceForX=False,
              makeSequenceForY=False,
              timesteps=0,
              setNames=('train', 'valid', 'test')):
    """ Loads the dataset
    :type datasetFileName: string
    :param datasetFileName: the path to the dataset pickle file that returns (train_set, valid_set, test_set) who
    are tuples with 0 being features and 1 being class values

    :type rogueClasses: list of class outputs (int)
    :param rogueClasses: List of classes to exclude from training to do rogue agalysis

    :type makeSharedData: bool
    :param makeSharedData: if true will return a theano shared variable representation
    instead of a numpy array representation

    :type makeSequenceForX: bool
    :param makeSequenceForX: if true will use the timesteps to break up the dimensions to be [sample timestep data]

    :type makeSequenceForY: bool
    :param makeSequenceForY: if true will use the timesteps to break up the dimensions to be [sample timestep data]

    :type timesteps: int
    :param timesteps: used to make sequences for X or y
    """

    #############
    # LOAD DATA #
    #############

    # # Download the MNIST dataset if it is not present
    # data_dir, data_file = os.path.split(datasetFileName)
    # if data_dir == "" and not os.path.isfile(datasetFileName):
    #     # Check if dataset is in the data directory.
    #     new_path = os.path.join(
    #         os.path.split(__file__)[0],
    #         "..",
    #         "data",
    #         datasetFileName
    #     )
    #     if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
    #         datasetFileName = new_path
    #
    # if (not os.path.isfile(datasetFileName)) and data_file == 'mnist.pkl.gz':
    #     import urllib
    #     origin = (
    #         'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    #     )
    #     print 'Downloading data from %s' % origin
    #     urllib.urlretrieve(origin, datasetFileName)
    #
    print '... loading data'

    setDict = {}
    # Load the dataset
    if re.match('''.*\.hf$''', datasetFileName):
        with pd.HDFStore(datasetFileName, 'r') as featureStore:
            for setName in setNames:
                if setName is not None:
                    set_x = featureStore[setName + '_set_x'].as_matrix()
                    set_y = featureStore[setName + '_set_y'].as_matrix()
                    setDict[setName] = (set_x, set_y)
            labels = featureStore['labels'].as_matrix()
    else:
        raise ValueError('Only .hf file types are supported')

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # witch row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target
    # target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    if len(rogueClasses) > 0:
        for setName in setNames:
            if setName is not None:
                set_x = setDict[setName][0]
                set_y = setDict[setName][1]
                nonRogueMask = np.logical_not(np.in1d(set_y, np.array(rogueClasses)))
                set_x = set_x[nonRogueMask]
                set_y = set_y[nonRogueMask]
                for rogueClass in list(np.flipud(rogueClasses)):
                    set_y[set_y > rogueClass] -= 1
                setDict[setName] = (set_x, set_y)

    for setName in setNames:
        if setName is not None:
            set_x, set_y = setDict[setName]
            if makeSequenceForX:
                set_x = np.reshape(set_x, (set_x.shape[0], timesteps, set_x.shape[1] / timesteps))
            if makeSequenceForY:
                set_y = np.reshape(set_y, (set_y.shape[0], timesteps, set_y.shape[1] / timesteps))
            setDict[setName] = (set_x, set_y)

    rval = []
    for setName in setNames:
        if setName is not None:
            thisSet = setDict[setName]
            if makeSharedData:
                thisSet = shared_dataset(thisSet)
                setDict[setName] = thisSet
            rval.append(thisSet)
        else:
            rval.append((None, None))

    inputFeatures = rval[0][0].shape[1]

    realSetYTuple = ()
    for thisSet in rval:
        if thisSet[1] is not None:
            realSetYTuple += (thisSet[1],)

    allOutputClasses = np.concatenate(realSetYTuple, 0)
    outputClassesByUnique = np.unique(allOutputClasses).shape[0]
    outputClassesByLargest = int(np.max(allOutputClasses)) + 1
    outputClassesByLabels = labels.size
    outputClassesTotal = outputClassesByLabels
    assert outputClassesTotal <= max(outputClassesByUnique, outputClassesByLargest), "Class indices are beyond total classes"

    largestSampleSetPossible = min([thisSet1.shape[0] for thisSet1 in realSetYTuple])
    print ("loading complete")
    return rval, inputFeatures, outputClassesTotal, largestSampleSetPossible


def getLabelsForDataset(processedDataFolder, datasetFileName, includeRawLabels=False, nClassesTotal=None):
    classLabelsRaw = None
    if re.match('''.*\.pkl\.gz$''', datasetFileName):
        labels = datasetFileName.split('.')[0] + 'labels.pkl.gz'
        labelsFile = os.path.join(processedDataFolder, labels)
        if os.path.exists(labelsFile):
            f = gzip.open(labelsFile, 'rb')
            classLabelsRaw = cPickle.load(f)
    elif re.match('''.*\.hf$''', datasetFileName):
        with pd.HDFStore(datasetFileName, 'r') as featureStore:
            classLabelsRaw = featureStore['labels'].as_matrix()
            classLabelsRaw = classLabelsRaw.squeeze()

    if classLabelsRaw is not None:
        if len(np.array(classLabelsRaw).shape) == 0:
            classLabelsRaw = [classLabelsRaw]
        numLabels = np.array(classLabelsRaw).shape[0]
        classLabels = np.array(
            ["{className} ({classIndex})".format(className=classLabelsRaw[classValue], classIndex=classValue) for
             classValue in range(numLabels)])
    else:
        if nClassesTotal is not None:
            classLabels = ["{classIndex}".format(classIndex=classValue) for classValue in range(nClassesTotal)]
        else:
            classLabels = None
    if includeRawLabels:
        return classLabels, classLabelsRaw
    else:
        return classLabels


def getEERThreshold(predicted_class, predictedValues, true_class, nTotalClassesArg=None, rogueClasses=(), classLabels=None, thresholdsSaveFile=None):
    nRogue = len(rogueClasses)
    if nTotalClassesArg is None:
        # nTotalClasses = np.unique(true_class).shape[0] # this way doesn't work if some of your
        # output classes never happened
        # nTotalClasses = int(np.max(true_class)) + 1  # and this is gonna fail if the largest
        # output clases never happened
        # assert(np.unique(true_class).shape[0] == int(np.max(true_class)) + 1)

        if classLabels is None:
            # this way doesn't work if some of your output classes never happened
            # nTotalClasses = np.unique(true_class).shape[0]
            # and this is gonna fail if the largest output clases never happened
            nTotalClasses = int(np.max(true_class)) + 1
        else:
            nTotalClasses = len(classLabels)
    else:
        nTotalClasses = nTotalClassesArg

    nClasses = nTotalClasses - nRogue
    modelClasses = sorted(list((set(range(nTotalClasses)) - set(rogueClasses))))

    eerThresholds = np.zeros((nClasses,))

    # mask of all the times the input was a rogue from this analysis
    nonRogueMask = np.logical_not(np.in1d(true_class, np.array(rogueClasses)))
    predictedValues = predictedValues[nonRogueMask]
    predicted_class = predicted_class[nonRogueMask]
    true_class = true_class[nonRogueMask]
    true_class = true_class.squeeze()

    thresholdBins = 500

    for classIndex, classValue in tqdm(zip(range(nClasses), modelClasses), "EER Class Loop"):
        # mask of all the times it was actually x
        actualClassXMask = true_class == classValue
        # maks of all the times i think it was x
        predictedClassXMask = predicted_class == classIndex
        # now make mask  i was correct AND that i thought it was this class
        correctRows = np.logical_and(actualClassXMask, predictedClassXMask)
        # mask i was wrong AND that i thought i was correct
        incorrectRows = np.logical_and(np.logical_not(actualClassXMask), predictedClassXMask)

        # correctValues = np.empty( (0,) )
        # incorrectValues = np.empty( (0,) )

        # values when I predicted X and it was actually X
        correctValues = predictedValues[correctRows]
        # values when I predicted X and it wasn't X
        incorrectValues = predictedValues[incorrectRows]

        # Total times of predicted X and actually X
        correctCount = float(np.count_nonzero(correctValues))  # sum true positive
        # Total times of predicted X and not X
        incorrectCount = float(np.count_nonzero(incorrectValues))  # sum false positive

        allValues = np.append(correctValues, incorrectValues, axis=0)
        haveValues = allValues.shape[0] > 0
        if haveValues:
            maxValue = np.max(allValues)
            minValue = np.min(allValues)
            increment = ((maxValue - minValue) / float(thresholdBins))
            truePositiveRates = np.zeros((thresholdBins, 1))
            falsePositiveRates = np.zeros((thresholdBins, 1))

            for counter in range(thresholdBins):
                currentThresholdValue = minValue + increment * counter
                correctCountAtThreshold = np.count_nonzero(correctValues > currentThresholdValue)
                incorrectCountAtThreshold = np.count_nonzero(incorrectValues > currentThresholdValue)
                truePositiveRates[counter] = correctCountAtThreshold / correctCount if correctCount > 0 else 0
                falsePositiveRates[counter] = incorrectCountAtThreshold / incorrectCount if incorrectCount > 0 else 0

            if incorrectCount == 0:
                eer = np.min(correctValues)
            else:
                binarySearch = True
                if binarySearch:
                    maxIterations = 100
                    maxValueTemp = maxValue
                    minValueTemp = minValue
                    eer = (maxValueTemp - minValueTemp) / 2.0 + minValueTemp
                    for iterationIndex in range(maxIterations):
                        eer = (maxValueTemp - minValueTemp) / 2.0 + minValueTemp
                        correctCountAtThreshold = np.count_nonzero(correctValues > eer)
                        incorrectCountAtThreshold = np.count_nonzero(incorrectValues > eer)
                        truePositiveRateTemp = correctCountAtThreshold / correctCount if correctCount > 0 else 0
                        falsePositiveRateTemp = incorrectCountAtThreshold / incorrectCount if incorrectCount > 0 else 0
                        if falsePositiveRateTemp > 1 - truePositiveRateTemp:
                            minValueTemp = eer
                        elif falsePositiveRateTemp < 1 - truePositiveRateTemp:
                            maxValueTemp = eer
                        else:
                            break
                else:
                    eerIndex = np.argmin(abs(1 - truePositiveRates - falsePositiveRates))
                    eer = eerIndex * increment
            eerThresholds[classIndex] = eer
    if thresholdsSaveFile is not None:
        thresholdsDict = {
            'eerThresholds': [float(value) for value in eerThresholds], }
        with open(thresholdsSaveFile, 'w') as resultsFile:
            yaml.dump(thresholdsDict, resultsFile, default_flow_style=False, width=1000)
    return eerThresholds


def plotThresholds(predicted_class,
                   predictedValues,
                   true_class,
                   inputThresholds,
                   nTotalClassesArg=None,
                   classLabels=None,
                   rogueClasses=(),
                   staticFigures=7,
                   setName="",
                   valueMethodName="",
                   statisticsStoreFolder='',
                   plotThresholdFigures=True,
                   plotLabels=False,
                   useVoroni=False):
    nRogue = len(rogueClasses)
    if nTotalClassesArg is None:
        if classLabels is None:
            # this way doesn't work if some of your output classes never happened
            # nTotalClasses = np.unique(true_class).shape[0]
            # and this is gonna fail if the largest output clases never happened
            nTotalClasses = int(np.max(true_class)) + 1
        else:
            nTotalClasses = len(classLabels)
    else:
        nTotalClasses = nTotalClassesArg

    nClasses = nTotalClasses - nRogue
    modelClasses = sorted(list((set(range(nTotalClasses)) - set(rogueClasses))))
    confusionMatrix = np.zeros((nClasses, nClasses))
    confusionMatrixThresholded = np.zeros((nClasses, nClasses * 2))

    # mask of all the times the input was a rogue from this analysis
    nonRogueMask = np.logical_not(np.in1d(true_class, np.array(rogueClasses)))
    predictedValues = predictedValues[nonRogueMask]
    predicted_class = predicted_class[nonRogueMask]
    true_class = true_class[nonRogueMask]
    true_class = true_class.squeeze()

    thresholdValueDict = {}

    if classLabels is None:
        classLabels = np.array(["{0}".format(classValue) for classValue in range(1, nTotalClasses + 1)])

    thresholdBins = 500

    plt.figure(1)
    plt.clf()
    for classIndex, classValue in tqdm(zip(range(nClasses), modelClasses), "Plotting Class Loop"):
        # mask of all the times it was actually x
        actualClassXMask = true_class == classValue
        # maks of all the times i think it was x
        predictedClassXMask = predicted_class == classIndex
        # now make mask  i was correct AND that i thought it was this class
        correctRows = np.logical_and(actualClassXMask, predictedClassXMask)
        # mask i was wrong AND that i thought i was correct
        incorrectRows = np.logical_and(np.logical_not(actualClassXMask), predictedClassXMask)

        if np.any(actualClassXMask):
            confusionMatrix[classIndex, :] = np.bincount(predicted_class[actualClassXMask], minlength=nClasses)
        else:
            confusionMatrix[classIndex, :] = np.zeros(shape=(1, nClasses))

        # correctValues = np.empty( (0,) )
        # incorrectValues = np.empty( (0,) )

        # values when I predicted X and it was actually X
        correctValues = predictedValues[correctRows]
        # values when I predicted X and it wasn't X
        incorrectValues = predictedValues[incorrectRows]

        # Total times of predicted X and actually X
        correctCount = float(np.count_nonzero(correctValues))  # sum true positive
        # Total times of predicted X and not X
        incorrectCount = float(np.count_nonzero(incorrectValues))  # sum false positive

        thresholdValueDict[classIndex] = {}
        thresholdValueDict[classIndex][0] = correctValues
        thresholdValueDict[classIndex][1] = incorrectValues

        allValues = np.append(correctValues, incorrectValues, axis=0)
        haveValues = allValues.shape[0] > 0
        if haveValues:
            maxValue = np.max(allValues)
            minValue = np.min(allValues)
            increment = ((maxValue - minValue) / float(thresholdBins))
            truePositiveRates = np.zeros((thresholdBins, 1))
            falsePositiveRates = np.zeros((thresholdBins, 1))
            for counter in range(thresholdBins):
                currentThresholdValue = minValue + increment * counter
                correctCountAtThreshold = np.count_nonzero(correctValues > currentThresholdValue)
                incorrectCountAtThreshold = np.count_nonzero(incorrectValues > currentThresholdValue)
                truePositiveRates[counter] = correctCountAtThreshold / correctCount if correctCount > 0 else 0
                falsePositiveRates[counter] = incorrectCountAtThreshold / incorrectCount if incorrectCount > 0 else 0

            plt.figure(1)
            lineTypes = ['-', '--', '-.', ':', ]
            markerTypes = ['o', '*', '^', '+']
            lineType = lineTypes[classIndex % len(lineTypes)]
            markerType = markerTypes[int(classIndex / len(lineTypes)) % len(markerTypes)]
            plt.plot(falsePositiveRates, truePositiveRates, label=classLabels[classValue], linestyle=lineType,
                     marker=markerType)
            plt.xlim([-.1, 1.1])
            plt.ylim([-.1, 1.1])
            plt.title('ROC Curves for each class')
            plt.xlabel('False Verification Rate')
            plt.ylabel('True Verification Rate')

            if plotThresholdFigures:
                thisfig = plt.figure(classIndex + 1 + staticFigures)
                plt.clf()
                if correctValues.shape[0] > 1:
                    if np.unique(correctValues).shape[0] > 1:
                        density = gaussian_kde(correctValues)
                        xs = np.linspace(0, 1, 200)
                        plt.plot(xs, density(xs), color='green', label='Correct')
                    else:
                        plt.hist(correctValues, bins=thresholdBins, normed=1, facecolor='green', edgecolor='green',
                                 alpha=0.75, range=(0, maxValue), label='Correct', figure=thisfig)
                if incorrectValues.shape[0] > 1:
                    if np.unique(incorrectValues).shape[0] > 1:
                        density = gaussian_kde(incorrectValues)
                        xs = np.linspace(0, 1, 200)
                        plt.plot(xs, density(xs), color='red', label='Incorrect')
                    else:
                        plt.hist(incorrectValues, bins=thresholdBins, normed=1, facecolor='red', edgecolor='red',
                                 alpha=0.75, range=(0, maxValue), label='False', figure=thisfig)

                ax = plt.gca()
                plt.plot([inputThresholds[classIndex], inputThresholds[classIndex]], [0, ax.get_ylim()[1]], label='threshold')
                plt.title('Threshold Probabilities for Class {0}'.format(classLabels[classValue]))
                plt.xlabel('Threshold')
                plt.ylabel('Probability of that class')
                plt.legend()
                ax = plt.gca()
                if ax.get_xlim()[1] > 100:
                    ax.set_xscale('log')
                    ax.autoscale(enable=True, axis='x')
                plt.tight_layout()
                plt.savefig(os.path.join(statisticsStoreFolder,
                                         "Threshold Probabilities for class{0}.png".format(classLabels[classValue])))
                plt.close()

    plt.figure(1)
    lgd = plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
    plt.tight_layout(rect=(0, 0, 0.7, 1))
    plt.savefig(os.path.join(statisticsStoreFolder, "ROC Curve"), bbox_extra_artist=(lgd,), bbox_inches='tight')

    for classIndex, classValue in zip(range(nClasses), modelClasses):
        # mask of all the times it was actually x
        actualClassXMask = true_class == classValue
        # maks of all the times i think it was x
        # predictedClassXMask = predicted_class == classIndex
        # make threshold values masks
        thresholdGoodMask = np.logical_and(predictedValues >= inputThresholds[classIndex], actualClassXMask)
        thresholdBadMask = np.logical_and(predictedValues < inputThresholds[classIndex], actualClassXMask)

        confusionMatrixThresholded[classIndex, :nClasses] = np.bincount(predicted_class[thresholdGoodMask],
                                                                        minlength=nClasses)
        confusionMatrixThresholded[classIndex, nClasses:] = np.bincount(predicted_class[thresholdBadMask],
                                                                        minlength=nClasses)

    maxClassesToShow = 20.0
    classStep = int(np.ceil(nClasses / maxClassesToShow))
    classxticksToShow = range(0, nClasses, classStep)
    classxticksToShow2 = range(nClasses, nClasses * 2, classStep)
    modelClasses = np.array(modelClasses)
    xTickRotation = 90
    horizontalalignment = 'center'

    confusionMatrixNormalized = normalize(confusionMatrix, axis=1, norm='l1')
    plt.figure(2)
    plt.clf()
    plt.imshow(confusionMatrixNormalized, aspect='equal', interpolation='nearest',
               cmap=cm.get_cmap('coolwarm'))  # , extent = [0,nClasses,nClasses,0]
    plt.plot((-0.5, nClasses - 0.5), (-0.5, nClasses - 0.5), 'k-')
    plt.title('Confusion Matrix with no threshold')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(classxticksToShow, classLabels[modelClasses[classxticksToShow]], rotation=xTickRotation,
               horizontalalignment=horizontalalignment)
    plt.yticks(classxticksToShow, classLabels[modelClasses[classxticksToShow]])
    plt.gca().autoscale(enable=True, axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(statisticsStoreFolder, "confusionMatrx"))

    confusionMatrixThresholdedNormalized = np.append(
        normalize(confusionMatrixThresholded[:, :nClasses], axis=1, norm='l1'),
        normalize(confusionMatrixThresholded[:, nClasses:], axis=1, norm='l1'), axis=1)
    plt.figure(3)
    plt.clf()
    plt.imshow(confusionMatrixThresholdedNormalized, aspect='equal', interpolation='nearest',
               cmap=cm.get_cmap('coolwarm'))
    plt.plot((-0.5, nClasses - 0.5), (-0.5, nClasses - 0.5), 'k-')
    plt.plot((nClasses - 0.5, nClasses - 0.5), (-0.5, nClasses - 0.5), 'k-')
    plt.plot((nClasses - 0.5, nClasses * 2 - 0.5), (-0.5, nClasses - 0.5), 'k-')
    plt.title('Confusion Matrix with thresholds')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(np.append(classxticksToShow, classxticksToShow2),
               np.append(classLabels[modelClasses[classxticksToShow]], classLabels[modelClasses[classxticksToShow]]),
               rotation=xTickRotation, horizontalalignment=horizontalalignment)
    plt.yticks(classxticksToShow, classLabels[modelClasses[classxticksToShow]])
    plt.gca().autoscale(enable=True, axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(statisticsStoreFolder, "confusionMatrxWithThresholds"))

    plt.figure(4)
    plt.clf()
    for classIndex, classValue in zip(range(nClasses), modelClasses):
        plt.scatter(classIndex * np.ones((1, thresholdValueDict[classIndex][0].shape[0])),
                    thresholdValueDict[classIndex][0], marker='o', c='b', s=50, alpha=.1)
        plt.scatter(classIndex * np.ones((1, thresholdValueDict[classIndex][1].shape[0])),
                    thresholdValueDict[classIndex][1], marker='*', c='r', s=50, alpha=.1)
        plt.plot((classIndex - 0.25, classIndex + 0.25), (inputThresholds[classIndex], inputThresholds[classIndex]),
                 'k-')
    plt.title('EER threshold for each class')
    plt.xlabel('Classes')
    plt.ylabel('EER threshold')
    plt.ylim(ymin=-.05)
    ax = plt.gca()
    if ax.get_ylim()[1] > 100:
        ax.set_yscale('log')
    plt.xticks(classxticksToShow, classLabels[modelClasses[classxticksToShow]], rotation=xTickRotation,
               horizontalalignment=horizontalalignment)
    plt.gca().autoscale(enable=True, axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(statisticsStoreFolder, "eerthresholds"))

    totalCounts = np.sum(confusionMatrix)
    totalCountsPerClass = np.sum(confusionMatrix, axis=1)
    accuracyPerClass = np.diag(confusionMatrix) / totalCountsPerClass
    truePositiveRateNT = np.trace(confusionMatrix) / totalCounts
    falsePositiveRateNT = (np.sum(confusionMatrix) - np.trace(confusionMatrix)) / totalCounts

    upThresholdConfusionMatrix = confusionMatrixThresholded[:, :nClasses]
    downThresholdConfusionMatrix = confusionMatrixThresholded[:, nClasses:]
    totalCounts = np.sum(confusionMatrixThresholded)
    truePositiveRate = np.trace(upThresholdConfusionMatrix) / totalCounts
    falseNegativeRate = np.trace(downThresholdConfusionMatrix) / totalCounts
    falsePositiveRate = (np.sum(upThresholdConfusionMatrix) - np.trace(upThresholdConfusionMatrix)) / totalCounts
    trueNegativeRate = (np.sum(downThresholdConfusionMatrix) - np.trace(
        downThresholdConfusionMatrix)) / totalCounts
    accuracyThresholded = truePositiveRate + trueNegativeRate

    if plotLabels:
        plt.figure(5)
        ax = plt.gca()

        classLabelsRaw = []
        for classLabel in classLabels:
            classLabelsRaw.append(classLabel.split(' ')[0])

        outputString = '\n'.join(classLabelsRaw)
        outputLabelsAsArray = np.genfromtxt(StringIO.StringIO(outputString), delimiter=',')

        if useVoroni:
            polygons = []
            polygons_nocounts = []
            vor = Voronoi(points=outputLabelsAsArray)
            regions, vertices = voronoi_finite_polygons_2d(vor)
            vertices = np.hstack((vertices[:, 1][:, None], vertices[:, 0][:, None]))
            for polygonIndex in range(outputLabelsAsArray.shape[0]):
                region = regions[polygonIndex]
                polygon = vertices[region]
                xMid = outputLabelsAsArray[polygonIndex][1]
                yMid = outputLabelsAsArray[polygonIndex][0]

                acc = accuracyPerClass[polygonIndex]
                totalCountsThisClass = totalCountsPerClass[polygonIndex]
                thisPatch = matplotlib.patches.Polygon(polygon)
                polygons.append(thisPatch)

                if totalCountsThisClass <= 0:
                    polygons_nocounts.append(thisPatch)

                    # plt.text(x=xMid,
                    #          y=yMid,
                    #          s="{countsThisClass} \n{tpRateThisClass:2.2%}".format(tpRateThisClass=acc,
                    #                                                                countsThisClass=totalCountsThisClass),
                    #          size=12,
                    #          zorder=2,
                    #          color='k')

            colorMap = cm.get_cmap('coolwarm')
            patchCollection = matplotlib.collections.PatchCollection(patches=polygons, cmap=colorMap, norm=matplotlib.colors.NoNorm())
            patchCollection.set_array(accuracyPerClass)
            ax.add_collection(patchCollection)

            patch_nullCollection = matplotlib.collections.PatchCollection(patches=polygons_nocounts)
            patch_nullCollection.set_facecolor('g')
            ax.add_collection(patch_nullCollection)

            edgeBuffer = 100
            plt.scatter(outputLabelsAsArray[totalCountsPerClass > 0, 1], outputLabelsAsArray[totalCountsPerClass > 0, 0], marker='*')
            # plt.xlim([np.min(outputLabelsAsArray[:, 1]) - edgeBuffer, np.max(outputLabelsAsArray[:, 1]) + edgeBuffer])
            # plt.ylim([np.min(outputLabelsAsArray[:, 0]) - edgeBuffer, np.max(outputLabelsAsArray[:, 0]) + edgeBuffer])
            plt.xlim(vor.min_bound[1] - edgeBuffer, vor.max_bound[1] + edgeBuffer)
            plt.ylim(vor.min_bound[0] - edgeBuffer, vor.max_bound[0] + edgeBuffer)
            plt.xlabel('East (m)')
            plt.ylabel('North (m)')
            plt.title("True Positive percentage for each polygon")
            plt.savefig(os.path.join(statisticsStoreFolder, "labelsplot"))
        else:
            diffArray = np.diff(np.sort(outputLabelsAsArray[:, 0]))
            gridSizeX = np.min(diffArray[diffArray > 0])
            diffArray = np.diff(np.sort(outputLabelsAsArray[:, 1]))
            gridSizeY = np.min(diffArray[diffArray > 0])
            gridSize = (gridSizeX, gridSizeY)

            rects = []
            for outputLabelIndex in range(outputLabelsAsArray.shape[0]):
                outputLabelsAsNumbers = outputLabelsAsArray[outputLabelIndex, :]
                acc = accuracyPerClass[outputLabelIndex]
                totalCountsThisClass = totalCountsPerClass[outputLabelIndex]
                xMid = outputLabelsAsNumbers[1]
                yMid = outputLabelsAsNumbers[0]

                thisPatch = matplotlib.patches.Rectangle((xMid - gridSize[1] / 2.0, yMid - gridSize[0] / 2.0),
                                                         gridSize[1],
                                                         gridSize[0])
                rects.append(thisPatch)
                # plt.text(x=xMid - gridSize[1] / 2.0,
                #          y=yMid,
                #          s="{countsThisClass} \n{tpRateThisClass:2.2%}".format(tpRateThisClass=acc,
                #                                                                countsThisClass=totalCountsThisClass),
                #          size=12,
                #          zorder=2,
                #          color='k')

            colorMap = cm.get_cmap('coolwarm')
            patchCollection = matplotlib.collections.PatchCollection(patches=rects, cmap=colorMap, norm=matplotlib.colors.NoNorm())
            patchCollection.set_array(accuracyPerClass)
            ax.add_collection(patchCollection)
            plt.xlim([np.min(outputLabelsAsArray[:, 1]) - gridSize[1], np.max(outputLabelsAsArray[:, 1]) + gridSize[1]])
            plt.ylim([np.min(outputLabelsAsArray[:, 0]) - gridSize[0], np.max(outputLabelsAsArray[:, 0]) + gridSize[0]])
            plt.xlabel('East (m)')
            plt.ylabel('North (m)')
            plt.title("True Positive percentage for each grid location")
            plt.savefig(os.path.join(statisticsStoreFolder, "labelsplot"))

    thresholdString = """Set {setName}
No threshold
TP: {tpnt:1.3f} FP: {fpnt:1.3f}
Threshold EER using threshold method {valueMethodName}
TP: {tp:1.3f} FP: {fp:1.3f} \nFN: {fn:1.3f} TN: {tn:1.3f}
Thresholds for each class:\n{eerThresholds} """.format(setName=setName,
                                                       valueMethodName=valueMethodName,
                                                       tpnt=truePositiveRateNT, fpnt=falsePositiveRateNT,
                                                       tp=truePositiveRate, fp=falsePositiveRate, fn=falseNegativeRate,
                                                       tn=trueNegativeRate,
                                                       eerThresholds=str(inputThresholds))

    statsDict = {
        setName: {
            'noThreshold': {
                'tp': float(truePositiveRateNT),
                'fp': float(falsePositiveRateNT)
            },
            'threshold': {
                'tp': float(truePositiveRate),
                'fp': float(falsePositiveRate),
                'fn': float(falseNegativeRate),
                'tn': float(trueNegativeRate)
            }
        }
    }

    print (thresholdString)
    with open(os.path.join(statisticsStoreFolder, 'stats.log'), 'w') as logfile:
        logfile.write(thresholdString)
    with open(os.path.join(statisticsStoreFolder, 'results.yaml'), 'w') as resultsFile:
        yaml.dump(statsDict, resultsFile, default_flow_style=False, width=1000)


def rogueAnalysis(thresholds,
                  predicted_class,
                  predictedValues,
                  true_class,
                  nTotalClassesArg=None,
                  classLabels=None,
                  rogueClasses=(),
                  staticFigures=7,
                  setName="",
                  valueMethodName="",
                  statisticsStoreFolder="",
                  plotAllFigures=1):
    nRogue = len(rogueClasses)
    if nTotalClassesArg is None:
        if classLabels is None:
            # this is gonna fail if some output clases were never the true class
            # nTotalClasses = np.unique(true_class).shape[0]
            # and this one fails when the max class wasn't a true class
            nTotalClasses = int(np.max(true_class)) + 1
        else:
            nTotalClasses = len(classLabels)
    else:
        nTotalClasses = nTotalClassesArg
    nClasses = nTotalClasses - nRogue
    modelClasses = sorted(list(set(range(nTotalClasses)) - set(rogueClasses)))
    confusionMatrix = np.zeros((nRogue, nClasses))
    confusionMatrixThresholded = np.zeros((nRogue, nClasses * 2))

    thresholdValueDict = {}

    if classLabels is None:
        classLabels = np.array(["{0}".format(classValue) for classValue in range(1, nTotalClasses + 1)])

    for x, rogueClass in zip(range(nRogue), rogueClasses):
        # mask of all the times it was actually x
        actualClassXMask = true_class == rogueClass

        confusionMatrix[x, :] = np.bincount(predicted_class[actualClassXMask], minlength=nClasses)

        thresholdValueDict[x] = {}
        thresholdGoodMaskCombined = np.zeros((predicted_class.shape[0],))
        thresholdBadMaskCombined = np.zeros((predicted_class.shape[0],))
        for y, modelClass in zip(range(nClasses), modelClasses):
            # make mask of all time i thought it was y
            predictedClassYMask = (predicted_class == y)
            # make a mask for when i thought it was y but it was x and split that up based on threshold
            thresholdGoodMask = np.logical_and(predictedValues >= thresholds[y],
                                               np.logical_and(actualClassXMask, predictedClassYMask))
            thresholdBadMask = np.logical_and(predictedValues < thresholds[y],
                                              np.logical_and(actualClassXMask, predictedClassYMask))
            # save off these values for the end
            thresholdValueDict[x][y] = {}
            thresholdValueDict[x][y][0] = predictedValues[thresholdGoodMask]
            thresholdValueDict[x][y][1] = predictedValues[thresholdBadMask]
            # make a combined mask for the final confusion matrix
            thresholdGoodMaskCombined = np.logical_or(thresholdGoodMaskCombined, thresholdGoodMask)
            thresholdBadMaskCombined = np.logical_or(thresholdBadMaskCombined, thresholdBadMask)

        confusionMatrixThresholded[x, :nClasses] = np.bincount(predicted_class[thresholdGoodMaskCombined],
                                                               minlength=nClasses)
        confusionMatrixThresholded[x, nClasses:] = np.bincount(predicted_class[thresholdBadMaskCombined],
                                                               minlength=nClasses)

    maxClassesToShow = 20.0
    classStep = int(np.ceil(nClasses / maxClassesToShow))
    classxticksToShow = range(0, nClasses, classStep)
    classxticksToShow2 = range(nClasses, nClasses * 2, classStep)
    modelClasses = np.array(modelClasses)
    xTickRotation = 90
    horizontalalignment = 'center'

    confusionMatrixNormalized = normalize(confusionMatrix, axis=1, norm='l1')
    plt.figure(5)
    plt.clf()
    plt.imshow(confusionMatrixNormalized, aspect='equal', interpolation='nearest',
               cmap=cm.get_cmap('coolwarm'))  # , extent = [0,nClasses,nRogue,0]
    plt.title('Rogue Confusion Matrix with no threshold')
    plt.xlabel('Predicted Class')
    plt.ylabel('Rogue Class')
    plt.xticks(classxticksToShow, classLabels[modelClasses[classxticksToShow]], rotation=xTickRotation,
               horizontalalignment=horizontalalignment)
    plt.yticks(range(nRogue), classLabels[rogueClasses])
    plt.gca().autoscale(enable=True, axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(statisticsStoreFolder, "Rogue Confusion Matrix"))

    confusionMatrixThresholdedNormalized = np.append(
        normalize(confusionMatrixThresholded[:, :nClasses], axis=1, norm='l1'),
        normalize(confusionMatrixThresholded[:, nClasses:], axis=1, norm='l1'), axis=1)
    plt.figure(6)
    plt.clf()
    plt.imshow(confusionMatrixThresholdedNormalized, aspect='equal', interpolation='nearest',
               cmap=cm.get_cmap('coolwarm'))
    plt.plot((nClasses - 0.5, nClasses - 0.5), (-0.5, nRogue - 0.5), 'k-')
    plt.title('Rogue Confusion Matrix with threshold')
    plt.xlabel('Predicted Class')
    plt.ylabel('Rogue Class')
    plt.xticks(np.append(classxticksToShow, classxticksToShow2),
               np.append(classLabels[modelClasses[classxticksToShow]], classLabels[modelClasses[classxticksToShow]]),
               rotation=xTickRotation, horizontalalignment=horizontalalignment)
    plt.yticks(range(nRogue), classLabels[rogueClasses])
    plt.gca().autoscale(enable=True, axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(statisticsStoreFolder, "Rogue Confusion Matrix with thresholds"))

    if plotAllFigures == 1:
        for x, rogueClass in zip(range(nRogue), rogueClasses):
            plt.figure(nClasses + 1 + staticFigures + x)
            plt.clf()
            for y, modelClass in zip(range(nClasses), modelClasses):
                plt.scatter(y * np.ones((1, thresholdValueDict[x][y][0].shape[0])), thresholdValueDict[x][y][0],
                            marker='*', c='r', s=50, alpha=.1)
                plt.scatter(y * np.ones((1, thresholdValueDict[x][y][1].shape[0])), thresholdValueDict[x][y][1],
                            marker='o', c='b', s=50, alpha=.1)
                plt.plot((y - 0.25, y + 0.25), (thresholds[y], thresholds[y]), 'k-')
            plt.title('EER thresholds for Rogue Class {0}'.format(classLabels[rogueClass]))
            plt.xlabel('Classes')
            plt.ylabel('EER threshold')
            plt.ylim(ymin=-.05)
            ax = plt.gca()
            if ax.get_ylim()[1] > 100:
                ax.set_yscale('log')
            plt.xticks(classxticksToShow, classLabels[modelClasses[classxticksToShow]], rotation=xTickRotation)
            plt.gca().autoscale(enable=True, axis='both')
            plt.tight_layout()
            plt.savefig(os.path.join(statisticsStoreFolder,
                                     'EER thresholds for Rogue Class {0}.png'.format(classLabels[rogueClass])))

    upThresholdConfusionMatrix = confusionMatrixThresholded[:, :nClasses]
    downThresholdConfusionMatrix = confusionMatrixThresholded[:, nClasses:]
    totalCounts = np.sum(confusionMatrixThresholded)
    rogueAcceptRate = np.sum(upThresholdConfusionMatrix) / totalCounts
    rogueRejectRate = np.sum(downThresholdConfusionMatrix) / totalCounts

    rogueThresholdsString = """Set {setName}
Rogue Statistics using threshold method {valueMethodName}
RA: {ra:1.3f} RR: {rr:1.3f} """.format(setName=setName, valueMethodName=valueMethodName, ra=rogueAcceptRate,
                                       rr=rogueRejectRate)

    statsDictRogue = {
        'ra': float(rogueAcceptRate),
        'rr': float(rogueRejectRate)
    }
    print (rogueThresholdsString)

    with open(os.path.join(statisticsStoreFolder, 'results.yaml'), 'r') as resultsFile:
        statsDict = yaml.load(resultsFile)
    statsDict[setName]['rogue'] = statsDictRogue
    with open(os.path.join(statisticsStoreFolder, 'thresholdstats.log'), 'a') as logfile:
        logfile.write(rogueThresholdsString)
    with open(os.path.join(statisticsStoreFolder, 'results.yaml'), 'w') as resultsFile:
        yaml.dump(statsDict, resultsFile, default_flow_style=False, width=1000)
