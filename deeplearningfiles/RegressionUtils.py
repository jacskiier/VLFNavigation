import cPickle
import gzip
import os
import numpy as np

import theano
import matplotlib.pylab as plt
from matplotlib import collections as mc
import yaml
import re
import pandas as pd


def load_data(datasetFileName,
              rogueClasses=(),
              makeSharedData=True,
              makeSequenceForX=False,
              makeSequenceForY=False,
              timesteps=0,
              setNames=('train', 'test', 'valid')):
    """ Loads the dataset

    :type datasetFileName: string
    :param datasetFileName: the path to the dataset pickle file that returns (train_set, valid_set, test_set)
    who are tuples with 0 being features and 1 being class values

    :type rogueClasses: list of class outputs (int)
    :param rogueClasses: List of classes to exclude from training to do rogue agalysis

    :type makeSharedData: bool
    :param makeSharedData: if true will return a theano shared variable representation instead of a np array representation

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
                set_x = featureStore['train_set_x'].as_matrix()
                set_y = featureStore['train_set_y'].as_matrix()
                setDict[setName] = (set_x, set_y)

    else:
        raise ValueError('Only .hf file types are supported')

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an np.ndarray of 2 dimensions (a matrix)
    # witch row's correspond to an example. target is a
    # np.ndarray of 1 dimensions (vector)) that have the same length as
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
        return shared_x, shared_y

    if len(rogueClasses) > 0:
        for setName in setNames:
            thisSet = setDict[setName]
            nonRogueMask = np.logical_not(np.in1d(thisSet[1], np.array(rogueClasses)))
            thisSet = (thisSet[0][nonRogueMask], thisSet[1][nonRogueMask])
            for rogueClass in list(np.flipud(rogueClasses)):
                thisSet[1][thisSet[1] > rogueClass] -= 1
            setDict[setName] = thisSet

    for setName in setNames:
        set_x, set_y = setDict[setName]
        if makeSequenceForX:
            set_x = np.reshape(set_x, (set_x.shape[0], timesteps, set_x.shape[1] / timesteps))
        if makeSequenceForY:
            set_y = np.reshape(set_y, (set_y.shape[0], timesteps, set_y.shape[1] / timesteps))
        setDict[setName] = (set_x, set_y)

    rval = []
    for setName in setNames:
        thisSet = setDict[setName]
        if makeSharedData:
            thisSet = shared_dataset(thisSet)
        rval.append(thisSet)
    inputFeatures = rval[0][0].shape[1]
    outputFeatures = rval[0][1].shape[1]
    largestSampleSetPossible = min([thisSet[1].shape[0] for thisSet in rval])
    print ("loading complete")
    return rval, inputFeatures, outputFeatures, largestSampleSetPossible


def getStatistics(predictedValues,
                  trueValues,
                  setName="",
                  statisticsStoreFolder='',
                  datasetParameters=None, weightsName=""):
    if datasetParameters is not None and datasetParameters['yValueType'] == 'gpsPolar':
        rtemp = predictedValues[:, 0]
        thetatemp = predictedValues[:, 1] * np.pi / 180.0
        predictedValues[:, 1] = rtemp * np.cos(thetatemp)
        predictedValues[:, 0] = rtemp * np.sin(thetatemp)

        rtemp = trueValues[:, 0]
        thetatemp = trueValues[:, 1] * np.pi / 180.0
        trueValues[:, 1] = rtemp * np.cos(thetatemp)
        trueValues[:, 0] = rtemp * np.sin(thetatemp)

    print("Getting statistics for regression model")
    if predictedValues.ndim < 2:
        predictedValues = np.expand_dims(predictedValues, axis=1)
    if trueValues.ndim < 2:
        trueValues = np.expand_dims(trueValues, axis=1)

    plt.figure(1)
    for outputDimension in range(predictedValues.shape[1]):
        ax = plt.subplot(1, predictedValues.shape[1], outputDimension + 1)
        plt.scatter(trueValues[:, outputDimension], predictedValues[:, outputDimension])

        plt.title('{setName} set Y dim {od}'.format(setName=setName, od=outputDimension))
        plt.xlabel('True Y value')
        plt.ylabel('Predicted Y value')
        plt.plot(plt.xlim(), plt.xlim())
        ax.set_aspect('equal', 'datalim')

    plt.savefig(os.path.join(statisticsStoreFolder, "yTrueVSyPred{0}{1}".format(setName, weightsName)))

    if predictedValues.shape[1] > 1:
        plt.figure(2)
        lines = []
        for lineNumber in range(predictedValues.shape[0]):
            lines.append([(trueValues[lineNumber, 1], trueValues[lineNumber, 0]),
                          (predictedValues[lineNumber, 1], predictedValues[lineNumber, 0])])
        lc = mc.LineCollection(lines, linewidths=0.5, zorder=1)
        plt.gca().add_collection(lc)
        plt.scatter(predictedValues[:, 1], predictedValues[:, 0], c='red', marker='v')
        plt.scatter(trueValues[:, 1], trueValues[:, 0], c='blue', marker='o')
        plt.axis('equal')
        plt.title('{setName} set 2D Output Map With Error'.format(setName=setName))
        plt.xlabel('Y Output Dimension 1')
        plt.ylabel('Y Output Dimension 0')
        plt.legend(['True Location', 'Predicted Location'])
        plt.savefig(os.path.join(statisticsStoreFolder, "2derror{0}{1}".format(setName, weightsName)))

    rss = np.sum((predictedValues - trueValues) ** 2)

    rse = np.sqrt(rss / (trueValues.shape[0] - 2))

    tss = np.sum((trueValues - np.mean(trueValues)) ** 2)

    r2 = (tss - rss) / tss

    mse = np.mean((predictedValues - trueValues) ** 2)

    meanDistance = np.mean(np.linalg.norm(predictedValues - trueValues, axis=1, ord=2))

    rmse = np.sqrt(mse)

    RMatrix = np.cov((predictedValues - trueValues).transpose())

    thresholdString = """Set {setName} Weights {weightsName}
    Residual Standard Error (RSE) {rse}
    Root Sum Square (RSS) Error {rss}
    Total Sum of Squares (TSS) {tss}
    R^2 (R squared) {r2}
    Mean Square Error (MSE) {mse}
    Mean Distance {meanDistance}
    Root Mean Square Error (RMSE) {rmse}
    Measurement Noise Matrix (R) {RMatrix}
    """.format(setName=setName,
               rse=rse,
               r2=r2,
               tss=tss,
               rss=rss,
               mse=mse,
               meanDistance=meanDistance,
               rmse=rmse,
               weightsName=weightsName,
               RMatrix=RMatrix)

    statsDict = {
        setName: {
            'RSS': rss,
            "MSE": mse,
            "MSE Vector": meanDistance,
        }
    }

    print (thresholdString)
    with open(os.path.join(statisticsStoreFolder, 'outputstats.log'), 'w') as logfile:
        logfile.write(thresholdString)
    with open(os.path.join(statisticsStoreFolder, 'results.yaml'), 'w') as resultsFile:
        yaml.dump(statsDict, resultsFile, default_flow_style=False, width=1000)
    return []


def rogueAnalysis(thresholds,
                  predictedValues,
                  trueValues,
                  classLabels=None,
                  rogueClasses=(),
                  staticFigures=7,
                  setName="",
                  statisticsStoreFolder=""):
    raise NotImplementedError, "Rogue Analysis is not implemented for Linear Regression type problems"
