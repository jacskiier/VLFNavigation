import os

import csv
import warnings
from collections import Iterable, OrderedDict
import shutil

import matplotlib.pylab as plt
import numpy as np

import theano.tensor as T

import keras.callbacks
from keras.callbacks import Callback
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, TimeDistributed, LSTM
from keras.optimizers import rmsprop, adam
from keras.regularizers import WeightRegularizer
import keras.constraints
import keras.backend as K

import ClassificationUtils
import RegressionUtils
from KalmanFilter import getDiscreteSystem
from KalmanFilter import KalmanFilterLayer
from WaveletTransformLayer import WaveletTransformLayer
import CreateUtils


def showModel(modelArg):
    from keras.utils.visualize_util import model_to_dot
    from cStringIO import StringIO
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    png_str = model_to_dot(modelArg, show_shapes=True, show_layer_names=True).create(prog='dot', format='png')
    # treat the dot output string as an image file
    sio = StringIO()
    sio.write(png_str)
    sio.seek(0)
    img = mpimg.imread(sio)
    # plot the image
    plt.imshow(img, aspect='equal')
    plt.show()


def saveModelToFile(modelArg, filenameArg):
    from keras.utils.visualize_util import plot
    plot(modelArg, to_file=filenameArg, show_shapes=True, show_layer_names=True)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


yScaleFactorRMS = 1.0
yBiasRMS = 0


def root_mean_squared_error_unscaled(y_true, y_pred):
    y_pred = (y_pred / yScaleFactorRMS) - yBiasRMS
    y_true = (y_true / yScaleFactorRMS) - yBiasRMS
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def mean_square_error_with_variance_estimate(y_true, y_pred):
    trueOutputs = y_true.shape[1]
    y_predEstimate = y_pred[:, 0:trueOutputs]
    y_predVariance = y_pred[:, trueOutputs:]
    mse = K.mean(K.square(y_predEstimate - y_true))
    covarianceMSE = K.mean(K.square(K.square(y_predEstimate - y_true) - y_predVariance))
    return mse + covarianceMSE


def falsePositiveRate(y_true, p_y_given_x):
    # input is the probability of each class p_y_given_x
    # find the most likely class and make that y_pred
    y_pred = T.argmax(p_y_given_x, axis=p_y_given_x.ndim - 1)
    y_pred = y_pred[:, :, None]
    # compare the predicted class to the ture and mean to get error
    return T.mean(T.neq(y_pred, y_true))


def truePositiveRate(y_true, p_y_given_x):
    # input is the probability of each class p_y_given_x
    # find the most likely class and make that y_pred
    y_pred = T.argmax(p_y_given_x, axis=p_y_given_x.ndim - 1)
    y_pred = y_pred[:, :, None]
    # compare the predicted class to the ture and mean to get error
    return T.mean(T.eq(y_pred, y_true))


def crossEntropyWithIndex(y_true, p_y_given_x):
    y_trueFlat = T.flatten(y_true)
    p_y_given_xFlat = T.reshape(p_y_given_x, (
        T.prod(p_y_given_x.shape) / p_y_given_x.shape[p_y_given_x.ndim - 1], p_y_given_x.shape[p_y_given_x.ndim - 1]))
    vals = - T.log(p_y_given_xFlat[:, y_trueFlat])
    return T.mean(vals)


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    # Example
        ```python
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=5, min_lr=0.001)
            model.fit(X_train, Y_train, callbacks=[reduce_lr])
        ```

    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', epsilon=1e-4, cooldown=0, min_lr=0):
        super(Callback, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.reset()

    def reset(self):
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode), RuntimeWarning)
            self.mode = 'auto'
        if self.mode == 'min' or (self.mode == 'auto' and 'acc' not in self.monitor):
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1e-4

    def on_train_begin(self, logs={}):
        self.reset()

    def on_epoch_end(self, epoch, logs={}):
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires %s available!' %
                          self.monitor, RuntimeWarning)
        else:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif self.cooldown_counter <= 0:
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr + self.lr_epsilon:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: reducing learning rate to %s.' % (epoch, new_lr))
                        self.cooldown_counter = self.cooldown
                self.wait += 1


class CSVLogger(Callback):
    """Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example
        ```python
            csv_logger = CSVLogger('training.log')
            model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```

    Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs={}):
        if self.append:
            self.csv_file = open(self.filename, 'a')
        else:
            self.csv_file = open(self.filename, 'w')

    def on_epoch_end(self, epoch, logs={}):
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(lambda x: str(x), k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())
            self.writer = csv.DictWriter(self.csv_file, fieldnames=['epoch'] + self.keys)
            self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs={}):
        self.csv_file.close()


def compileAModel(model, classifierParameters, datasetParameters):
    learning_rate = classifierParameters['learning_rate']
    optimizerType = classifierParameters['optimizerType']
    lossType = classifierParameters['lossType']
    metrics = classifierParameters['metrics']

    # loss type
    if lossType == 'falsePositiveRate':
        lossType = falsePositiveRate
    elif lossType == 'crossEntropyWithIndex':
        lossType = crossEntropyWithIndex

    # optimizer
    epsilon = classifierParameters['epsilon']
    if optimizerType == 'rmsprop':
        rho = classifierParameters['rho']
        thisOptimizer = rmsprop(lr=learning_rate, rho=rho, epsilon=epsilon)
    elif optimizerType == 'adam':
        beta_1 = classifierParameters['beta_1']
        beta_2 = classifierParameters['beta_2']
        thisOptimizer = adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    else:
        raise ValueError("Optimizer type of {0} is not supported".format(optimizerType))

    # metrics
    # set up globas for root_mean_squared_error_unscaled metric
    global yScaleFactorRMS
    global yBiasRMS
    yScaleFactorRMS = datasetParameters['y value parameters']['yScaleFactor'] if 'yScaleFactor' in datasetParameters[
        'y value parameters'] else 1.0
    yBiasRMS = datasetParameters['y value parameters']['yBias'] if 'yBias' in datasetParameters[
        'y value parameters'] else 0.0
    # swap in custom functions for the string names
    for i in range(len(metrics)):
        if metrics[i] == "root_mean_squared_error":
            metrics[i] = root_mean_squared_error
        if metrics[i] == "root_mean_squared_error_unscaled":
            metrics[i] = root_mean_squared_error_unscaled
        if metrics[i] == "truePositiveRate":
            metrics[i] = truePositiveRate
        if metrics[i] == "falsePositiveRate":
            metrics[i] = falsePositiveRate

    model.compile(loss=lossType,
                  optimizer=thisOptimizer,
                  metrics=metrics)


def getPredictedClasses_Values_TrueClasses_Labels(datasetFileName='mnist.pkl.gz',
                                                  experimentStoreFolder='',
                                                  valueMethod=0,
                                                  whichSetArg=2,
                                                  datasetParameters=None,
                                                  classifierParameters=None,
                                                  modelStoreNameType="best",
                                                  runByShape=False,
                                                  returnClassProbabilities=False):
    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'model.json'.format(modelStoreNameType))
    if not os.path.exists(modelStoreFilePathFullTemp):
        modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.json'.format(modelStoreNameType))

    with open(modelStoreFilePathFullTemp, 'r') as modelStoreFile:
        json_string = modelStoreFile.read()
    model = model_from_json(json_string, custom_objects={"KalmanFilterLayer": KalmanFilterLayer, "WaveletTransformLayer": WaveletTransformLayer})

    modelStoreWeightsFilePathFullTemp = os.path.join(experimentStoreFolder,
                                                     '{0}_modelWeights.h5'.format(modelStoreNameType))
    model.load_weights(modelStoreWeightsFilePathFullTemp)

    compileAModel(model,
                  classifierParameters=classifierParameters,
                  datasetParameters=datasetParameters)

    makeSequences = datasetParameters['makeSequences'] if 'makeSequences' in datasetParameters else False
    alternateRowsForKeras = datasetParameters[
        'alternateRowsForKeras'] if 'alternateRowsForKeras' in datasetParameters else False
    if alternateRowsForKeras:
        timesteps = datasetParameters['timestepsPerKerasBatchRow']
        makeSequencesForX = True
    elif makeSequences:
        timesteps = datasetParameters['timestepsPerSequence']
        makeSequencesForX = True
    else:
        timesteps = 0
        makeSequencesForX = False
    useTimeDistributedOutput = classifierParameters[
        'useTimeDistributedOutput'] if 'useTimeDistributedOutput' in classifierParameters else False

    datasets, inputs, outputs, max_batch_size = ClassificationUtils.load_data(datasetFileName,
                                                                              rogueClasses=(),
                                                                              makeSharedData=False,
                                                                              makeSequenceForX=makeSequencesForX,
                                                                              makeSequenceForY=useTimeDistributedOutput,
                                                                              timesteps=timesteps)
    if whichSetArg > 2 or whichSetArg < 0:
        raise ValueError("Invalid which set number {0}".format(whichSetArg))
    X_test = datasets[whichSetArg][0]
    if 'useWaveletTransform' in classifierParameters and classifierParameters['useWaveletTransform'] is True:
        X_test = X_test[:, :, None, :]
    set_X = X_test
    trueValues = datasets[whichSetArg][1]

    batch_size = min(classifierParameters['batch_size'], max_batch_size)
    stateful = classifierParameters['stateful'] if 'stateful' in classifierParameters else False
    if stateful:
        whichSetName = ['train', 'valid', 'test'][whichSetArg]
        packagedRowsPerSetDict = datasetParameters['packagedRowsPerSetDict']
        packagedRows = packagedRowsPerSetDict[whichSetName]
        auto_stateful_batch = classifierParameters[
            'auto_stateful_batch'] if 'auto_stateful_batch' in classifierParameters else False
        if auto_stateful_batch:
            batch_size = packagedRows
        else:
            assert batch_size == packagedRows, \
                "You chose stateful but your batch size didn't match the files in the training set"
    print("Making Prediction Classes")
    predicted_probabilities = model.predict_proba(set_X, batch_size=batch_size, verbose=2)
    print("Predictions over")
    # reshape predicted_probabilities to be rows and indices
    predicted_probabilities = np.reshape(predicted_probabilities, (
        predicted_probabilities.shape[0] * predicted_probabilities.shape[1], predicted_probabilities.shape[2]))

    trueValues = np.reshape(trueValues, (trueValues.shape[0] * trueValues.shape[1], trueValues.shape[2]))
    # trueValues = totalSamples x output data dim

    processedDataFolder = os.path.dirname(datasetFileName)
    classLabels = ClassificationUtils.getLabelsForDataset(processedDataFolder, datasetFileName)

    # predicted_class ranges from 0 to nClasses
    predicted_class = np.argmax(predicted_probabilities, axis=1)

    predictedValues = None
    if valueMethod == 0:  # probability
        predictedValues = np.max(predicted_probabilities, axis=1)
    elif valueMethod == 1:  # probability relative to next highest
        predicted_probabilitiesSorted = np.fliplr(np.sort(predicted_probabilities, axis=1))
        predictedValues = predicted_probabilitiesSorted[:, 0] / predicted_probabilitiesSorted[:, 1]
    elif valueMethod == 2:  # probability difference to next highest
        predicted_probabilitiesSorted = np.fliplr(np.sort(predicted_probabilities, axis=1))
        predictedValues = predicted_probabilitiesSorted[:, 0] - predicted_probabilitiesSorted[:, 1]

    if runByShape and stateful:
        kerasRowMultiplier = datasetParameters['kerasRowMultiplier']
        totalyColumns = datasetParameters['totalyColumns']

        predictedValues = np.reshape(predictedValues, newshape=(batch_size * kerasRowMultiplier, timesteps * totalyColumns))

        predictedValues = np.reshape(predictedValues,
                                     newshape=(batch_size, predictedValues.shape[0] / batch_size, predictedValues.shape[1]),
                                     order='F')
        predictedValues = np.reshape(predictedValues,
                                     newshape=(batch_size, predictedValues.shape[1] * timesteps, predictedValues.shape[2] / timesteps))

        # trueValues.shape = (batch_size * kerasRowMultiplier * timesteps x totalColumnsY
        trueValues = np.reshape(trueValues, newshape=(batch_size * kerasRowMultiplier, timesteps * totalyColumns))
        # true_values.shape = (run count in timestep x timestep data dim)
        trueValues = np.reshape(trueValues, newshape=(batch_size, trueValues.shape[0] / batch_size, trueValues.shape[1]), order='F')
        # true_values.shape = (run x slice of run x timestep data dim)
        trueValues = np.reshape(trueValues, newshape=(batch_size, trueValues.shape[1] * timesteps, trueValues.shape[2] / timesteps))

        predicted_class = np.reshape(predicted_class, newshape=(batch_size * kerasRowMultiplier, timesteps * 1))
        predicted_class = np.reshape(predicted_class, newshape=(batch_size, kerasRowMultiplier, timesteps * 1), order='F')
        predicted_class = np.reshape(predicted_class, newshape=(batch_size, kerasRowMultiplier * timesteps, 1))

        predicted_probabilities = np.reshape(predicted_probabilities, newshape=(batch_size * kerasRowMultiplier, timesteps * outputs))
        predicted_probabilities = np.reshape(predicted_probabilities, newshape=(batch_size, kerasRowMultiplier, timesteps * outputs), order='F')
        predicted_probabilities = np.reshape(predicted_probabilities, newshape=(batch_size, kerasRowMultiplier * timesteps, outputs))

    if returnClassProbabilities is False:
        returnTuple = (predicted_class, predictedValues, trueValues, classLabels)
    else:
        returnTuple = (predicted_class, predictedValues, trueValues, classLabels, predicted_probabilities)
    return returnTuple


def getPred_Values_True_Labels(datasetFileName='mnist.pkl.gz', experimentStoreFolder='', whichSetArg=2,
                               datasetParameters=None, classifierParameters=None, modelStoreNameType="best",
                               shapeByRun=False):
    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'model.json'.format(modelStoreNameType))
    if not os.path.exists(modelStoreFilePathFullTemp):
        modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.json'.format(modelStoreNameType))
    with open(modelStoreFilePathFullTemp, 'r') as modelStoreFile:
        json_string = modelStoreFile.read()
    model = model_from_json(json_string, custom_objects={"KalmanFilterLayer": KalmanFilterLayer})

    modelStoreWeightsFilePathFullTemp = os.path.join(experimentStoreFolder,
                                                     '{0}_modelWeights.h5'.format(modelStoreNameType))
    model.load_weights(modelStoreWeightsFilePathFullTemp)

    compileAModel(model,
                  classifierParameters=classifierParameters,
                  datasetParameters=datasetParameters)

    makeSequences = datasetParameters['makeSequences'] if 'makeSequences' in datasetParameters else False
    alternateRowsForKeras = datasetParameters[
        'alternateRowsForKeras'] if 'alternateRowsForKeras' in datasetParameters else False
    if alternateRowsForKeras:
        timesteps = datasetParameters['timestepsPerKerasBatchRow']
        makeSequencesForX = True
    elif makeSequences:
        timesteps = datasetParameters['timestepsPerSequence']
        makeSequencesForX = True
    else:
        timesteps = 0
        makeSequencesForX = False
    useTimeDistributedOutput = classifierParameters[
        'useTimeDistributedOutput'] if 'useTimeDistributedOutput' in classifierParameters else False

    datasets, inputs, nClassesTotal, max_batch_size = RegressionUtils.load_data(datasetFileName, rogueClasses=(),
                                                                                makeSharedData=False,
                                                                                makeSequenceForX=makeSequencesForX,
                                                                                makeSequenceForY=useTimeDistributedOutput,
                                                                                timesteps=timesteps)

    if whichSetArg > 2 or whichSetArg < 0:
        raise ValueError("Invalid which set number {0}".format(whichSetArg))
    set_X = datasets[whichSetArg][0]
    true_values = datasets[whichSetArg][1]

    batch_size = min(classifierParameters['batch_size'], max_batch_size)
    stateful = classifierParameters['stateful'] if 'stateful' in classifierParameters else False
    if stateful:
        whichSetName = ['train', 'valid', 'test'][whichSetArg]
        packagedRowsPerSetDict = datasetParameters['packagedRowsPerSetDict']
        packagedRows = packagedRowsPerSetDict[whichSetName]
        auto_stateful_batch = classifierParameters[
            'auto_stateful_batch'] if 'auto_stateful_batch' in classifierParameters else False
        if auto_stateful_batch:
            batch_size = packagedRows
        else:
            assert batch_size == packagedRows, \
                "You chose stateful but your batch size didn't match the files in the training set"
    # xer = true_values[0::batch_size, 0::2].flatten()
    # yer = true_values[0::batch_size, 1::2].flatten()
    # # plt.plot(yer, xer, marker=None, c='b')
    # plt.subplot(1, 2, 1)
    # plt.scatter(yer, xer, marker='x', c=range(xer.size), cmap=cm.get_cmap("spectral"))
    # plt.ylim([0, 1])
    # plt.xlim([0, 1])
    #
    # newTruth = np.reshape(true_values, newshape=(batch_size, true_values.shape[0] / batch_size, true_values.shape[1]),
    #                       order='F')
    # newTruth = np.reshape(newTruth, newshape=(batch_size, newTruth.shape[1] * \
    # timesteps, newTruth.shape[2] / timesteps))
    # xer = newTruth[0, :, 0]
    # yer = newTruth[0, :, 1]
    # plt.subplot(1, 2, 2)
    # # plt.plot(yer, xer, marker=None, c='g')
    # plt.scatter(yer, xer, marker='x', c=range(xer.size), cmap=cm.get_cmap("spectral"))
    # plt.ylim([0, 1])
    # plt.xlim([0, 1])
    # plt.show()
    print("Making Prediction Regression")
    predicted_y_values = model.predict(set_X, batch_size=batch_size)
    print("Predictions over")

    useTimeDistributedOutput = classifierParameters[
        'useTimeDistributedOutput'] if 'useTimeDistributedOutput' in classifierParameters else False
    if useTimeDistributedOutput:
        if not shapeByRun:
            newShape = (predicted_y_values.shape[0] * predicted_y_values.shape[1], predicted_y_values.shape[2])
            predicted_y_values = np.reshape(predicted_y_values, newShape)
            true_values = np.reshape(true_values, newShape)
        else:
            kerasRowMultiplier = datasetParameters['kerasRowMultiplier']
            totalyColumns = datasetParameters['totalyColumns']
            # xer = predicted_y_values[0::batch_size, :, 0].flatten()
            # yer = predicted_y_values[0::batch_size, :, 1].flatten()
            # # plt.plot(yer, xer, marker=None, c='b')
            # plt.subplot(1, 2, 1)
            # plt.scatter(yer, xer, marker='x', c=range(xer.size), cmap=cm.get_cmap("spectral"))
            # plt.ylim([0, 1])
            # plt.xlim([0, 1])
            predicted_y_values = np.reshape(predicted_y_values, newshape=(
                predicted_y_values.shape[0], predicted_y_values.shape[1] * predicted_y_values.shape[2]))
            predicted_y_values = np.reshape(predicted_y_values,
                                            newshape=(batch_size, predicted_y_values.shape[0] / batch_size,
                                                      predicted_y_values.shape[1]),
                                            order='F')
            predicted_y_values = np.reshape(predicted_y_values,
                                            newshape=(
                                                batch_size, predicted_y_values.shape[1] * timesteps,
                                                predicted_y_values.shape[2] / timesteps))
            # xer = predicted_y_values[0, :, 0]
            # yer = predicted_y_values[0, :, 1]
            # plt.subplot(1, 2, 2)
            # # plt.plot(yer, xer, marker=None, c='g')
            # plt.scatter(yer, xer, marker='x', c=range(xer.size), cmap=cm.get_cmap("spectral"))
            # plt.ylim([0, 1])
            # plt.xlim([0, 1])
            # plt.show()

            # true_values.shape = (run count in timestep x timestep x data dim)
            true_values = np.reshape(true_values, newshape=(batch_size * kerasRowMultiplier, timesteps * totalyColumns))
            # true_values.shape = (run count in timestep x timestep data dim)
            true_values = np.reshape(true_values,
                                     newshape=(batch_size, true_values.shape[0] / batch_size, true_values.shape[1]),
                                     order='F')
            # true_values.shape = (run x slice of run x timestep data dim)
            true_values = np.reshape(true_values,
                                     newshape=(batch_size, true_values.shape[1] * timesteps, true_values.shape[2] / timesteps))
            # true_values.shape =  (run x timesteps x data dim)

    processedDataFolder = os.path.dirname(datasetFileName)
    classLabels = ClassificationUtils.getLabelsForDataset(processedDataFolder, datasetFileName, nClassesTotal=nClassesTotal)

    return predicted_y_values, true_values, classLabels


def makeStatisticsForModel(experimentsFolder, statisticStoreFolder, featureParameters, datasetParameters,
                           classifierParameters, valueMethod=0, useLabels=True, whichSet=1, showFigures=True,
                           modelStoreNameType="best"):
    """
    Make staticstics for a model using the features, datset, and classifier given whose model is already made

    :type experimentsFolder: str
    :param experimentsFolder: Location of the pre made model and where the statistics will be saved

    :type statisticStoreFolder:
    :param statisticStoreFolder:

    :type featureParameters: dict
    :param featureParameters: parameters for the features

    :type datasetParameters: dict
    :param datasetParameters: parameters for the dataset

    :type classifierParameters: dict
    :param classifierParameters: parameters for the classifier

    :type valueMethod: int
    :param valueMethod: Values type to be for classification and thresholding,
    0 = use highest probability, 1 = use ration of higest prob to second, 2 = use difference of highest prob to second

    :type useLabels: bool
    :param useLabels: If labels should be used in charts True or just the class number False

    :type whichSet: int
    :param whichSet: Which of the sets to do the statistics on training=0 validation=1 testing=2

    :type showFigures: bool
    :param showFigures: If you want to see the figures now instead of viewing them on disk later
    (still saves them no matter what)

    :type modelStoreNameType: str
    :param modelStoreNameType: the string prefix of the type of weights to use (best|last)
    """

    valueMethods = ['Probability', 'Probability Ratio', 'Probability Difference']
    setNames = ['Training', 'Validation', 'Testing']
    processedDataFolder = CreateUtils.convertPathToThisOS(datasetParameters['processedDataFolder'])

    if os.path.exists(os.path.join(processedDataFolder, featureParameters['featureSetName'] + '.hf')):
        datasetFile = os.path.join(processedDataFolder, featureParameters['featureSetName'] + '.hf')
    else:
        datasetFile = os.path.join(processedDataFolder, featureParameters['featureSetName'],
                                   datasetParameters['datasetName'] + '.pkl.gz')

    rogueClassesMaster = classifierParameters['rogueClasses']
    if classifierParameters['classifierGoal'] == 'regression':
        (predicted_values_master, true_values_master, classLabelsMaster) = getPred_Values_True_Labels(
            datasetFileName=datasetFile,
            experimentStoreFolder=experimentsFolder,
            whichSetArg=whichSet,
            classifierParameters=classifierParameters,
            datasetParameters=datasetParameters,
            modelStoreNameType=modelStoreNameType)
        if not useLabels:
            classLabelsMaster = None
        yScaleFactor = datasetParameters['y value parameters']['yScaleFactor'] if 'yScaleFactor' in datasetParameters[
            'y value parameters'] else 1.0
        yBias = datasetParameters['y value parameters']['yBias'] if 'yBias' in datasetParameters[
            'y value parameters'] else 0.0
        predicted_values_master = (predicted_values_master / yScaleFactor) - yBias
        true_values_master = (true_values_master / yScaleFactor) - yBias
        thresholdsMaster = RegressionUtils.getStatistics(predicted_values_master, true_values_master,
                                                         setName=setNames[whichSet],
                                                         statisticsStoreFolder=statisticStoreFolder,
                                                         datasetParameters=datasetParameters,
                                                         weightsName=modelStoreNameType)
        if len(rogueClassesMaster) > 0:
            RegressionUtils.rogueAnalysis(thresholdsMaster,
                                          predicted_values_master,
                                          true_values_master,
                                          classLabels=classLabelsMaster,
                                          rogueClasses=rogueClassesMaster,
                                          setName=setNames[whichSet],
                                          statisticsStoreFolder=statisticStoreFolder)
    else:
        thresholdSet = 1
        yValueType = datasetParameters['yValueType']
        plotLabels = (yValueType == 'gpsD' or yValueType == 'particle') and useLabels
        # Grab the validation set in order to calculate EER thresholds
        tupleOutputTemp = getPredictedClasses_Values_TrueClasses_Labels(datasetFileName=datasetFile,
                                                                        experimentStoreFolder=experimentsFolder,
                                                                        valueMethod=valueMethod,
                                                                        whichSetArg=thresholdSet,
                                                                        datasetParameters=datasetParameters,
                                                                        classifierParameters=classifierParameters,
                                                                        modelStoreNameType=modelStoreNameType)
        (predicted_class_master, predicted_values_master, true_class_master, classLabelsMaster) = tupleOutputTemp
        # make EER thresholds with the validation set
        eerThresholdsMaster = ClassificationUtils.getEERThreshold(predicted_class_master,
                                                                  predicted_values_master,
                                                                  true_class_master,
                                                                  rogueClasses=rogueClassesMaster)

        # Now get the test set and test the thresholds I just made
        presentSet = whichSet
        tupleOutputTemp = getPredictedClasses_Values_TrueClasses_Labels(datasetFileName=datasetFile,
                                                                        experimentStoreFolder=experimentsFolder,
                                                                        valueMethod=valueMethod,
                                                                        whichSetArg=presentSet,
                                                                        datasetParameters=datasetParameters,
                                                                        classifierParameters=classifierParameters,
                                                                        modelStoreNameType=modelStoreNameType)
        (predicted_class_master, predicted_values_master, true_class_master, classLabelsMaster) = tupleOutputTemp
        if not useLabels:
            classLabelsMaster = None
        # using the thresholds from before but now the test set plot what this looks like
        ClassificationUtils.plotThresholds(predicted_class_master,
                                           predicted_values_master,
                                           true_class_master,
                                           eerThresholdsMaster,
                                           classLabels=classLabelsMaster,
                                           rogueClasses=rogueClassesMaster,
                                           setName=setNames[presentSet],
                                           valueMethodName=valueMethods[valueMethod],
                                           statisticsStoreFolder=statisticStoreFolder,
                                           plotLabels=plotLabels)

        if len(rogueClassesMaster) > 0:
            ClassificationUtils.rogueAnalysis(eerThresholdsMaster,
                                              predicted_class_master,
                                              predicted_values_master,
                                              true_class_master,
                                              classLabels=classLabelsMaster,
                                              rogueClasses=rogueClassesMaster,
                                              setName=setNames[presentSet],
                                              valueMethodName=valueMethods[valueMethod],
                                              statisticsStoreFolder=statisticStoreFolder)
    if showFigures:
        plt.show()


def convertIndicesToOneHots(y_trainLabel, totalClasses):
    if len(y_trainLabel.shape) == 2:
        y_train = np.zeros((y_trainLabel.shape[0], totalClasses))
        for rower in np.arange(y_trainLabel.shape[0]):
            y_train[rower, y_trainLabel[rower]] = 1
    else:
        y_train = np.zeros((y_trainLabel.shape[0], y_trainLabel.shape[1], totalClasses))
        for rower in np.arange(y_trainLabel.shape[0]):
            for timestep in np.arange(y_trainLabel.shape[1]):
                y_train[rower, timestep, int(y_trainLabel[rower, timestep, 0])] = 1
    return y_train


def getListParameterAndPadLength(parameterName, padLength, classifierParameters, default=0.0):
    parameter = classifierParameters[parameterName] if parameterName in classifierParameters else default
    parameter = [parameter] if (type(parameter) != tuple or type(parameter) != list) else parameter
    assert len(parameter) == 1 or len(parameter) == len(padLength), "Bad number of {0}".format(parameterName)
    parameter = parameter * len(padLength) if len(parameter) == 1 else parameter
    return parameter


def kerasClassifier_parameterized(featureParameters, datasetParameters, classifierParameters, forceRebuildModel=False,
                                  showModelAsFigure=False):
    """
    Train a Logistic Regression model using the features, datset, and classifier parameters given

    :type featureParameters: dict
    :param featureParameters: parameters for the features

    :type datasetParameters: dict
    :param datasetParameters: parameters for the dataset

    :type classifierParameters: dict
    :param classifierParameters: parameters for the classifier

    :type forceRebuildModel: bool
    :param forceRebuildModel: forces to rebuild the model and train it again

    :type showModelAsFigure: bool
    :param showModelAsFigure: do you want to show the model right before running
    """

    rawDataFolder = CreateUtils.convertPathToThisOS(datasetParameters['rawDataFolder'])
    processedDataFolder = CreateUtils.convertPathToThisOS(datasetParameters['processedDataFolder'])

    if os.path.exists(os.path.join(processedDataFolder, featureParameters['featureSetName'] + '.hf')):
        datasetFile = os.path.join(processedDataFolder, featureParameters['featureSetName'] + '.hf')
    else:
        datasetFile = os.path.join(processedDataFolder, featureParameters['featureSetName'],
                                   datasetParameters['datasetName'] + '.pkl.gz')

    experimentsFolder = os.path.join(rawDataFolder, "Data Experiments", featureParameters['featureSetName'],
                                     datasetParameters['datasetName'],
                                     classifierParameters['classifierType'], classifierParameters['classifierSetName'])

    modelStoreWeightsBestFilePathFullTemp = os.path.join(experimentsFolder, 'best_modelWeights.h5')
    if not os.path.exists(modelStoreWeightsBestFilePathFullTemp) or forceRebuildModel:
        makeSequences = datasetParameters['makeSequences'] if 'makeSequences' in datasetParameters else False
        alternateRowsForKeras = datasetParameters[
            'alternateRowsForKeras'] if 'alternateRowsForKeras' in datasetParameters else False
        if alternateRowsForKeras:
            timesteps = datasetParameters['timestepsPerKerasBatchRow']
            makeSequencesForX = True
        elif makeSequences:
            timesteps = datasetParameters['timestepsPerSequence']
            makeSequencesForX = True
        else:
            timesteps = 0
            makeSequencesForX = False
        useTimeDistributedOutput = classifierParameters['useTimeDistributedOutput'] if 'useTimeDistributedOutput' in classifierParameters else False
        onlyBuildModel = classifierParameters['onlyBuildModel'] if 'onlyBuildModel' in classifierParameters else False

        if classifierParameters['classifierGoal'] == 'regression':
            (datasets,
             inputs,
             outputs,
             max_batch_size) = RegressionUtils.load_data(datasetFile,
                                                         rogueClasses=(),
                                                         makeSharedData=False,
                                                         makeSequenceForX=makeSequencesForX,
                                                         makeSequenceForY=useTimeDistributedOutput,
                                                         timesteps=timesteps)
            (X_train, X_valid, X_test) = (datasets[0][0], datasets[1][0], datasets[2][0])
            (y_train, y_valid, y_test) = (datasets[0][1], datasets[1][1], datasets[2][1])
        else:
            (datasets,
             inputs,
             outputs,
             max_batch_size) = ClassificationUtils.load_data(datasetFile,
                                                             rogueClasses=(),
                                                             makeSharedData=False,
                                                             makeSequenceForX=makeSequencesForX,
                                                             makeSequenceForY=useTimeDistributedOutput,
                                                             timesteps=timesteps)
            (X_train, X_valid, X_test) = (datasets[0][0], datasets[1][0], datasets[2][0])
            lossType = classifierParameters['lossType']
            if lossType in ['falsePositiveRate']:
                (y_train, y_valid, y_test) = (datasets[0][1], datasets[1][1], datasets[2][1])
            else:
                y_train = convertIndicesToOneHots(datasets[0][1], outputs)
                y_valid = convertIndicesToOneHots(datasets[1][1], outputs)
                y_test = convertIndicesToOneHots(datasets[2][1], outputs)

        imageShape = featureParameters['imageShape']
        useMetadata = datasetParameters['useMetadata'] if 'useMetadata' in datasetParameters else False
        metadataShape = datasetParameters['metadataShape'] if 'metadataShape' in datasetParameters else (0,)
        metadata_dim = np.product(metadataShape) if useMetadata else 0
        data_dim = np.product(imageShape) + metadata_dim

        lstm_layers_sizes = classifierParameters['lstm_layers_sizes']
        n_epochs = classifierParameters['n_epochs']
        batch_size = min(classifierParameters['batch_size'], max_batch_size)
        hidden_layers_sizes = classifierParameters['hidden_layers_sizes']
        finalActivationType = classifierParameters['finalActivationType']
        stateful = classifierParameters['stateful'] if 'stateful' in classifierParameters else False
        consume_less = classifierParameters['consume_less'] if 'consume_less' in classifierParameters else 'cpu'
        trainLSTM = classifierParameters['trainLSTM'] if 'trainLSTM' in classifierParameters else True
        trainMLP = classifierParameters['trainMLP'] if 'trainMLP' in classifierParameters else True

        useKalman = classifierParameters['useKalman'] if 'useKalman' in classifierParameters else False

        useAppendMLPLayers = classifierParameters['useAppendMLPLayers'] if 'useAppendMLPLayers' in classifierParameters else False
        appendExpectedInput = classifierParameters['appendExpectedInput'] if 'appendExpectedInput' in classifierParameters else 1

        activations = getListParameterAndPadLength('activations', lstm_layers_sizes, classifierParameters)
        inner_activations = getListParameterAndPadLength('inner_activations', lstm_layers_sizes, classifierParameters)
        hidden_activations = getListParameterAndPadLength('hidden_activations', hidden_layers_sizes,
                                                          classifierParameters)
        dropout_W = getListParameterAndPadLength('dropout_W', lstm_layers_sizes, classifierParameters)
        dropout_U = getListParameterAndPadLength('dropout_U', lstm_layers_sizes, classifierParameters)
        dropout_LSTM = getListParameterAndPadLength('dropout_LSTM', lstm_layers_sizes, classifierParameters)
        dropout_Hidden = getListParameterAndPadLength('dropout_Hidden', hidden_layers_sizes, classifierParameters)

        W_regularizer_l1_LSTM = getListParameterAndPadLength('W_regularizer_l1)LSTM', lstm_layers_sizes,
                                                             classifierParameters)
        U_regularizer_l1_LSTM = getListParameterAndPadLength('U_regularizer_l1_LSTM', lstm_layers_sizes,
                                                             classifierParameters)
        b_regularizer_l1_LSTM = getListParameterAndPadLength('b_regularizer_l1_LSTM', lstm_layers_sizes,
                                                             classifierParameters)
        W_regularizer_l2_LSTM = getListParameterAndPadLength('W_regularizer_l2_LSTM', lstm_layers_sizes,
                                                             classifierParameters)
        U_regularizer_l2_LSTM = getListParameterAndPadLength('U_regularizer_l2_LSTM', lstm_layers_sizes,
                                                             classifierParameters)
        b_regularizer_l2_LSTM = getListParameterAndPadLength('b_regularizer_l2_LSTM', lstm_layers_sizes,
                                                             classifierParameters)

        W_regularizer_l1_hidden = getListParameterAndPadLength("W_regularizer_l1_hidden", hidden_layers_sizes,
                                                               classifierParameters)
        b_regularizer_l1_hidden = getListParameterAndPadLength("b_regularizer_l1_hidden", hidden_layers_sizes,
                                                               classifierParameters)
        W_regularizer_l2_hidden = getListParameterAndPadLength("W_regularizer_l2_hidden", hidden_layers_sizes,
                                                               classifierParameters)
        b_regularizer_l2_hidden = getListParameterAndPadLength("b_regularizer_l2_hidden", hidden_layers_sizes,
                                                               classifierParameters)

        if stateful:
            whichSetName = ['train', 'valid', 'test'][0]
            packagedRowsPerSetDict = datasetParameters['packagedRowsPerSetDict']
            packagedRows = packagedRowsPerSetDict[whichSetName]
            auto_stateful_batch = classifierParameters[
                'auto_stateful_batch'] if 'auto_stateful_batch' in classifierParameters else False
            if auto_stateful_batch:
                batch_size = packagedRows
            else:
                assert batch_size == packagedRows, \
                    "You chose stateful but your batch size didn't match the files in the training set"

        useWaveletTransform = classifierParameters['useWaveletTransform'] \
            if 'useWaveletTransform' in classifierParameters else False
        waveletBanks = classifierParameters['waveletBanks'] if 'waveletBanks' in classifierParameters else 1
        kValues = classifierParameters['kValues'] if 'kValues' in classifierParameters else None
        sigmaValues = classifierParameters['sigmaValues'] if 'sigmaValues' in classifierParameters else None
        maxWindowSize = classifierParameters['maxWindowSize'] if 'maxWindowSize' in classifierParameters else 10000
        model = Sequential()

        if useWaveletTransform:
            waveletLayer = WaveletTransformLayer(batch_input_shape=(batch_size, timesteps, 1, data_dim),
                                                 output_dim=waveletBanks,
                                                 maxWindowSize=maxWindowSize,
                                                 kValues=kValues,
                                                 sigmaValues=sigmaValues,
                                                 useConvolution=False,  # without this it does way too much computation
                                                 name="Wavelet Layer")
            model.add(waveletLayer)
            # I am padding out the input_channel dimension to be 1 because I only have one input channel
            (X_train, X_valid, X_test) = (X_train[:, :, None, :], X_valid[:, :, None, :], X_test[:, :, None, :])

        for layerIndex in range(1, 1 + len(lstm_layers_sizes)):
            lstmIndex = layerIndex - 1
            return_sequences = (layerIndex != len(lstm_layers_sizes)) or useTimeDistributedOutput
            W_regularizer_LSTM = WeightRegularizer(l1=W_regularizer_l1_LSTM[lstmIndex],
                                                   l2=W_regularizer_l2_LSTM[lstmIndex])
            U_regularizer_LSTM = WeightRegularizer(l1=U_regularizer_l1_LSTM[lstmIndex],
                                                   l2=U_regularizer_l2_LSTM[lstmIndex])
            b_regularizer_LSTM = WeightRegularizer(l1=b_regularizer_l1_LSTM[lstmIndex],
                                                   l2=b_regularizer_l2_LSTM[lstmIndex])
            model.add(
                LSTM(batch_input_shape=(batch_size, timesteps, data_dim),
                     output_dim=lstm_layers_sizes[lstmIndex],
                     activation=activations[lstmIndex],
                     inner_activation=inner_activations[lstmIndex],
                     return_sequences=return_sequences,
                     consume_less=consume_less,
                     stateful=stateful,
                     dropout_U=dropout_U[lstmIndex],
                     dropout_W=dropout_W[lstmIndex],
                     W_regularizer=W_regularizer_LSTM,
                     U_regularizer=U_regularizer_LSTM,
                     b_regularizer=b_regularizer_LSTM,
                     trainable=trainLSTM,
                     name="LSTM Layer {0}".format(layerIndex)))
            if (dropout_LSTM[lstmIndex]) > 0:
                model.add(Dropout(dropout_LSTM[lstmIndex], name="Dropout Layer {0}".format(layerIndex)))

        for layerIndex in range(1 + len(lstm_layers_sizes), 1 + len(lstm_layers_sizes) + len(hidden_layers_sizes)):
            hiddenIndex = layerIndex - (1 + len(lstm_layers_sizes))
            W_regularizer_hidden = WeightRegularizer(l1=W_regularizer_l1_hidden[hiddenIndex],
                                                     l2=W_regularizer_l2_hidden[hiddenIndex])
            b_regularizer_hidden = WeightRegularizer(l1=b_regularizer_l1_hidden[hiddenIndex],
                                                     l2=b_regularizer_l2_hidden[hiddenIndex])
            thisDenseLayer = Dense(hidden_layers_sizes[hiddenIndex],
                                   W_regularizer=W_regularizer_hidden,
                                   b_regularizer=b_regularizer_hidden,
                                   activation=hidden_activations[hiddenIndex],
                                   trainable=trainMLP,
                                   name="Dense Layer {0} {1}".format(layerIndex, hidden_activations[hiddenIndex]))
            if useTimeDistributedOutput:
                thisDenseLayer = TimeDistributed(thisDenseLayer, name="TD {0}".format(thisDenseLayer.name))
            model.add(thisDenseLayer)
            if (dropout_Hidden[hiddenIndex]) > 0:
                model.add(Dropout(dropout_Hidden[hiddenIndex], name="Dropout Layer {0}".format(layerIndex)))

        if useAppendMLPLayers:
            finalOutputDataDimension = appendExpectedInput
        else:
            finalOutputDataDimension = y_train.shape[1] if useTimeDistributedOutput is False else y_train.shape[2]
        lastDenseLayer = Dense(finalOutputDataDimension,
                               trainable=trainMLP,
                               name="Dense Layer Output")
        if useTimeDistributedOutput:
            lastDenseLayer = TimeDistributed(lastDenseLayer, name="TD {0}".format(lastDenseLayer.name))
        model.add(lastDenseLayer)
        model.add(Activation(finalActivationType, name="{0} Layer Output".format(finalActivationType)))

        loadPreviousModelWeightsForTraining = classifierParameters['loadPreviousModelWeightsForTraining'] \
            if 'loadPreviousModelWeightsForTraining' in classifierParameters else False
        if loadPreviousModelWeightsForTraining:
            loadWeightsFilePath = classifierParameters['loadWeightsFilePath'] if 'loadWeightsFilePath' in classifierParameters else ''
            if loadWeightsFilePath == '':
                loadWeightsFilePath = os.path.join(experimentsFolder, 'best_modelWeights.h5')
            previousWeightsPath = loadWeightsFilePath
            # if useKalman and addKalmanLayerToWeights:
            #     previousWeightsWithKalmanPath = os.path.join(os.path.dirname(previousWeightsPath),
            #                                                  'kalman' + os.path.basename(previousWeightsPath))
            #     shutil.copyfile(previousWeightsPath, previousWeightsWithKalmanPath)
            #     previousWeightsPath = previousWeightsWithKalmanPath
            #     with h5py.File(previousWeightsPath, mode='a') as f:
            #         kalmanGroup = f.create_group("/model_weights/Kalman Layer/")
            #         states_batch = np.repeat(x_t0[np.newaxis, :], batch_size, axis=0)
            #         if len(states_batch.shape) > 2:
            #             states_batch = np.squeeze(states_batch)
            #         kalmanGroup['statesX'] = states_batch
            #         kalmanGroup['PMatrix'] = np.repeat(P_t0[np.newaxis, :, :], batch_size, axis=0)
            #         kalmanGroup['phiMatrix'] = phi
            #         kalmanGroup['BMatrix'] = B
            #         kalmanGroup['CMatrix'] = C
            #         kalmanGroup['DMatrix'] = D
            #         kalmanGroup['QMatrix'] = Q
            #         kalmanGroup['HMatrix'] = H
            #         kalmanGroup['RMatrix'] = R
            #         kalmanGroup.attrs['weight_names'] = ['statesX', 'PMatrix', 'phiMatrix', 'BMatrix', 'CMatrix',
            #                                              'DMatrix', 'QMatrix', 'HMatrix', 'RMatrix']
            #         layer_names = f['model_weights'].attrs['layer_names']
            #         layer_names = np.hstack((layer_names[:-1], np.array(["Kalman Layer"]), layer_names[-1:]))
            #         f['model_weights'].attrs['layer_names'] = layer_names
            if os.path.exists(previousWeightsPath):
                print ("Loading previous weights \n{0}".format(previousWeightsPath))
                model.load_weights(previousWeightsPath)
                shutil.copyfile(previousWeightsPath, os.path.join(experimentsFolder, os.path.basename('loaded_modelWeights.h5')))
            else:
                raise ValueError("The previous weights file does not exist \n{0}".format(previousWeightsPath))

        if useAppendMLPLayers:
            append_layers_sizes = classifierParameters['append_layers_sizes']
            append_activations = getListParameterAndPadLength('append_activations', append_layers_sizes, classifierParameters)
            dropout_Append = getListParameterAndPadLength('dropout_append', append_layers_sizes, classifierParameters)
            appendWeightsFile = getListParameterAndPadLength('appendWeightsFile', append_layers_sizes, classifierParameters, default=None)
            trainAppend = classifierParameters['trainAppend']

            totalLayersSoFar = 1 + len(lstm_layers_sizes) + len(hidden_layers_sizes)
            for layerIndex in range(totalLayersSoFar, totalLayersSoFar + len(append_layers_sizes)):
                appendIndex = layerIndex - totalLayersSoFar

                if appendWeightsFile[appendIndex] is not None:
                    thisWeights = np.genfromtxt(appendWeightsFile[appendIndex], delimiter=',', skip_header=1)
                    thisBiases = np.zeros((thisWeights.shape[1]))
                    thisWeightsList = [thisWeights, thisBiases]
                else:
                    thisWeightsList = None

                thisDenseLayer = Dense(append_layers_sizes[appendIndex],
                                       trainable=trainAppend,
                                       weights=thisWeightsList,
                                       activation=append_activations[appendIndex],
                                       name="Append Dense Layer {0} {1}".format(layerIndex, append_activations[appendIndex]))
                if useTimeDistributedOutput:
                    thisDenseLayer = TimeDistributed(thisDenseLayer, name="TD {0}".format(thisDenseLayer.name))
                model.add(thisDenseLayer)
                if (dropout_Append[appendIndex]) > 0:
                    model.add(Dropout(dropout_Append[appendIndex], name="Dropout Layer {0}".format(layerIndex)))

        if useKalman:
            trainMatrices = classifierParameters['trainMatrices'] if 'trainMatrices' in classifierParameters else None
            matrixIsDiscrete = classifierParameters[
                'matrixIsDiscrete'] if 'matrixIsDiscrete' in classifierParameters else {}
            x_t0 = classifierParameters['x_t0']
            P_t0 = classifierParameters['P_t0']
            dt = featureParameters['feature parameters']['windowTimeLength']
            F = classifierParameters['F']
            B = classifierParameters['B']
            C = classifierParameters['C']
            D = classifierParameters['D']
            G = classifierParameters['G']
            Q = classifierParameters['Q']
            H = classifierParameters['H']
            R = classifierParameters['R']
            (phi, Bd, Cd, Dd, Qd) = getDiscreteSystem(F, B, C, D, G, Q, dt, matrixIsDiscrete)
            kalmanLayer = KalmanFilterLayer(output_dim=Cd.shape[0],
                                            statesXInitArg=x_t0,
                                            PInitArg=P_t0,
                                            phi=phi, B=Bd, C=Cd, D=Dd, Q=Qd, H=H, R=R,
                                            trainMatrices=trainMatrices,
                                            name="Kalman Layer")
            if useTimeDistributedOutput:
                kalmanLayer = TimeDistributed(kalmanLayer, name="TD {0}".format(kalmanLayer.name))
            model.add(kalmanLayer)
        # else:
        #     (x_t0, P_t0, phi, B, C, D, Q, H, R) = [None, ] * 9

        # Visualizations
        if showModelAsFigure:
            showModel(model)
        modelImage = os.path.join(experimentsFolder, 'modelImage.png')
        saveModelToFile(model, modelImage)

        print ("Compiling Model")
        compileAModel(model,
                      classifierParameters=classifierParameters,
                      datasetParameters=datasetParameters)

        modelStoreFilePathFullTemp = os.path.join(experimentsFolder, 'model.json')
        with open(modelStoreFilePathFullTemp, 'w') as modelStoreFile:
            json_string = model.to_json()
            modelStoreFile.write(json_string)

        # Callbacks
        myCallbacks = []
        # Best Val_loss model
        modelStoreWeightsBestFilePathFullTemp = os.path.join(experimentsFolder, 'best_modelWeights.h5')
        checkpointerBest = keras.callbacks.ModelCheckpoint(filepath=modelStoreWeightsBestFilePathFullTemp,
                                                           verbose=1,
                                                           save_best_only=True)
        myCallbacks.append(checkpointerBest)
        # Best Loss model
        modelStoreWeightsBestLossFilePathFullTemp = os.path.join(experimentsFolder, 'bestLoss_modelWeights.h5')
        checkpointerBestLoss = keras.callbacks.ModelCheckpoint(filepath=modelStoreWeightsBestLossFilePathFullTemp,
                                                               verbose=0,
                                                               save_best_only=True,
                                                               monitor='loss')
        myCallbacks.append(checkpointerBestLoss)
        # Last model
        modelStoreWeightsLastFilePathFullTemp = os.path.join(experimentsFolder, 'last_modelWeights.h5')
        checkpointerLast = keras.callbacks.ModelCheckpoint(filepath=modelStoreWeightsLastFilePathFullTemp,
                                                           verbose=0,
                                                           save_best_only=False)
        myCallbacks.append(checkpointerLast)
        # Logging
        csvFilePathFull = os.path.join(experimentsFolder, 'trainingLog.csv')
        csvLoggerCallback = CSVLogger(csvFilePathFull, separator=',', append=False)
        myCallbacks.append(csvLoggerCallback)
        # Reduce Learning Rate
        reduceLearningRate = classifierParameters[
            'reduceLearningRate'] if 'reduceLearningRate' in classifierParameters else False
        if reduceLearningRate:
            rlrMonitor = classifierParameters['rlrMonitor'] if 'rlrMonitor' in classifierParameters else 'val_loss'
            rlrFactor = classifierParameters['rlrFactor'] if 'rlrFactor' in classifierParameters else 0.1
            rlrPatience = classifierParameters['rlrPatience'] if 'rlrPatience' in classifierParameters else 10
            rlrCooldown = classifierParameters['rlrCooldown'] if 'rlrCooldown' in classifierParameters else 0
            rlrEpsilon = classifierParameters['rlrEpsilon'] if 'rlrEpsilon' in classifierParameters else 1e-4
            rlrCallback = ReduceLROnPlateau(monitor=rlrMonitor,
                                            factor=rlrFactor,
                                            patience=rlrPatience,
                                            verbose=1,
                                            mode='auto',
                                            epsilon=rlrEpsilon, cooldown=rlrCooldown)
            myCallbacks.append(rlrCallback)
        if not onlyBuildModel:
            try:
                model.fit(X_train, y_train,
                          batch_size=batch_size,
                          nb_epoch=n_epochs,
                          validation_data=(X_valid, y_valid),
                          callbacks=myCallbacks,
                          verbose=2)
            except KeyboardInterrupt:
                print ("\nUser stopped training")
        else:
            model.save_weights(modelStoreWeightsBestFilePathFullTemp, overwrite=True)
            model.save_weights(modelStoreWeightsBestLossFilePathFullTemp, overwrite=True)
            model.save_weights(modelStoreWeightsLastFilePathFullTemp, overwrite=True)

        score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
        print("The final test score is {0}".format(score))
