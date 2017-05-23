import os
import re

import csv
import warnings
from collections import Iterable, OrderedDict
import shutil
import json

import matplotlib.pylab as plt
import numpy as np

import theano.tensor as T

import keras.callbacks
from keras.callbacks import Callback
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, TimeDistributed, LSTM, MaxoutDense
from keras.engine.topology import Container
from keras.optimizers import rmsprop, adam
from keras.regularizers import WeightRegularizer
import keras.constraints
import keras.backend as K

import ClassificationUtils
import RegressionUtils
from KalmanFilter import getDiscreteSystem
from KalmanFilter import KalmanFilterLayer
from WaveletTransformLayer import WaveletTransformLayer
from TeacherForcing import TeacherForcingModel
import CreateUtils
import tictoc

timer = tictoc.tictoc()

try:
    # pydot-ng is a fork of pydot that is better maintained
    import pydot_ng as pydot
except ImportError:
    # fall back on pydot if necessary
    import pydot
if not pydot.find_graphviz():
    raise RuntimeError('Failed to import pydot. You must install pydot'
                       ' and graphviz for `pydotprint` to work.')


def getContainerLayers(layer):
    if issubclass(type(layer), Container):
        subLayers = []
        for lay in layer.layers:
            subLayers += getContainerLayers(lay)
    else:
        subLayers = [layer]
    return subLayers


def model_to_dot2(model, show_shapes=False, show_layer_names=True, expand_containers=False):
    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    if model.__class__.__name__ == 'Sequential':
        if not model.built:
            model.build()
        model = model.model
    if not expand_containers:
        layers = model.layers
    else:
        layers = getContainerLayers(model)

    # first, populate the nodes of the graph
    for layer in layers:
        layer_id = str(id(layer))
        if show_layer_names:
            label = str(layer.name) + ' (' + layer.__class__.__name__ + ')'
        else:
            label = layer.__class__.__name__

        if show_shapes:
            # Build the label that will actually contain a table with the
            # input/output
            try:
                outputlabels = str(layer.output_shape)
            except:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, inputlabels, outputlabels)

        node = pydot.Node(layer_id, label=label)
        dot.add_node(node)

    # second, add the edges
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model.container_nodes:
                # add edges
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
    return dot


def plot2(model, to_file='model.png', show_shapes=False, show_layer_names=True, expand_containers=False):
    dot = model_to_dot2(model, show_shapes, show_layer_names, expand_containers=expand_containers)
    dot.write_png(to_file)


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
                                                  whichSetName='valid',
                                                  datasetParameters=None,
                                                  classifierParameters=None,
                                                  modelStoreNameType="best",
                                                  runByShape=False,
                                                  returnClassProbabilities=False):
    # get parameters
    totalyColumns = datasetParameters['totalyColumns']

    # sequence params
    makeSequences = datasetParameters['makeSequences'] if 'makeSequences' in datasetParameters else False
    timestepsOfASample = datasetParameters['timestepsPerSequence'] if 'timestepsPerSequence' in datasetParameters else None

    # packing
    rowPackagingStyle = datasetParameters['rowPackagingStyle'] if 'rowPackagingStyle' in datasetParameters else None
    alternateRowsForKeras = datasetParameters['alternateRowsForKeras'] if 'alternateRowsForKeras' in datasetParameters else False
    kerasRowMultiplier = datasetParameters['kerasRowMultiplier'] if 'kerasRowMultiplier' in datasetParameters else 1
    timestepsPerKerasBatchRow = datasetParameters['timestepsPerKerasBatchRow'] if 'timestepsPerKerasBatchRow' in datasetParameters else None
    packagedRowsPerSetDict = datasetParameters['packagedRowsPerSetDict'] if 'packagedRowsPerSetDict' in datasetParameters else {whichSetName: None}
    packagedColumnsPerSetDict = datasetParameters['packagedColumnsPerSetDict'] if 'packagedColumnsPerSetDict' in datasetParameters else {
        whichSetName: None}
    assert whichSetName in packagedRowsPerSetDict, "The set name {0} is not in the dataset".format(whichSetName)
    packagedRows = packagedRowsPerSetDict[whichSetName]
    packagedColumns = packagedColumnsPerSetDict[whichSetName]

    if rowPackagingStyle is not None:
        if alternateRowsForKeras:
            timesteps_load = timestepsPerKerasBatchRow
        else:
            timesteps_load = packagedColumns
        makeSequencesForX = True
    else:
        timesteps_load = 1
        makeSequencesForX = False

    useTimeDistributedOutput = classifierParameters['useTimeDistributedOutput'] if 'useTimeDistributedOutput' in classifierParameters else False
    datasets, inputs, outputs, max_batch_size = ClassificationUtils.load_data(datasetFileName,
                                                                              rogueClasses=(),
                                                                              makeSharedData=False,
                                                                              makeSequenceForX=makeSequencesForX,
                                                                              makeSequenceForY=useTimeDistributedOutput,
                                                                              timesteps=timesteps_load,
                                                                              setNames=[whichSetName])
    X_test = datasets[0][0]
    if 'useWaveletTransform' in classifierParameters and classifierParameters['useWaveletTransform'] is True:
        X_test = X_test[:, :, None, :]
    set_X = X_test
    trueValues = datasets[0][1]

    # get parameters that depended on size of data
    batch_size = min(classifierParameters['batch_size'], max_batch_size)
    stateful = classifierParameters['stateful'] if 'stateful' in classifierParameters else False

    if stateful:
        auto_stateful_batch = classifierParameters['auto_stateful_batch'] if 'auto_stateful_batch' in classifierParameters else False
        if auto_stateful_batch:
            batch_size = packagedRows
        else:
            assert batch_size == packagedRows, "You chose stateful but your batch size didn't match the files in the training set"

    trueRowsPerSetDict = datasetParameters['trueRowsPerSetDict'] if 'trueRowsPerSetDict' in datasetParameters else {}
    trueRows = trueRowsPerSetDict[whichSetName] if whichSetName in trueRowsPerSetDict else batch_size

    print("Loading model")
    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'model.json'.format(modelStoreNameType))
    if not os.path.exists(modelStoreFilePathFullTemp):
        modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.json'.format(modelStoreNameType))

    with open(modelStoreFilePathFullTemp, 'r') as modelStoreFile:
        json_string = modelStoreFile.read()
    json_string = changeBatchInputShapeOfModel(json_string, batch_size)
    model = model_from_json(json_string, custom_objects={"KalmanFilterLayer": KalmanFilterLayer, "WaveletTransformLayer": WaveletTransformLayer})
    modelStoreWeightsFilePathFullTemp = os.path.join(experimentStoreFolder, '{0}_modelWeights.h5'.format(modelStoreNameType))
    model.load_weights(modelStoreWeightsFilePathFullTemp)

    compileAModel(model,
                  classifierParameters=classifierParameters,
                  datasetParameters=datasetParameters)

    print("Making Prediction Classes")
    predicted_probabilities = model.predict_proba(set_X, batch_size=batch_size, verbose=2)
    print("Predictions over")
    modelOutputClasses = predicted_probabilities.shape[-1]
    assert modelOutputClasses == outputs, "The dataset and model were configured for a different number of classes"
    # reshape predicted_probabilities to be rows and indices
    predicted_probabilities = np.reshape(predicted_probabilities,
                                         newshape=(
                                             predicted_probabilities.shape[0] * predicted_probabilities.shape[1], predicted_probabilities.shape[2]))

    trueValues = np.reshape(trueValues, (trueValues.shape[0] * trueValues.shape[1], trueValues.shape[2]))
    # trueValues = totalSamples x output data dim

    # both True and predicted shape = (batch_size * kerasRowMultiplier * timesteps x totalColumnsY)

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

    if rowPackagingStyle is not None:
        reshapeSets = [trueValues, predicted_probabilities, predicted_class, predictedValues]
        for setIndex, final_dim in enumerate([totalyColumns, outputs, 1, 1]):
            reshapeSet = reshapeSets[setIndex]
            if alternateRowsForKeras:
                # shape = (batch_size * kerasRowMultiplier * timestepsPerKerasBatchRow x final_dim)
                reshapeSet = np.reshape(reshapeSet, newshape=(batch_size * kerasRowMultiplier, timestepsPerKerasBatchRow * final_dim))
                # shape = (run count in timestep x timestep data dim)
                reshapeSet = np.reshape(reshapeSet, newshape=(batch_size, kerasRowMultiplier, timestepsPerKerasBatchRow * final_dim), order='F')
                # shape = (run x slice of run x timestep data dim)
                reshapeSet = np.reshape(reshapeSet, newshape=(batch_size, kerasRowMultiplier * timestepsPerKerasBatchRow, final_dim))
                # shape = (run x timestep x data dim)
            else:
                # shape = (batch_size * timestep x final_dim)
                reshapeSet = np.reshape(reshapeSet, newshape=(batch_size, packagedColumns, final_dim))
                # shape = (run x timestep x data dim)

            # slice out the actual rows
            reshapeSet = reshapeSet[0:trueRows]
            # reshape output as desired
            if not runByShape:
                reshapeSet = np.reshape(reshapeSet, newshape=(trueRows * kerasRowMultiplier * timestepsPerKerasBatchRow, final_dim))
            reshapeSets[setIndex] = reshapeSet
        (trueValues, predicted_probabilities, predicted_class, predictedValues) = reshapeSets
        predicted_class = np.squeeze(predicted_class, axis=-1)
        predictedValues = np.squeeze(predictedValues, axis=-1)

    if returnClassProbabilities is False:
        returnTuple = (predicted_class, predictedValues, trueValues, classLabels, outputs)
    else:
        returnTuple = (predicted_class, predictedValues, trueValues, classLabels, outputs, predicted_probabilities)
    return returnTuple


def getPred_Values_True_Labels(datasetFileName='mnist.pkl.gz',
                               experimentStoreFolder='',
                               whichSetName='valid',
                               datasetParameters=None,
                               classifierParameters=None,
                               modelStoreNameType="best",
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
                                                                                timesteps=timesteps,
                                                                                setNames=[whichSetName])

    set_X = datasets[0][0]
    true_values = datasets[0][1]

    batch_size = min(classifierParameters['batch_size'], max_batch_size)
    stateful = classifierParameters['stateful'] if 'stateful' in classifierParameters else False
    if stateful:
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


def makeStatisticsForModel(experimentsFolder,
                           statisticStoreFolder,
                           featureParameters,
                           datasetParameters,
                           classifierParameters,
                           valueMethod=0,
                           useLabels=True,
                           whichSetName='valid',
                           whichSetNameStat='valid',
                           showFigures=True,
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

    :type whichSetName: str
    :param whichSetName: Which of the sets to do the thresholding on

    :type whichSetNameStat: str
    :param whichSetNameStat: Which of the sets to do the statistics

    :type showFigures: bool
    :param showFigures: If you want to see the figures now instead of viewing them on disk later
    (still saves them no matter what)

    :type modelStoreNameType: str
    :param modelStoreNameType: the string prefix of the type of weights to use (best|last)
    """

    valueMethodName = ClassificationUtils.valueMethodNames[valueMethod]

    # get datasetFile
    datasetFile = CreateUtils.getDatasetFile(featureSetName=featureParameters['featureSetName'], datasetName=datasetParameters['datasetName'])

    # get statDatasetFile
    statDatasetConfigFileName = CreateUtils.getDatasetStatConfigFileName(statisticsFolder=statisticStoreFolder)
    statDatasetParameters = CreateUtils.loadConfigFile(statDatasetConfigFileName)
    statDatasetFile = CreateUtils.getDatasetFile(featureParameters['featureSetName'], statDatasetParameters['datasetName'])

    rogueClassesMaster = classifierParameters['rogueClasses']
    if classifierParameters['classifierGoal'] == 'regression':
        tupleOutputTemp = getPred_Values_True_Labels(datasetFileName=statDatasetFile,
                                                     experimentStoreFolder=experimentsFolder,
                                                     whichSetName=whichSetNameStat,
                                                     classifierParameters=classifierParameters,
                                                     datasetParameters=statDatasetParameters,
                                                     modelStoreNameType=modelStoreNameType)
        (predicted_values_master, true_values_master, classLabelsMaster) = tupleOutputTemp

        if not useLabels:
            classLabelsMaster = None
        yScaleFactor = statDatasetParameters['y value parameters']['yScaleFactor'] if 'yScaleFactor' in statDatasetParameters[
            'y value parameters'] else 1.0
        yBias = statDatasetParameters['y value parameters']['yBias'] if 'yBias' in statDatasetParameters['y value parameters'] else 0.0
        predicted_values_master = (predicted_values_master / yScaleFactor) - yBias
        true_values_master = (true_values_master / yScaleFactor) - yBias
        thresholdsMaster = RegressionUtils.getStatistics(predicted_values_master, true_values_master,
                                                         setName=whichSetNameStat,
                                                         statisticsStoreFolder=statisticStoreFolder,
                                                         datasetParameters=statDatasetParameters,
                                                         weightsName=modelStoreNameType)
        if len(rogueClassesMaster) > 0:
            RegressionUtils.rogueAnalysis(thresholdsMaster,
                                          predicted_values_master,
                                          true_values_master,
                                          classLabels=classLabelsMaster,
                                          rogueClasses=rogueClassesMaster,
                                          setName=whichSetNameStat,
                                          statisticsStoreFolder=statisticStoreFolder)
    else:
        yValueType = statDatasetParameters['yValueType']
        plotLabels = (yValueType == 'gpsD' or yValueType == 'particle') and useLabels
        useVoroni = yValueType == 'particle'
        # Grab the dataset used in the model in order to calculate EER thresholds
        tupleOutputTemp = getPredictedClasses_Values_TrueClasses_Labels(datasetFileName=datasetFile,
                                                                        experimentStoreFolder=experimentsFolder,
                                                                        valueMethod=valueMethod,
                                                                        whichSetName=whichSetName,
                                                                        datasetParameters=datasetParameters,
                                                                        classifierParameters=classifierParameters,
                                                                        modelStoreNameType=modelStoreNameType)
        (predicted_class_master, predicted_values_master, true_class_master, classLabelsMaster, totalOuputClasses) = tupleOutputTemp
        # make EER thresholds with the dataset used in the model
        thresholdsSaveFile = os.path.join(experimentsFolder, 'eerthresholds_{0}.yaml'.format(whichSetName))
        eerThresholdsMaster = ClassificationUtils.getEERThreshold(predicted_class_master,
                                                                  predicted_values_master,
                                                                  true_class_master,
                                                                  rogueClasses=rogueClassesMaster,
                                                                  classLabels=classLabelsMaster,
                                                                  nTotalClassesArg=totalOuputClasses,
                                                                  thresholdsSaveFile=thresholdsSaveFile)

        # Now get the stat set to show the results
        tupleOutputTemp = getPredictedClasses_Values_TrueClasses_Labels(datasetFileName=statDatasetFile,
                                                                        experimentStoreFolder=experimentsFolder,
                                                                        valueMethod=valueMethod,
                                                                        whichSetName=whichSetNameStat,
                                                                        datasetParameters=statDatasetParameters,
                                                                        classifierParameters=classifierParameters,
                                                                        modelStoreNameType=modelStoreNameType)
        (predicted_class_master, predicted_values_master, true_class_master, classLabelsMaster, totalStatOuputClasses) = tupleOutputTemp
        assert totalOuputClasses == totalStatOuputClasses, "The model dataset class number doesn't equal the stat dataset class number"

        if not useLabels:
            classLabelsMaster = None
        # using the thresholds from before but now the test set plot what this looks like
        ClassificationUtils.plotThresholds(predicted_class_master,
                                           predicted_values_master,
                                           true_class_master,
                                           eerThresholdsMaster,
                                           classLabels=classLabelsMaster,
                                           rogueClasses=rogueClassesMaster,
                                           setName=whichSetNameStat,
                                           valueMethodName=valueMethodName,
                                           statisticsStoreFolder=statisticStoreFolder,
                                           plotLabels=plotLabels,
                                           useVoroni=useVoroni,
                                           plotThresholdFigures=True,
                                           nTotalClassesArg=totalOuputClasses)

        if len(rogueClassesMaster) > 0:
            ClassificationUtils.rogueAnalysis(eerThresholdsMaster,
                                              predicted_class_master,
                                              predicted_values_master,
                                              true_class_master,
                                              classLabels=classLabelsMaster,
                                              rogueClasses=rogueClassesMaster,
                                              setName=whichSetNameStat,
                                              valueMethodName=valueMethodName,
                                              statisticsStoreFolder=statisticStoreFolder)
    if showFigures:
        plt.show()


def convertIndicesToOneHots(y_trainLabel, totalClasses):
    if y_trainLabel is not None:
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
    else:
        return None


def getListParameterAndPadLength(parameterName, padLength, classifierParameters, default=0.0):
    parameter = classifierParameters[parameterName] if parameterName in classifierParameters else default
    parameter = [parameter] if (type(parameter) != tuple or type(parameter) != list) else parameter
    assert len(parameter) == 1 or len(parameter) == len(padLength), "Bad number of {0}".format(parameterName)
    parameter = parameter * len(padLength) if len(parameter) == 1 else parameter
    return parameter


modelJsonExample = {"class_name": "Sequential", "keras_version": "1.1.0", "config":
    [{"class_name": "LSTM", "config":
        {"U_regularizer": {"l2": 0.0, "name": "WeightRegularizer", "l1": 9.999999747378752e-05}, "name": "LSTM Layer 1",
         "inner_activation": "hard_sigmoid",
         "go_backwards": False, "activation": "tanh", "trainable": True, "unroll": False, "consume_less": "gpu", "stateful": True,
         "init": "glorot_uniform", "inner_init": "orthogonal", "dropout_U": 0.5, "dropout_W": 0.5, "input_dtype": "float32", "return_sequences": True,
         "batch_input_shape": [10, 100, 1107], "W_regularizer": {"l2": 0.0, "name": "WeightRegularizer", "l1": 0.0}, "output_dim": 500,
         "forget_bias_init": "one", "b_regularizer": {"l2": 0.0, "name": "WeightRegularizer", "l1": 0.0}}},
     {"class_name": "LSTM", "config": {
         "U_regularizer": {"l2": 0.0, "name": "WeightRegularizer", "l1": 9.999999747378752e-05}, "name": "LSTM Layer 2",
         "inner_activation": "hard_sigmoid", "go_backwards": False, "activation": "tanh", "trainable": True, "unroll": False, "consume_less": "gpu",
         "stateful": True, "init": "glorot_uniform", "inner_init": "orthogonal", "dropout_U": 0.5, "dropout_W": 0.5, "input_dtype": "float32",
         "return_sequences": True, "batch_input_shape": [10, 100, 500], "W_regularizer": {"l2": 0.0, "name": "WeightRegularizer", "l1": 0.0},
         "output_dim": 500, "forget_bias_init": "one", "b_regularizer": {"l2": 0.0, "name": "WeightRegularizer", "l1": 0.0}}},
     {"class_name": "TimeDistributed", "config":
         {"layer": {"class_name": "Dense",
                    "config": {"W_constraint": None, "b_constraint": None, "name": "Dense Layer 3 tanh",
                               "activity_regularizer": None, "trainable": True, "init": "glorot_uniform",
                               "bias": True, "input_dim": None,
                               "b_regularizer": {"l2": 0.0, "name": "WeightRegularizer", "l1": 0.0},
                               "W_regularizer": {"l2": 0.0, "name": "WeightRegularizer",
                                                 "l1": 9.999999747378752e-05}, "activation": "tanh",
                               "output_dim": 500}}, "trainable": True, "name": "TD Dense Layer 3 tanh"}},
     {"class_name": "Dropout", "config":
         {"p": 0.5, "trainable": True, "name": "Dropout Layer 3"}},
     {"class_name": "TimeDistributed", "config":
         {"layer": {"class_name": "Dense",
                    "config": {"W_constraint": None, "b_constraint": None, "name": "Dense Layer 4 tanh", "activity_regularizer": None,
                               "trainable": True, "init": "glorot_uniform", "bias": True, "input_dim": None,
                               "b_regularizer": {"l2": 0.0, "name": "WeightRegularizer", "l1": 0.0},
                               "W_regularizer": {"l2": 0.0, "name": "WeightRegularizer", "l1": 9.999999747378752e-05}, "activation": "tanh",
                               "output_dim": 500}}, "trainable": True, "name": "TD Dense Layer 4 tanh"}},
     {"class_name": "Dropout", "config":
         {"p": 0.5, "trainable": True, "name": "Dropout Layer 4"}},
     {"class_name": "TimeDistributed", "config": {
         "layer": {"class_name": "Dense",
                   "config": {"W_constraint": None, "b_constraint": None, "name": "Dense Layer Output", "activity_regularizer": None,
                              "trainable": True, "init": "glorot_uniform", "bias": True, "input_dim": None, "b_regularizer": None,
                              "W_regularizer": None, "activation": "linear", "output_dim": 100}}, "trainable": True,
         "name": "TD Dense Layer Output"}},
     {"class_name": "Activation", "config": {"activation": "softmax", "trainable": True, "name": "softmax Layer Output"}}]}


def changeBatchInputShapeOfModel(model_json_string, newBatchSize):
    modelJson = json.loads(model_json_string)
    # if I am using the teacher forcing it wraps the config in a layers key
    layerArray = modelJson['config']['layers'] if 'layers' in modelJson else modelJson['config']
    for layer in layerArray:
        if 'batch_input_shape' in layer['config']:
            layer['config']['batch_input_shape'][0] = newBatchSize
    return json.dumps(modelJson)


def kerasClassifier_parameterized(featureParameters,
                                  datasetParameters,
                                  classifierParameters,
                                  forceRebuildModel=False,
                                  showModelAsFigure=False,
                                  trainValidTestSetNames=('train', 'valid', 'test'),
                                  experimentsFolder=None):
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

    datasetFile = CreateUtils.getDatasetFile(featureSetName=featureParameters['featureSetName'], datasetName=datasetParameters['datasetName'])

    if experimentsFolder is None:
        experimentsFolder = CreateUtils.getExperimentFolder(featureParameters['featureSetName'],
                                                            datasetParameters['datasetName'],
                                                            classifierParameters['classifierType'],
                                                            classifierParameters['classifierSetName'])
    modelWeightsRegex = re.compile(""".*modelWeights.h5""")
    experimentsFolderFiles = os.listdir(experimentsFolder)
    foundAnyWeights = any([re.match(modelWeightsRegex, filePath) for filePath in experimentsFolderFiles])

    modelStoreFilePathFullTemp = os.path.join(experimentsFolder, 'model.json')
    if not (os.path.exists(modelStoreFilePathFullTemp) and foundAnyWeights) or forceRebuildModel:
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
        useTeacherForcing = classifierParameters['useTeacherForcing'] if 'useTeacherForcing' in classifierParameters else False
        teacherForcingDropout = classifierParameters['teacherForcingDropout'] if 'teacherForcingDropout' in classifierParameters else 0.

        if classifierParameters['classifierGoal'] == 'regression':
            (datasets,
             inputs,
             outputs,
             max_batch_size) = RegressionUtils.load_data(datasetFile,
                                                         rogueClasses=(),
                                                         makeSharedData=False,
                                                         makeSequenceForX=makeSequencesForX,
                                                         makeSequenceForY=useTimeDistributedOutput,
                                                         timesteps=timesteps,
                                                         setNames=trainValidTestSetNames)
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
                                                             timesteps=timesteps,
                                                             setNames=trainValidTestSetNames)
            (X_train, X_valid, X_test) = (datasets[0][0], datasets[1][0], datasets[2][0])
            lossType = classifierParameters['lossType']
            if lossType in ['falsePositiveRate']:
                (y_train, y_valid, y_test) = (datasets[0][1], datasets[1][1], datasets[2][1])
            else:
                y_train = convertIndicesToOneHots(datasets[0][1], outputs)
                y_valid = convertIndicesToOneHots(datasets[1][1], outputs)
                y_test = convertIndicesToOneHots(datasets[2][1], outputs)
        assert X_train is not None and y_train is not None, "You didn't select any training data"

        final_output_dim = outputs

        imageShape = featureParameters['imageShape']
        useMetadata = datasetParameters['useMetadata'] if 'useMetadata' in datasetParameters else False
        metadataShape = datasetParameters['metadataShape'] if 'metadataShape' in datasetParameters else (0,)
        metadata_dim = np.product(metadataShape) if useMetadata else 0
        data_dim = np.product(imageShape) + metadata_dim
        if useTeacherForcing:
            data_dim += final_output_dim

        lstm_layers_sizes = classifierParameters['lstm_layers_sizes']
        hidden_layers_sizes = classifierParameters['hidden_layers_sizes']
        finalActivationType = classifierParameters['finalActivationType']
        maxout_layers_sizes = classifierParameters['maxout_layers_sizes'] if 'maxout_layers_sizes' in classifierParameters else []

        n_epochs = classifierParameters['n_epochs']
        batch_size = min(classifierParameters['batch_size'], max_batch_size)
        stateful = classifierParameters['stateful'] if 'stateful' in classifierParameters else False
        consume_less = classifierParameters['consume_less'] if 'consume_less' in classifierParameters else 'cpu'
        trainLSTM = classifierParameters['trainLSTM'] if 'trainLSTM' in classifierParameters else True
        trainMLP = classifierParameters['trainMLP'] if 'trainMLP' in classifierParameters else True
        trainMaxout = classifierParameters['trainMaxout'] if 'trainMaxout' in classifierParameters else True

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
        dropout_Maxout = getListParameterAndPadLength('dropout_Maxout', maxout_layers_sizes, classifierParameters)

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

        W_regularizer_l1_maxout = getListParameterAndPadLength("W_regularizer_l1_maxout", maxout_layers_sizes,
                                                               classifierParameters)
        b_regularizer_l1_maxout = getListParameterAndPadLength("b_regularizer_l1_maxout", maxout_layers_sizes,
                                                               classifierParameters)
        W_regularizer_l2_maxout = getListParameterAndPadLength("W_regularizer_l2_maxout", maxout_layers_sizes,
                                                               classifierParameters)
        b_regularizer_l2_maxout = getListParameterAndPadLength("b_regularizer_l2_maxout", maxout_layers_sizes,
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
            if X_train is not None:
                X_train = X_train[:, :, None, :]
            if X_valid is not None:
                X_valid = X_valid[:, :, None, :]
            if X_test is not None:
                X_test = X_test[:, :, None, :]

        # LSTM
        for layerIndex in range(1, 1 + len(lstm_layers_sizes)):
            lstmIndex = layerIndex - 1

            if (dropout_LSTM[lstmIndex]) > 0:
                model.add(Dropout(dropout_LSTM[lstmIndex], name="Dropout Layer {0}".format(layerIndex)))

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

        # MLP
        for layerIndex in range(1 + len(lstm_layers_sizes), 1 + len(lstm_layers_sizes) + len(hidden_layers_sizes)):
            hiddenIndex = layerIndex - (1 + len(lstm_layers_sizes))

            if (dropout_Hidden[hiddenIndex]) > 0:
                model.add(Dropout(dropout_Hidden[hiddenIndex], name="Dropout Layer {0}".format(layerIndex)))

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

        # Maxout Dense
        for layerIndex in range(1 + len(lstm_layers_sizes) + len(hidden_layers_sizes), 1 + len(lstm_layers_sizes),
                                len(hidden_layers_sizes) + len(maxout_layers_sizes)):
            maxoutIndex = layerIndex - (1 + len(lstm_layers_sizes) + len(hidden_layers_sizes))
            if (dropout_Maxout[maxoutIndex]) > 0:
                model.add(Dropout(dropout_Maxout[maxoutIndex], name="Dropout Layer {0}".format(layerIndex)))

            W_regularizer_maxout = WeightRegularizer(l1=W_regularizer_l1_maxout[maxoutIndex],
                                                     l2=W_regularizer_l2_maxout[maxoutIndex])
            b_regularizer_maxout = WeightRegularizer(l1=b_regularizer_l1_maxout[maxoutIndex],
                                                     l2=b_regularizer_l2_maxout[maxoutIndex])
            thisMaxoutLayer = MaxoutDense(maxout_layers_sizes[maxoutIndex],
                                          W_regularizer=W_regularizer_maxout,
                                          b_regularizer=b_regularizer_maxout,
                                          trainable=trainMaxout)
            if useTimeDistributedOutput:
                thisMaxoutLayer = TimeDistributed(thisMaxoutLayer, name="TD {0}".format(thisMaxoutLayer.name))
            model.add(thisMaxoutLayer)

        # Final dense layer to match expected output
        if useAppendMLPLayers:
            finalOutputDataDimension = appendExpectedInput
        else:
            finalOutputDataDimension = y_train.shape[1] if useTimeDistributedOutput is False else y_train.shape[2]
        lastDenseLayer = Dense(finalOutputDataDimension,
                               trainable=trainMLP,
                               activation=finalActivationType,
                               name="Dense Layer Output")
        if useTimeDistributedOutput:
            lastDenseLayer = TimeDistributed(lastDenseLayer, name="TD {0}".format(lastDenseLayer.name))
        model.add(lastDenseLayer)

        # load previous weights to the LSTM and Dense Layers
        loadPreviousModelWeightsForTraining = classifierParameters['loadPreviousModelWeightsForTraining'] \
            if 'loadPreviousModelWeightsForTraining' in classifierParameters else False
        if loadPreviousModelWeightsForTraining:
            loadWeightsFilePath = CreateUtils.getAbsolutePath(
                classifierParameters['loadWeightsFilePath']) if 'loadWeightsFilePath' in classifierParameters else ''
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

        # Append one more Dense layer with usually predefined weights
        if useAppendMLPLayers:
            append_layers_sizes = classifierParameters['append_layers_sizes']
            append_activations = getListParameterAndPadLength('append_activations', append_layers_sizes, classifierParameters)
            dropout_Append = getListParameterAndPadLength('dropout_append', append_layers_sizes, classifierParameters)
            appendWeightsFile = getListParameterAndPadLength('appendWeightsFile', append_layers_sizes, classifierParameters, default=None)
            trainAppend = classifierParameters['trainAppend']

            totalLayersSoFar = 1 + len(lstm_layers_sizes) + len(hidden_layers_sizes) + len(maxout_layers_sizes)
            for layerIndex in range(totalLayersSoFar, totalLayersSoFar + len(append_layers_sizes)):
                appendIndex = layerIndex - totalLayersSoFar

                if appendWeightsFile[appendIndex] is not None:
                    appendWeightsFileAbsolute = CreateUtils.getAbsolutePath(appendWeightsFile[appendIndex])
                    thisWeights = np.genfromtxt(appendWeightsFileAbsolute, delimiter=',', skip_header=1)
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

        if useTeacherForcing:
            # if showModelAsFigure:
            #     showModel(model)
            #     modelImage = os.path.join(experimentsFolder, 'modelImageOriginal.png')
            #     saveModelToFile(model, modelImage)
            model = TeacherForcingModel(original_model=model,
                                        last_layer_dropout_p=teacherForcingDropout)
        #########################
        # Model Building Complete
        #########################

        # Visualizations
        if showModelAsFigure:
            showModel(model)
        modelImage = os.path.join(experimentsFolder, 'modelImage.png')
        saveModelToFile(model, modelImage)

        timer.tic("Compiling Model")
        compileAModel(model,
                      classifierParameters=classifierParameters,
                      datasetParameters=datasetParameters)
        timer.toc()

        with open(modelStoreFilePathFullTemp, 'w') as modelStoreFile:
            json_string = model.to_json()
            modelStoreFile.write(json_string)

        # Callbacks
        myCallbacks = []
        # Best Val_loss model
        if X_valid is not None and y_valid is not None:
            modelStoreWeightsBestFilePathFullTemp = os.path.join(experimentsFolder, 'best_modelWeights.h5')
            checkpointerBest = keras.callbacks.ModelCheckpoint(filepath=modelStoreWeightsBestFilePathFullTemp,
                                                               verbose=1,
                                                               save_best_only=True)
            myCallbacks.append(checkpointerBest)
            model.save_weights(modelStoreWeightsBestFilePathFullTemp, True)
        # Best Loss model
        modelStoreWeightsBestLossFilePathFullTemp = os.path.join(experimentsFolder, 'bestLoss_modelWeights.h5')
        checkpointerBestLoss = keras.callbacks.ModelCheckpoint(filepath=modelStoreWeightsBestLossFilePathFullTemp,
                                                               verbose=0,
                                                               save_best_only=True,
                                                               monitor='loss')
        myCallbacks.append(checkpointerBestLoss)
        model.save_weights(modelStoreWeightsBestLossFilePathFullTemp, True)
        # Last model
        modelStoreWeightsLastFilePathFullTemp = os.path.join(experimentsFolder, 'last_modelWeights.h5')
        checkpointerLast = keras.callbacks.ModelCheckpoint(filepath=modelStoreWeightsLastFilePathFullTemp,
                                                           verbose=0,
                                                           save_best_only=False)
        myCallbacks.append(checkpointerLast)
        model.save_weights(modelStoreWeightsLastFilePathFullTemp, True)
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
        didFinishTraining = True
        if not onlyBuildModel:
            print("Begin training")
            try:
                validation_data = None
                if X_valid is not None and y_valid is not None:
                    validation_data = (X_valid, y_valid)
                model.fit(x=X_train,
                          y=y_train,
                          batch_size=batch_size,
                          nb_epoch=n_epochs,
                          validation_data=validation_data,
                          callbacks=myCallbacks,
                          verbose=2)
            except KeyboardInterrupt:
                didFinishTraining = False
                print ("\nUser stopped training")

        if X_test is not None and y_test is not None:
            model.reset_states()
            score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
            print("The final test score is {0}".format(score))
        return didFinishTraining
