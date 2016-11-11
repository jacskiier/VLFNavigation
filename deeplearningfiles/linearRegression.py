import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

import logistic_sgd

import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib import collections as mc
from sklearn.preprocessing import normalize
import yaml
import re
import pandas as pd
from scipy.stats import gaussian_kde

from readwav import shuffle_in_unison_inplace


class LinearRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, inputs, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type inputs: theano.tensor.TensorType
        :param inputs: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        if W is None:
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
            self.W = theano.shared(
                value=numpy.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        else:
            self.W = W

        if b is None:
            # initialize the biases b as a vector of n_out 0s
            self.b = theano.shared(
                value=numpy.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            self.b = b

        self.y_pred = T.dot(inputs, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = inputs

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct regression values
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        return (self.y_pred - y).norm(2, axis=1)

    def negative_log_likelihood(self, y):
        return T.mean((self.y_pred - y).norm(2, axis=1))


def load_data(datasetFileName, rogueClasses=(), makeSharedData=True, makeSequenceForX=False, makeSequenceForY=False,
              timesteps=0):
    """ Loads the dataset

    :type datasetFileName: string
    :param datasetFileName: the path to the dataset pickle file that returns (train_set, valid_set, test_set) who are tuples with 0 being features and 1 being class values

    :type rogueClasses: list of class outputs (int)
    :param rogueClasses: List of classes to exclude from training to do rogue agalysis

    :type makeSharedData: bool
    :param makeSharedData: if true will return a theano shared variable representation instead of a numpy array representation

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

    # Load the dataset
    if re.match('''.*\.pkl\.gz$''', datasetFileName):
        f = gzip.open(datasetFileName, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
    elif re.match('''.*\.hf$''', datasetFileName):
        with pd.HDFStore(datasetFileName, 'r') as featureStore:
            train_set_x = featureStore['train_set_x'].as_matrix()
            valid_set_x = featureStore['valid_set_x'].as_matrix()
            test_set_x = featureStore['test_set_x'].as_matrix()
            train_set_y = featureStore['train_set_y'].as_matrix()
            valid_set_y = featureStore['valid_set_y'].as_matrix()
            test_set_y = featureStore['test_set_y'].as_matrix()
            train_set = [train_set_x, train_set_y]
            valid_set = [valid_set_x, valid_set_y]
            test_set = [test_set_x, test_set_y]
    else:
        raise ValueError('Only .pkl.gz or .hf or .csv file types are supported')

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
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
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
        nonRogueMask = numpy.logical_not(numpy.in1d(test_set[1], numpy.array(rogueClasses)))
        test_set = (test_set[0][nonRogueMask], test_set[1][nonRogueMask])

        nonRogueMask = numpy.logical_not(numpy.in1d(valid_set[1], numpy.array(rogueClasses)))
        valid_set = (valid_set[0][nonRogueMask], valid_set[1][nonRogueMask])

        nonRogueMask = numpy.logical_not(numpy.in1d(train_set[1], numpy.array(rogueClasses)))
        train_set = (train_set[0][nonRogueMask], train_set[1][nonRogueMask])

        for rogueClass in list(numpy.flipud(rogueClasses)):
            test_set[1][test_set[1] > rogueClass] -= 1
            valid_set[1][valid_set[1] > rogueClass] -= 1
            train_set[1][train_set[1] > rogueClass] -= 1

    for setter in [train_set, valid_set, test_set]:
        if makeSequenceForX:
            setter[0] = numpy.reshape(setter[0], (setter[0].shape[0], timesteps, setter[0].shape[1] / timesteps))
        if makeSequenceForY:
            setter[1] = numpy.reshape(setter[1], (setter[1].shape[0], timesteps, setter[1].shape[1] / timesteps))

    if makeSharedData:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)
    else:
        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    inputFeatures = test_set[0].shape[1]

    largestSampleSetPossible = min((test_set[1].shape[0], valid_set[1].shape[0], train_set[1].shape[0]))
    print ("loading complete")
    return rval, inputFeatures, train_set[1].shape[1], largestSampleSetPossible


def sgd_optimization_mnist(datasetFileName='mnist.pkl.gz',
                           experimentStoreFolder='',
                           learning_rate=0.13,
                           n_epochs=1000,
                           batch_size=600,
                           patience=5000,
                           patience_increase=2,
                           improvement_threshold=0.995,
                           rogueClasses=()):
    """
    Stochastic gradient descent optimization of a log-linearmodel

    :type datasetFileName: string
    :param datasetFileName: the path to the dataset pickle file that returns (train_set, valid_set, test_set) who are tuples with 0 being features and 1 being class values

    :type experimentStoreFolder: basestring
    :param experimentStoreFolder: the folder to store the model after it is trained

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type batch_size: int
    :param batch_size: size of batches to train on

    :type patience: int
    :param patience:  look as this many examples regardless

    :type patience_increase: int
    :param patience_increase: wait this much longer when a new best is found

    :type improvement_threshold: float
    :param improvement_threshold: a relative improvement of this much is considered significant (0-1)

    :type rogueClasses: tuple of class outputs (int)
    :param rogueClasses: Tuple of classes to exclude from training to do rogue agalysis

    """

    datasets, inputs, outputs, max_batch_size = load_data(datasetFileName, rogueClasses=rogueClasses)

    batch_size = min(max_batch_size, batch_size)
    if batch_size == max_batch_size:
        print("Doing only one batch due to size constraint")

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.matrix('y')  # labels, presented as matrix

    # construct the logistic regression class
    classifier = LinearRegression(inputs=x, n_in=inputs, n_out=outputs)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'

    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # print ('minibatch_avg_cost {0}'.format(minibatch_avg_cost))
            # iteration number
            currentIteration = (epoch - 1) * n_train_batches + minibatch_index

            if (currentIteration + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch {epoch}/{n_epochs} {epochpercent:3.2f}%, minibatch {minibatch}/{batches}, iter/patience {iteration}/{patience} {doneness:3.2f}%, training error {trainloss:4.5f}, validation error {valloss:4.5f}'.format
                        (
                        epoch=epoch,
                        n_epochs=n_epochs,
                        epochpercent=100 * epoch / float(n_epochs),
                        minibatch=minibatch_index + 1,
                        batches=n_train_batches,
                        trainloss=numpy.max(minibatch_avg_cost),
                        valloss=this_validation_loss,
                        patience=patience,
                        iteration=currentIteration,
                        doneness=100 * currentIteration / float(patience)
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, currentIteration * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t    New best model of test error {test_score:4.5f}'.format(
                            test_score=test_score))

                    # save the best model
                    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.pkl')
                    with open(modelStoreFilePathFullTemp, 'wb') as f:
                        cPickle.dump(classifier, f)
                    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_modelNUMPY.pkl')
                    with open(modelStoreFilePathFullTemp, 'wb') as f:
                        W = numpy.array(classifier.W.eval())
                        b = numpy.array(classifier.b.eval())
                        cPickle.dump((W, b), f)

            if patience <= currentIteration:
                done_looping = True  # this will stop while loop
                break  # out of for loop

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % (end_time - start_time))
    return classifier


def sgd_optimization_parameterized(featureParameters, datasetParameters, classifierParameters, forceRebuildModel=False):
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
    """

    assert classifierParameters[
               'classifierType'] == 'LinearRegression', 'this config wasnt made for a Logistic Regression'

    rawDataFolder = datasetParameters['rawDataFolder']

    if os.path.exists(
            os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'] + '.hf')):
        datasetFile = os.path.join(datasetParameters['processedDataFolder'],
                                   featureParameters['featureSetName'] + '.hf')
    else:
        datasetFile = os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'],
                                   datasetParameters['datasetName'] + '.pkl.gz')

    experimentsFolder = os.path.join(rawDataFolder, "Data Experiments", featureParameters['featureSetName'],
                                     datasetParameters['datasetName'],
                                     classifierParameters['classifierType'], classifierParameters['classifierSetName'])

    bestModelFilePath = os.path.join(experimentsFolder, 'best_model.pkl')
    if not os.path.exists(bestModelFilePath) or forceRebuildModel:
        sgd_optimization_mnist(datasetFileName=datasetFile,
                               experimentStoreFolder=experimentsFolder,
                               learning_rate=classifierParameters['learning_rate'],
                               n_epochs=classifierParameters['n_epochs'],
                               batch_size=classifierParameters['batch_size'],
                               patience=classifierParameters['patience'],
                               patience_increase=classifierParameters['patience_increase'],
                               improvement_threshold=classifierParameters['improvement_threshold'],
                               rogueClasses=classifierParameters['rogueClasses'])


def getPredicted_Values_TrueValues_Labels(datasetFileName='mnist.pkl.gz', experimentStoreFolder='', whichSetArg=2):
    # load the saved model
    processedDataFolder = os.path.dirname(datasetFileName)
    baseDatasetFileName = os.path.basename(datasetFileName)
    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.pkl')
    classifier = cPickle.load(open(modelStoreFilePathFullTemp))

    # compile predictor function
    predict_model_y_values = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred,
    )

    # We can test it on some examples from test test
    datasets, inputs, outputs, max_batch_size = load_data(datasetFileName)
    (set_x, set_y) = datasets[whichSetArg]

    # partialYoverPartialX = T.grad(T.sum(classifier.y_pred.norm(2, axis=1)), classifier.input)
    partialYoverPartialX = T.grad(T.sum(classifier.y_pred), classifier.input)
    getAllPartials = theano.function(
        inputs=[],
        outputs=partialYoverPartialX,
        givens={
            classifier.input: set_x
        }
    )
    gradArray = getAllPartials()
    gradStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'gradArray.hf')
    with pd.HDFStore(gradStoreFilePathFullTemp, 'a') as datasetStore:
        datasetStore['gradArray{0}'.format(whichSetArg)] = pd.DataFrame(gradArray)
    # numpy.savetxt(os.path.join(experimentStoreFolder, 'gradArray.csv'), gradArray, delimiter=',')


    set_x = set_x.get_value()
    predicted_y_values = predict_model_y_values(set_x)

    # We can test it on some examples from test test
    datasets, inputs, outputs, max_batch_size = load_data(datasetFileName, makeSharedData=False)
    (set_x, set_y) = datasets[whichSetArg]

    classLabels = logistic_sgd.getLabelsForDataset(processedDataFolder, datasetFileName)

    return predicted_y_values, set_y, classLabels


def getStatistics(predictedValues, trueValues, classLabels=None, rogueClasses=(), staticFigures=7, setName="",
                  statisticsStoreFolder='', datasetParameters=None, weightsName=""):
    if datasetParameters is not None and datasetParameters['yValueType'] == 'gpsPolar':
        rtemp = predictedValues[:, 0]
        thetatemp = predictedValues[:, 1] * numpy.pi / 180.0
        predictedValues[:, 1] = rtemp * numpy.cos(thetatemp)
        predictedValues[:, 0] = rtemp * numpy.sin(thetatemp)

        rtemp = trueValues[:, 0]
        thetatemp = trueValues[:, 1] * numpy.pi / 180.0
        trueValues[:, 1] = rtemp * numpy.cos(thetatemp)
        trueValues[:, 0] = rtemp * numpy.sin(thetatemp)

    print("Getting statistics for regression model")
    if predictedValues.ndim < 2:
        predictedValues = numpy.expand_dims(predictedValues, axis=1)
    if trueValues.ndim < 2:
        trueValues = numpy.expand_dims(trueValues, axis=1)

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

    rss = numpy.sum((predictedValues - trueValues) ** 2)

    rse = numpy.sqrt(rss / (trueValues.shape[0] - 2))

    tss = numpy.sum((trueValues - numpy.mean(trueValues)) ** 2)

    r2 = (tss - rss) / tss

    mse = numpy.mean((predictedValues - trueValues) ** 2)

    meanDistance = numpy.mean(numpy.linalg.norm(predictedValues - trueValues, axis=1, ord=2))

    rmse = numpy.sqrt(mse)

    RMatrix = numpy.cov((predictedValues - trueValues).transpose())

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


def rogueAnalysis(thresholds, predictedValues, trueValues, classLabels=None, rogueClasses=(), staticFigures=7,
                  setName="",
                  statisticsStoreFolder=""):
    raise NotImplementedError, "Rogue Analysis is not implemented for Linear Regression type problems"


def makeStatisticsForModel(experimentsFolder, statisticsStoreFolder, featureParameters, datasetParameters,
                           classifierParameters, valueMethod=0, useLabels=True, whichSet=1, showFigures=True):
    """
    Make staticstics for a model using the features, datset, and classifier given whose model is already made

    :type experimentsFolder: str
    :param experimentsFolder: Location of the pre made model and where the statistics will be saved

    :type featureParameters: dict
    :param featureParameters: parameters for the features

    :type datasetParameters: dict
    :param datasetParameters: parameters for the dataset

    :type classifierParameters: dict
    :param classifierParameters: parameters for the classifier

    :type valueMethod: int
    :param valueMethod: Values type to be for classification and thresholding, 0 = use highest probability, 1 = use ration of higest prob to second, 2 = use difference of highest prob to second

    :type useLabels: bool
    :param useLabels: If labels should be used in charts True or just the class number False

    :type whichSet: int
    :param whichSet: Which of the sets to do the statistics on training=0 validation=1 testing=2

    :type showFigures: bool
    :param showFigures: If you want to see the figures now instead of viewing them on disk later (still saves them no matter what)
    """

    valueMethods = ['Probability', 'Probability Ratio', 'Probability Difference']
    setNames = ['Training', 'Validation', 'Testing']

    if os.path.exists(
            os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'] + '.hf')):
        datasetFile = os.path.join(datasetParameters['processedDataFolder'],
                                   featureParameters['featureSetName'] + '.hf')
    else:
        datasetFile = os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'],
                                   datasetParameters['datasetName'] + '.pkl.gz')

    rogueClassesMaster = classifierParameters['rogueClasses']
    (predicted_values_master, true_values_master, classLabelsMaster) = getPredicted_Values_TrueValues_Labels(
        datasetFileName=datasetFile,
        experimentStoreFolder=experimentsFolder,
        whichSetArg=whichSet)
    if not useLabels:
        classLabelsMaster = None
    thresholdsMaster = getStatistics(predicted_values_master, true_values_master, classLabels=classLabelsMaster,
                                     rogueClasses=rogueClassesMaster,
                                     setName=setNames[whichSet], statisticsStoreFolder=statisticsStoreFolder,
                                     datasetParameters=datasetParameters)
    if len(rogueClassesMaster) > 0:
        rogueAnalysis(thresholdsMaster, predicted_values_master, true_values_master, classLabels=classLabelsMaster,
                      rogueClasses=rogueClassesMaster,
                      setName=setNames[whichSet], statisticsStoreFolder=statisticsStoreFolder)
    if showFigures:
        plt.show()


if __name__ == '__main__':
    rawDataFolderMain = r"E:\\Users\\Joey\Documents\\Virtual Box Shared Folder\\"

    featureSetNameMain = 'DefaultPatchFeatures'
    datasetNameMain = 'staticLocations'
    classifierType = 'LogisticRegression'
    classifierSetNameMain = 'AllClassesDefault'

    featureDataFolderMain = os.path.join(rawDataFolderMain, "Processed Data Features", featureSetNameMain)
    featureConfigFileName = os.path.join(featureDataFolderMain, "feature parameters.yaml")
    with open(featureConfigFileName, 'r') as myConfigFile:
        featureParametersDefault = yaml.load(myConfigFile)

    processedDataFolderMain = os.path.join(rawDataFolderMain, "Processed Data Datasets", datasetNameMain)
    datasetConfigFileName = os.path.join(processedDataFolderMain, "datset parameters.yaml")
    with open(datasetConfigFileName, 'r') as myConfigFile:
        datasetParametersDefault = yaml.load(myConfigFile)

    modelStoreFolderMaster = os.path.join(rawDataFolderMain, "Processed Data Models", classifierSetNameMain)
    modelConfigFileName = os.path.join(modelStoreFolderMaster, "model set parameters.yaml")
    with open(modelConfigFileName, 'r') as myConfigFile:
        modelSetParametersDefault = yaml.load(myConfigFile)

    sgd_optimization_parameterized(featureParametersDefault, datasetParametersDefault, modelSetParametersDefault,
                                   forceRebuildModel=True)
    makeStatisticsForModel(featureParametersDefault, datasetParametersDefault, modelSetParametersDefault)
