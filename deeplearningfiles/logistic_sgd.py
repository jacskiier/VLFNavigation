"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T

import convolutional_mlp
import matplotlib.pylab as plt
import matplotlib.cm as cm
from sklearn.preprocessing import normalize
import yaml
import re
import pandas as pd
from scipy.stats import gaussian_kde
import StringIO
import matplotlib
import matplotlib.patches
import matplotlib.collections

from tqdm import tqdm

class LogisticRegression(object):
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
                value=np.zeros(
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
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            self.b = b

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(inputs, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = inputs

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(datasetFileName, rogueClasses=(), makeSharedData=True, makeSequenceForX=False, makeSequenceForY=False,
              timesteps=0):
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
        raise ValueError('Only .pkl.gz or .hf file types are supported')

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
        nonRogueMask = np.logical_not(np.in1d(test_set[1], np.array(rogueClasses)))
        test_set = (test_set[0][nonRogueMask], test_set[1][nonRogueMask])

        nonRogueMask = np.logical_not(np.in1d(valid_set[1], np.array(rogueClasses)))
        valid_set = (valid_set[0][nonRogueMask], valid_set[1][nonRogueMask])

        nonRogueMask = np.logical_not(np.in1d(train_set[1], np.array(rogueClasses)))
        train_set = (train_set[0][nonRogueMask], train_set[1][nonRogueMask])

        for rogueClass in list(np.flipud(rogueClasses)):
            test_set[1][test_set[1] > rogueClass] -= 1
            valid_set[1][valid_set[1] > rogueClass] -= 1
            train_set[1][train_set[1] > rogueClass] -= 1

    for setter in [train_set, valid_set, test_set]:
        if makeSequenceForX:
            setter[0] = np.reshape(setter[0], (setter[0].shape[0], timesteps, setter[0].shape[1] / timesteps))
        if makeSequenceForY:
            setter[1] = np.reshape(setter[1], (setter[1].shape[0], timesteps, setter[1].shape[1] / timesteps))

    if makeSharedData:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)
    else:
        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    inputFeatures = test_set[0].shape[1]

    allOutputClasses = np.concatenate((train_set[1], valid_set[1], test_set[1]), 0)
    outputClassesByUnique = np.unique(allOutputClasses).shape[0]
    outputClassesByLargest = int(np.max(allOutputClasses)) + 1
    assert outputClassesByUnique == outputClassesByLargest, "Unique output classes != largest output class"
    outputClassesTotal = max(outputClassesByUnique, outputClassesByLargest)
    # assert outputClassesTotal == (int(np.max(test_set[1] )) + 1), "test  set is missing output classes"
    # assert outputClassesTotal == (int(np.max(valid_set[1])) + 1), "valid set is missing output classes"
    # assert outputClassesTotal == (int(np.max(train_set[1])) + 1), "train set is missing output classes"

    largestSampleSetPossible = min((test_set[1].shape[0], valid_set[1].shape[0], train_set[1].shape[0]))
    print ("loading complete")
    return rval, inputFeatures, outputClassesTotal, largestSampleSetPossible


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
    :param datasetFileName: the path to the dataset pickle file that returns (train_set, valid_set, test_set)
    who are tuples with 0 being features and 1 being class values

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
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    classifier = LogisticRegression(inputs=x, n_in=inputs, n_out=outputs)

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

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # total number of times we have calculated a minibatch cost
            currentIteration = (epoch - 1) * n_train_batches + minibatch_index

            if (currentIteration + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch {0}, minibatch {1}/{2}, validation error {3:4.2f}%'.format
                        (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
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
                    test_score = np.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.pkl')
                    with open(modelStoreFilePathFullTemp, 'wb') as f:
                        cPickle.dump(classifier, f)
                    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_modelNUMPY.pkl')
                    with open(modelStoreFilePathFullTemp, 'wb') as f:
                        W = np.array(classifier.W.eval())
                        b = np.array(classifier.b.eval())
                        cPickle.dump((W, b), f)

            if patience <= currentIteration:
                done_looping = True
                break

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


def getPredictedClasses_Values_TrueClasses_Labels(datasetFileName='mnist.pkl.gz',
                                                  experimentStoreFolder='',
                                                  valueMethod=0, whichSetArg=2):
    # load the saved model
    processedDataFolder = os.path.dirname(datasetFileName)
    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.pkl')
    classifier = cPickle.load(open(modelStoreFilePathFullTemp))

    # compile predictor function
    predict_model_probabilities = theano.function(
        inputs=[classifier.input],
        outputs=classifier.p_y_given_x,
    )

    # We can test it on some examples from test test
    datasets, inputs, outputs, max_batch_size = load_data(datasetFileName)
    (set_x, set_y) = datasets[whichSetArg]
    set_x = set_x.get_value()

    predicted_probabilities = predict_model_probabilities(set_x)

    # predicted_class ranges from 0 to nClasses
    predicted_class = np.argmax(predicted_probabilities, axis=1)

    # We can test it on some examples from test test
    datasets, inputs, outputs, max_batch_size = load_data(datasetFileName, makeSharedData=False)
    (set_x, set_y) = datasets[whichSetArg]

    predictedValues = None
    if valueMethod == 0:  # probability
        predictedValues = np.max(predicted_probabilities, axis=1)
    elif valueMethod == 1:  # probability relative to next highest
        predicted_probabilitiesSorted = np.fliplr(np.sort(predicted_probabilities, axis=1))
        predictedValues = predicted_probabilitiesSorted[:, 0] / predicted_probabilitiesSorted[:, 1]
    elif valueMethod == 2:  # probability difference to next highest
        predicted_probabilitiesSorted = np.fliplr(np.sort(predicted_probabilities, axis=1))
        predictedValues = predicted_probabilitiesSorted[:, 0] - predicted_probabilitiesSorted[:, 1]

    classLabels = getLabelsForDataset(processedDataFolder, datasetFileName)

    return predicted_class, predictedValues, set_y, classLabels


def getEERThreshold(predicted_class, predictedValues, true_class, nTotalClassesArg=None, rogueClasses=()):
    nRogue = len(rogueClasses)
    if nTotalClassesArg is None:
        # nTotalClasses = np.unique(true_class).shape[0] # this way doesn't work if some of your
        # output classes never happened
        nTotalClasses = int(np.max(true_class)) + 1  # and this is gonna fail if the largest
        # output clases never happened
        # assert(np.unique(true_class).shape[0] == int(np.max(true_class)) + 1)
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
                   plotLabels=False):
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
                plt.plot([inputThresholds[classIndex], inputThresholds[classIndex]], [0, ax.get_ylim()[1]])
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
            plt.text(x=xMid - gridSize[1] / 2.0,
                     y=yMid,
                     s="{countsThisClass} \n{tpRateThisClass:2.2%}".format(tpRateThisClass=acc,
                                                                        countsThisClass=totalCountsThisClass),
                     size=12,
                     zorder=2,
                     color='k')

        colorMap = cm.get_cmap('coolwarm')
        p = matplotlib.collections.PatchCollection(patches=rects, cmap=colorMap)
        p.set_array(accuracyPerClass)
        ax.add_collection(p)
        plt.xlim([np.min(outputLabelsAsArray[:, 1]) - gridSize[1], np.max(outputLabelsAsArray[:, 1]) + gridSize[1]])
        plt.ylim([np.min(outputLabelsAsArray[:, 0]) - gridSize[0], np.max(outputLabelsAsArray[:, 0]) + gridSize[0]])
        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        plt.title("Counts and True Positive percentage for each grid location")
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
            },
            'eerThresholds': [float(value) for value in inputThresholds]
        }
    }

    print (thresholdString)
    with open(os.path.join(statisticsStoreFolder, 'thresholdstats.log'), 'w') as logfile:
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
               'classifierType'] == 'LogisticRegression', 'this config wasnt made for a Logistic Regression'

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


def makeStatisticsForModel(experimentsFolder, statisticsStoreFolder, featureParameters, datasetParameters,
                           classifierParameters, valueMethod=0, useLabels=True, whichSet=1, showFigures=True):
    """
    Make staticstics for a model using the features, datset, and classifier given whose model is already made

    :type experimentsFolder: str
    :param experimentsFolder: Location of the pre made model

    :type statisticsStoreFolder: str
    :param statisticsStoreFolder: Location to store the statistics that may be separate from the experiment

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
    thresholdSet = 1
    # Grab the validation set in order to calculate EER thresholds
    tupleOutputTemp = getPredictedClasses_Values_TrueClasses_Labels(datasetFileName=datasetFile,
                                                                    experimentStoreFolder=experimentsFolder,
                                                                    valueMethod=valueMethod,
                                                                    whichSetArg=thresholdSet)
    (predicted_class_master, predicted_values_master, true_class_master, classLabelsMaster) = tupleOutputTemp
    # make EER thresholds with the validation set
    eerThresholdsMaster = getEERThreshold(predicted_class_master, predicted_values_master, true_class_master,
                                          rogueClasses=rogueClassesMaster)

    # Now get the test set and test the thresholds I just made
    presentSet = whichSet
    tupleOutputTemp = getPredictedClasses_Values_TrueClasses_Labels(datasetFileName=datasetFile,
                                                                    experimentStoreFolder=experimentsFolder,
                                                                    valueMethod=valueMethod,
                                                                    whichSetArg=presentSet)
    (predicted_class_master, predicted_values_master, true_class_master, classLabelsMaster) = tupleOutputTemp
    if not useLabels:
        classLabelsMaster = None
    # using the thresholds from before but now the test set plot what this looks like
    plotThresholds(predicted_class_master,
                   predicted_values_master,
                   true_class_master,
                   eerThresholdsMaster,
                   classLabels=classLabelsMaster,
                   rogueClasses=rogueClassesMaster,
                   setName=setNames[presentSet],
                   valueMethodName=valueMethods[valueMethod],
                   statisticsStoreFolder=statisticsStoreFolder)

    if len(rogueClassesMaster) > 0:
        rogueAnalysis(eerThresholdsMaster,
                      predicted_class_master,
                      predicted_values_master,
                      true_class_master,
                      classLabels=classLabelsMaster,
                      rogueClasses=rogueClassesMaster,
                      setName=setNames[presentSet],
                      valueMethodName=valueMethods[valueMethod],
                      statisticsStoreFolder=statisticsStoreFolder)
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
