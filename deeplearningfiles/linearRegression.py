import cPickle
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T

import matplotlib.pylab as plt
import yaml
import pandas as pd

import ClassificationUtils
import RegressionUtils


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

    datasets, inputs, outputs, max_batch_size = RegressionUtils.load_data(datasetFileName, rogueClasses=rogueClasses)

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

    best_validation_loss = np.inf
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
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch {epoch}/{n_epochs} {epochpercent:3.2f}%, minibatch {minibatch}/{batches}, iter/patience {iteration}/{patience} {doneness:3.2f}%, training error {trainloss:4.5f}, validation error {valloss:4.5f}'.format
                        (
                        epoch=epoch,
                        n_epochs=n_epochs,
                        epochpercent=100 * epoch / float(n_epochs),
                        minibatch=minibatch_index + 1,
                        batches=n_train_batches,
                        trainloss=np.max(minibatch_avg_cost),
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
                    test_score = np.mean(test_losses)

                    print(
                        '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t    New best model of test error {test_score:4.5f}'.format(
                            test_score=test_score))

                    # save the best model
                    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.pkl')
                    with open(modelStoreFilePathFullTemp, 'wb') as f:
                        cPickle.dump(classifier, f)
                    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_modelnp.pkl')
                    with open(modelStoreFilePathFullTemp, 'wb') as f:
                        W = np.array(classifier.W.eval())
                        b = np.array(classifier.b.eval())
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


def getPredicted_Values_TrueValues_Labels(datasetFileName='mnist.pkl.gz', experimentStoreFolder='', whichSetName='valid'):
    # load the saved model
    processedDataFolder = os.path.dirname(datasetFileName)
    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.pkl')
    classifier = cPickle.load(open(modelStoreFilePathFullTemp))

    # compile predictor function
    predict_model_y_values = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred,
    )

    # We can test it on some examples from test test
    datasets, inputs, outputs, max_batch_size = RegressionUtils.load_data(datasetFileName)
    (set_x, set_y) = datasets[0]

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
        datasetStore['gradArray{0}'.format(whichSetName)] = pd.DataFrame(gradArray)
    # np.savetxt(os.path.join(experimentStoreFolder, 'gradArray.csv'), gradArray, delimiter=',')

    set_x = set_x.get_value()
    predicted_y_values = predict_model_y_values(set_x)

    # We can test it on some examples from test test
    datasets, inputs, outputs, max_batch_size = RegressionUtils.load_data(datasetFileName, makeSharedData=False)
    (set_x, set_y) = datasets[0]

    classLabels = ClassificationUtils.getLabelsForDataset(processedDataFolder, datasetFileName)

    return predicted_y_values, set_y, classLabels


def makeStatisticsForModel(experimentsFolder,
                           statisticsStoreFolder,
                           featureParameters,
                           datasetParameters,
                           classifierParameters,
                           useLabels=True,
                           whichSetName='valid',
                           showFigures=True):
    """
    Make staticstics for a model using the features, datset, and classifier given whose model is already made

    :type experimentsFolder: str
    :param experimentsFolder: Location of the pre made model

    :type statisticsStoreFolder: str
    :param statisticsStoreFolder: where im going to put the stats

    :type featureParameters: dict
    :param featureParameters: parameters for the features

    :type datasetParameters: dict
    :param datasetParameters: parameters for the dataset

    :type classifierParameters: dict
    :param classifierParameters: parameters for the classifier

    :type useLabels: bool
    :param useLabels: If labels should be used in charts True or just the class number False

    :type whichSetName: str
    :param whichSetName: Which of the sets to do the thresholding on

    :type showFigures: bool
    :param showFigures: If you want to see the figures now instead of viewing them on disk later (still saves them no matter what)
    """

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
        whichSetName=whichSetName)
    if not useLabels:
        classLabelsMaster = None
    thresholdsMaster = RegressionUtils.getStatistics(predicted_values_master, true_values_master,
                                                     setName=whichSetName,
                                                     statisticsStoreFolder=statisticsStoreFolder,
                                                     datasetParameters=datasetParameters)
    if len(rogueClassesMaster) > 0:
        RegressionUtils.rogueAnalysis(thresholdsMaster, predicted_values_master, true_values_master, classLabels=classLabelsMaster,
                                      rogueClasses=rogueClassesMaster,
                                      setName=whichSetName, statisticsStoreFolder=statisticsStoreFolder)
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
    experimentsFolderMain = ""
    statisticsStoreFolderMain = ""
    makeStatisticsForModel(experimentsFolderMain, statisticsStoreFolderMain, featureParametersDefault, datasetParametersDefault,
                           modelSetParametersDefault)
