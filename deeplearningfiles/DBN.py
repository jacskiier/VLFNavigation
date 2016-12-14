"""
"""
import os
import sys
import timeit

import cPickle
import gzip
import yaml

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from logistic_sgd import LogisticRegression, makeStatisticsForModel
from ClassificationUtils import load_data
from mlp import HiddenLayer
from rbm import RBM
import CreateUtils


# start-snippet-1
class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=(500, 500), n_outs=10):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: tuple
        :param hidden_layers_sizes: intermediate layers size, must contain at least one value (int)

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.hidden_layers_sizes = hidden_layers_sizes

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector
        # of [int] labels
        # end-snippet-1
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = self.hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        inputs=layer_input,
                                        n_in=input_size,
                                        n_out=self.hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            inputs=layer_input,
                            n_visible=input_size,
                            n_hidden=self.hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            inputs=self.sigmoid_layers[-1].output,
            n_in=self.hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

        self.input = self.sigmoid_layers[0].input
        self.y_pred = self.logLayer.y_pred
        self.p_y_given_x = self.logLayer.p_y_given_x

    def changeOutputSize(self, outputs):
        # now I need to rebuild the bottom loglayer to account for new outputs
        self.logLayer = LogisticRegression(
            inputs=self.sigmoid_layers[-1].output,
            n_in=self.hidden_layers_sizes[-1],
            n_out=outputs)
        self.params[-2:] = self.logLayer.params

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, k, persistent_chainList=None):
        """
        Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        :type persistent_chainList: list
        :param persistent_chainList: list of persistent holders for pervious gibbs state look at rbm.get_cost_updates for more info
        """

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        # n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        indexer = 0
        for rbm in self.rbm_layers:
            persistent = persistent_chainList[indexer] if persistent_chainList is not None else None
            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=persistent, k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.Param(learning_rate, default=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)
            indexer += 1

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        """
        Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        """

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                        index * batch_size: (index + 1) * batch_size
                        ],
                self.y: train_set_y[
                        index * batch_size: (index + 1) * batch_size
                        ]
            }
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                        index * batch_size: (index + 1) * batch_size
                        ],
                self.y: test_set_y[
                        index * batch_size: (index + 1) * batch_size
                        ]
            }
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                        index * batch_size: (index + 1) * batch_size
                        ],
                self.y: valid_set_y[
                        index * batch_size: (index + 1) * batch_size
                        ]
            }
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


def test_DBN(dataset='mnist.pkl.gz',
             experimentStoreFolder='',
             rogueClasses=(),
             finetune_lr=0.1,
             pretraining_epochs=100,
             pretrain_lr=0.01,
             k=1,
             training_epochs=1000,
             batch_size=10,
             savedModel=None,
             pretrain=True,
             hidden_layers_sizes=(500, 500),
             patience_increase=10.,
             improvement_threshold=0.990,
             patienceMultiplier=10,
             rngSeed=123,
             ):
    """
    Demonstrates how to train and test a Deep Belief Network.

    This is demonstrated on MNIST.

    :type experimentStoreFolder: basestring
    :param experimentStoreFolder: the folder to store the model after it is trained

    :type rogueClasses: tuple of class outputs (int)
    :param rogueClasses: Tuple of classes to exclude from training to do rogue agalysis

    :type finetune_lr: float
    :param finetune_lr: learning rate used in the finetune stage

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type k: int
    :param k: number of Gibbs steps in CD/PCD

    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    :type batch_size: int
    :param batch_size: the size of a minibatch

    :type savedModel: string
    :param savedModel: location of the previously saved file

    :type pretrain: bool
    :param pretrain: if True will train upper part of DBN if false only trains the lower logistic regression part

    :type hidden_layers_sizes: tuple
    :param hidden_layers_sizes: tuple of the size of each hidden layer

    :type patience_increase: int
    :param patience_increase: wait this much longer when a new best is found  2

    :type improvement_threshold: float
    :param improvement_threshold:  a relative improvement of this much is considered significant 0.995

    :type patienceMultiplier: int
    :param patienceMultiplier: the multiple of the n_train_batches to use to get final patience

    .
    """

    datasets, inputs, outputs, max_batch_size = load_data(dataset, rogueClasses=rogueClasses)

    train_set_x, train_set_y = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    if savedModel is None:
        # numpy random generator
        numpy_rng = numpy.random.RandomState(rngSeed)
        print '... building the model'
        # construct the Deep Belief Network
        dbn = DBN(numpy_rng=numpy_rng, n_ins=inputs,
                  hidden_layers_sizes=hidden_layers_sizes,
                  n_outs=outputs)
    else:
        dbn = cPickle.load(open(savedModel))
        dbn.changeOutputSize(outputs)

    # start-snippet-2
    if pretrain is True or savedModel is None:
        #########################
        # PRETRAINING THE MODEL #
        #########################

        persistent_chainList = []
        for n_hidden in dbn.hidden_layers_sizes:
            persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                         dtype=theano.config.floatX),
                                             borrow=True)
            persistent_chainList.append(persistent_chain)

        print '... getting the pretraining functions'
        # this is where the persistent chain goes if you want to use it
        ## persistent_chainList=persistent_chainList
        pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size,
                                                    k=k)

        print '... pre-training the model'
        start_time = timeit.default_timer()
        ## Pre-train layer-wise
        for i in xrange(dbn.n_layers):
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                                                lr=pretrain_lr))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)

        end_time = timeit.default_timer()
        # end-snippet-2
        print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetuning the model'
    # early-stopping parameters
    patience = patienceMultiplier * n_train_batches  # look as this many examples regardless 4
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    best_iter = 0
    while (epoch < training_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            current_iter = (epoch - 1) * n_train_batches + minibatch_index

            if (current_iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%'
                    % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                    ):
                        patience = max(patience, current_iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = current_iter

                    # save the best model
                    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.pkl')
                    with open(modelStoreFilePathFullTemp, 'w') as f:
                        cPickle.dump(dbn, f)

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= current_iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'obtained at iteration %i, '
            'with test performance %f %%'
        ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))


def predict(dataset='mnist.pkl.gz', numOfValues=10, experimentStoreFolder=''):
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.pkl')
    with open(modelStoreFilePathFullTemp) as f:
        dbn = cPickle.load(f)

    # compile a predictor function
    predict_model = theano.function(
        inputs=[dbn.sigmoid_layers[0].input],
        outputs=dbn.logLayer.y_pred)

    # We can test it on some examples from test test
    datasets, inputs, outputs, max_batch_size = load_data(dataset)
    (test_set_x, test_set_y) = datasets[2]
    test_set_x = test_set_x.get_value()
    # test_set_y = test_set_y.get_value()

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    predicted_values = predict_model(test_set_x[:numOfValues])
    print ("Predicted values for the first {0} examples in test set:".format(numOfValues))
    print predicted_values
    print ("Actual values for the first {0} examples in test set:".format(numOfValues))
    print test_set[1][:numOfValues]


def dbn_parameterized(featureParameters, datasetParameters, classifierParameters, forceRebuildModel=False):
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

    assert classifierParameters['classifierType'] == 'DBN', 'this config wasnt made for a Logistic Regression'

    datasetFile = CreateUtils.getDatasetFile(featureSetName=featureParameters['featureSetName'], datasetName=datasetParameters['datasetName'])
    experimentsFolder = CreateUtils.getExperimentFolder(featureParameters['featureSetName'],
                                                        datasetParameters['datasetName'],
                                                        classifierParameters['classifierType'],
                                                        classifierParameters['classifierSetName'])

    bestModelFilePath = os.path.join(experimentsFolder, 'best_model.pkl')
    if not os.path.exists(bestModelFilePath) or forceRebuildModel:
        test_DBN(dataset=datasetFile,
                 experimentStoreFolder=experimentsFolder,
                 finetune_lr=classifierParameters['finetune_lr'],
                 pretraining_epochs=classifierParameters['pretraining_epochs'],
                 pretrain_lr=classifierParameters['pretrain_lr'],
                 k=classifierParameters['k'],
                 training_epochs=classifierParameters['training_epochs'],
                 batch_size=classifierParameters['batch_size'],
                 hidden_layers_sizes=classifierParameters['hidden_layers_sizes'],
                 patience_increase=classifierParameters['patience_increase'],
                 improvement_threshold=classifierParameters['improvement_threshold'],
                 patienceMultiplier=classifierParameters['patienceMultiplier'],
                 rogueClasses=classifierParameters['rogueClasses']
                 )


if __name__ == '__main__':
    rawDataFolderMain = r"E:\\Users\\Joey\Documents\\Virtual Box Shared Folder\\"

    featureSetNameMain = 'DefaultPatchFeatures'
    datasetNameMain = 'staticLocations'
    classifierType = 'DBN'
    classifierSetNameMain = 'AllClassesDefault'

    (featureParametersDefault,
     datasetParametersDefault,
     modelSetParametersDefault) = CreateUtils.getParameters(featureSetName=featureSetNameMain,
                                                            datasetName=datasetNameMain,
                                                            classifierType=classifierType,
                                                            classifierSetName=classifierSetNameMain)

    dbn_parameterized(featureParametersDefault, datasetParametersDefault, modelSetParametersDefault, forceRebuildModel=True)
    makeStatisticsForModel(featureParametersDefault, datasetParametersDefault, modelSetParametersDefault)
