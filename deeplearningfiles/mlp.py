"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

import yaml
import cPickle
import logistic_sgd
import linearRegression
import ClassificationUtils
import RegressionUtils


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, inputs, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputs: theano.tensor.dmatrix
        :param inputs: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = inputs
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(inputs, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, inputs, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputs: theano.tensor.TensorType
        :param inputs: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # keep track of model input
        self.input = inputs

        self.hiddenLayers = []
        if isinstance(n_hidden, int):
            self.hiddenLayersSizes = (n_hidden,)
        else:
            self.hiddenLayersSizes = tuple(n_hidden)

        for i in range(len(self.hiddenLayersSizes)):
            if i == 0:
                inputSize = n_in
                layerInput = self.input
            else:
                inputSize = self.hiddenLayersSizes[i - 1]
                layerInput = self.hiddenLayers[-1].output

            # Since we are dealing with a one hidden layer MLP, this will translate
            # into a HiddenLayer with a tanh activation function connected to the
            # LogisticRegression layer; the activation function can be replaced by
            # sigmoid or any other nonlinear function
            hiddenLayer = HiddenLayer(
                rng=rng,
                inputs=layerInput,
                n_in=inputSize,
                n_out=self.hiddenLayersSizes[i],
                activation=T.tanh
            )
            self.hiddenLayers.append(hiddenLayer)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = logistic_sgd.LogisticRegression(
            inputs=self.hiddenLayers[-1].output,
            n_in=self.hiddenLayersSizes[-1],
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.logRegressionLayer.W).sum()
        for hiddenLayer in self.hiddenLayers:
            self.L1 = self.L1 + abs(hiddenLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.logRegressionLayer.W ** 2).sum()
        for hiddenLayer in self.hiddenLayers:
            self.L2_sqr = self.L2_sqr + (hiddenLayer.W ** 2).sum()

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.logRegressionLayer.params
        for hiddenLayer in self.hiddenLayers:
            self.params = self.params + hiddenLayer.params
        # end-snippet-3

        self.y_pred = self.logRegressionLayer.y_pred
        self.p_y_given_x = self.logRegressionLayer.p_y_given_x

    # negative log likelihood of the MLP is given by the negative
    # log likelihood of the output of the model, computed in the
    # logistic regression layer
    def errors(self, y):
        return self.logRegressionLayer.errors(y)

    # same holds for the function computing the number of errors
    def negative_log_likelihood(self, y):
        return self.logRegressionLayer.negative_log_likelihood(y)


class MLPRegression(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, inputs, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputs: theano.tensor.TensorType
        :param inputs: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        self.hiddenLayers = []
        if isinstance(n_hidden, int):
            self.hiddenLayersSizes = (n_hidden,)
        else:
            self.hiddenLayersSizes = tuple(n_hidden)

        for i in range(len(self.hiddenLayersSizes)):
            if i == 0:
                inputSize = n_in
                layerInput = inputs
            else:
                inputSize = self.hiddenLayersSizes[i - 1]
                layerInput = self.hiddenLayers[-1].output

            # Since we are dealing with a one hidden layer MLP, this will translate
            # into a HiddenLayer with a tanh activation function connected to the
            # LogisticRegression layer; the activation function can be replaced by
            # sigmoid or any other nonlinear function
            hiddenLayer = HiddenLayer(
                rng=rng,
                inputs=layerInput,
                n_in=inputSize,
                n_out=self.hiddenLayersSizes[i],
                activation=T.tanh
            )
            self.hiddenLayers.append(hiddenLayer)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.linearRegressionLayer = linearRegression.LinearRegression(
            inputs=self.hiddenLayers[-1].output,
            n_in=self.hiddenLayersSizes[-1],
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.linearRegressionLayer.W).sum()
        for hiddenLayer in self.hiddenLayers:
            self.L1 = self.L1 + abs(hiddenLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.linearRegressionLayer.W ** 2).sum()
        for hiddenLayer in self.hiddenLayers:
            self.L2_sqr = self.L2_sqr + (hiddenLayer.W ** 2).sum()

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.linearRegressionLayer.params
        for hiddenLayer in self.hiddenLayers:
            self.params = self.params + hiddenLayer.params
        # end-snippet-3

        self.y_pred = self.linearRegressionLayer.y_pred

        # keep track of model input
        self.input = inputs

    def errors(self, y):
        return self.linearRegressionLayer.errors(y)

    def negative_log_likelihood(self, y):
        return self.linearRegressionLayer.negative_log_likelihood(y)


def test_mlp(dataset='mnist.pkl.gz',
             experimentStoreFolder='',
             rogueClasses=(),
             learning_rate=0.01,
             L1_reg=0.00,
             L2_reg=0.0001,
             n_epochs=1000,
             batch_size=20,
             n_hidden=500,
             patience=10000,
             patience_increase=2,
             improvement_threshold=0.995,
             rngSeed=1234
             ):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    :type experimentStoreFolder: basestring
    :param experimentStoreFolder: the folder to store the model after it is trained

    :type rogueClasses: tuple of class outputs (int)
    :param rogueClasses: Tuple of classes to exclude from training to do rogue agalysis

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type batch_size: int
    :param batch_size: size of batches to train on

    :type n_hidden: int
    :param n_hidden: number of nodes in the hidden layer

    :type patience: int
    :param patience: look as this many examples regardless

    :type patience_increase: int
    :param patience_increase: wait this much longer when a new best is found

    :type improvement_threshold: float
    :param improvement_threshold: a relative improvement of this much is considered significant

    :type rngSeed: int
    :param rngSeed: the random seed to use for the number generator
   """
    datasets, inputs, outputs, max_batch_size = ClassificationUtils.load_data(dataset, rogueClasses=rogueClasses)

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
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    rng = numpy.random.RandomState(rngSeed)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        inputs=x,
        n_in=inputs,
        n_hidden=n_hidden,
        n_out=outputs
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
        ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
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
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters

    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            current_iter = (epoch - 1) * n_train_batches + minibatch_index

            if (current_iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch {epoch}/{n_epochs} {epochpercent:3.2f}%, minibatch {minibatch}/{batches}, iter/patience {iteration}/{patience} {doneness:3.2f}%, training cost {trainloss:4.5f}, validation error {valloss:3.2f}'.format
                        (
                        epoch=epoch,
                        n_epochs=n_epochs,
                        epochpercent=100.0 * epoch / float(n_epochs),
                        minibatch=minibatch_index + 1,
                        batches=n_train_batches,
                        trainloss=numpy.max(minibatch_avg_cost),
                        valloss=100.0 * this_validation_loss,
                        patience=patience,
                        iteration=current_iter,
                        doneness=100.0 * current_iter / float(patience)
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, current_iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = current_iter
                    print("\tNew Best Model")

                    # save the best model
                    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.pkl')
                    with open(modelStoreFilePathFullTemp, 'wb') as f:
                        cPickle.dump(classifier, f)

            if patience <= current_iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.pkl')
    classifier = cPickle.load(open(modelStoreFilePathFullTemp, 'r'))

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            classifier.input: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            classifier.input: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    train_model_error = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            classifier.input: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # test it on the train set
    train_losses = [train_model_error(i) for i in xrange(n_train_batches)]
    train_score = numpy.mean(train_losses)

    # test it on the train set
    valid_losses = [validate_model(i) for i in xrange(n_valid_batches)]
    valid_score = numpy.mean(valid_losses)

    # test it on the test set
    test_losses = [test_model(i) for i in xrange(n_test_batches)]
    test_score = numpy.mean(test_losses)

    print(
        'Optimization complete. Best epochs/Total eposh {best_epoch}/{epoch},  Train error {train_score:3.2f}, Validation error {valid_score:3.2f}, Test error {test_score:3.2f}'.format
        (epoch=epoch,
         best_epoch=best_iter,
         test_score=test_score * 100.0,
         valid_score=valid_score * 100.0,
         train_score=train_score * 100.0)
    )

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


def test_mlp_regression(dataset='mnist.pkl.gz',
                        experimentStoreFolder='',
                        rogueClasses=(),
                        learning_rate=0.01,
                        L1_reg=0.00,
                        L2_reg=0.0001,
                        n_epochs=1000,
                        batch_size=20,
                        n_hidden=500,
                        patience=10000,
                        patience_increase=2,
                        improvement_threshold=0.995,
                        rngSeed=1234
                        ):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    :type experimentStoreFolder: basestring
    :param experimentStoreFolder: the folder to store the model after it is trained

    :type rogueClasses: tuple of class outputs (int)
    :param rogueClasses: Tuple of classes to exclude from training to do rogue agalysis

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type batch_size: int
    :param batch_size: size of batches to train on

    :type n_hidden: int
    :param n_hidden: number of nodes in the hidden layer

    :type patience: int
    :param patience: look as this many examples regardless

    :type patience_increase: int
    :param patience_increase: wait this much longer when a new best is found

    :type improvement_threshold: float
    :param improvement_threshold: a relative improvement of this much is considered significant

    :type rngSeed: int
    :param rngSeed: the random seed to use for the number generator
    """
    datasets, inputs, outputs, max_batch_size = RegressionUtils.load_data(dataset, rogueClasses=rogueClasses)

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
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as matrix
    # [int] labels

    rng = numpy.random.RandomState(rngSeed)

    # construct the MLP class
    classifier = MLPRegression(
        rng=rng,
        inputs=x,
        n_in=inputs,
        n_hidden=n_hidden,
        n_out=outputs
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
        ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
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
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters

    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    try:
        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
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
                        best_iter = currentIteration

                        # test it on the test set
                        test_losses = [test_model(i) for i
                                       in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        print('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t    New best model of test error {test_score:4.5f}'.format(
                            test_score=test_score))

                        # save the best model
                        modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.pkl')
                        with open(modelStoreFilePathFullTemp, 'wb') as f:
                            cPickle.dump(classifier, f)

                if patience <= currentIteration:
                    done_looping = True
                    break
    except KeyboardInterrupt:
        print("Training interupted")

    # batch_size = 1
    # partialYoverPartialX = T.grad(T.sum(classifier.y_pred.norm(2,axis=1)), classifier.input)
    # getAPartial = theano.function(
    #     inputs=[index],
    #     outputs=partialYoverPartialX,
    #     givens={
    #         classifier.input: train_set_x[index * batch_size: (index + 1) * batch_size]
    #     }
    # )
    # gradArray = numpy.zeros(train_set_x.get_value(borrow=True).shape)
    # for idx in range(gradArray.shape[0]):
    #     gradArray[idx,:] = getAPartial(idx)
    # gradStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'gradArray.hf')
    # with pd.HDFStore(gradStoreFilePathFullTemp, 'a') as datasetStore:
    #     datasetStore['gradArray'] = pd.DataFrame(gradArray)

    # k = T.iscalar("k")
    # (outputs, updates) = theano.scan(fn=getAPartial, sequences = [theano.tensor.arange(100)], outputs_info = None)
    # getAllPartials = theano.function(inputs=[], outputs=outputs)
    # getAllPartials(train_set_x.get_value(borrow=True).shape[0])

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


def mlp_parameterized(featureParameters, datasetParameters, classifierParameters, forceRebuildModel=False):
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

    assert classifierParameters['classifierType'] == 'MLP', 'this config wasnt made for a MLP'

    rawDataFolder = datasetParameters['rawDataFolder']

    if os.path.exists(os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'] + '.hf')):
        datasetFile = os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'] + '.hf')
    else:
        datasetFile = os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'],
                                   datasetParameters['datasetName'] + '.pkl.gz')

    experimentsFolder = os.path.join(rawDataFolder, "Data Experiments", featureParameters['featureSetName'], datasetParameters['datasetName'],
                                     classifierParameters['classifierType'], classifierParameters['classifierSetName'])

    bestModelFilePath = os.path.join(experimentsFolder, 'best_model.pkl')
    if not os.path.exists(bestModelFilePath) or forceRebuildModel:
        if classifierParameters['classifierGoal'] == 'classification':
            test_mlp(dataset=datasetFile,
                     experimentStoreFolder=experimentsFolder,
                     learning_rate=classifierParameters['learning_rate'],
                     L1_reg=classifierParameters['L1_reg'],
                     L2_reg=classifierParameters['L2_reg'],
                     n_epochs=classifierParameters['n_epochs'],
                     batch_size=classifierParameters['batch_size'],
                     n_hidden=classifierParameters['n_hidden'],
                     patience=classifierParameters['patience'],
                     patience_increase=classifierParameters['patience_increase'],
                     improvement_threshold=classifierParameters['improvement_threshold'],
                     rngSeed=classifierParameters['rngSeed'],
                     rogueClasses=classifierParameters['rogueClasses']
                     )
        else:
            test_mlp_regression(dataset=datasetFile,
                                experimentStoreFolder=experimentsFolder,
                                learning_rate=classifierParameters['learning_rate'],
                                L1_reg=classifierParameters['L1_reg'],
                                L2_reg=classifierParameters['L2_reg'],
                                n_epochs=classifierParameters['n_epochs'],
                                batch_size=classifierParameters['batch_size'],
                                n_hidden=classifierParameters['n_hidden'],
                                patience=classifierParameters['patience'],
                                patience_increase=classifierParameters['patience_increase'],
                                improvement_threshold=classifierParameters['improvement_threshold'],
                                rngSeed=classifierParameters['rngSeed'],
                                rogueClasses=classifierParameters['rogueClasses']
                                )


if __name__ == '__main__':
    rawDataFolderMain = r"E:\\Users\\Joey\Documents\\Virtual Box Shared Folder\\"

    featureSetNameMain = 'DefaultPatchFeatures'
    datasetNameMain = 'staticLocations'
    classifierType = 'MLP'
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

    mlp_parameterized(featureParametersDefault, datasetParametersDefault, modelSetParametersDefault, forceRebuildModel=True)
    logistic_sgd.makeStatisticsForModel(featureParametersDefault, datasetParametersDefault, modelSetParametersDefault)
